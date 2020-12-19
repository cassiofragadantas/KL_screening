function [x, obj, x_it, R_it, screen_it, stop_crit_it, time_it, step_it] = SPIRAL_GAPSafe(A, y, lambda, x0, param, precalc)
% This function implements a particular case of the SPIRAL-TAP algorithm
% proposed in [1] and available at:
%
% http://drz.ac/code/spiraltap/
%
% The implemented algorithm solves the following optimization problem :
% 
% (1)                  min_x (g(x) + f(x)) = min_x E(x)
%
% where g(x) is a non-smooth function, and f(x) is a differentiable 
% function, with Lipschitz constant L.
%
%   Usage:
%       [x, obj] = SPIRAL(g, f, N, x0, param)
%
%   Input:
%       y         :
%       A         :
%       x0        : Initialization of the vector x to be learned. 
%                   (Default: zeros(N,1))
%       param     : Matlab structure with some additional parameters.
%           param.verbose   : Verbose mode (Default: false)
%           param.TOL       : Stop criterium. When ||x(k) - x(k-1)||_2, 
%                             where k is the iteration number, is less than
%                             TOL, we quit the iterative process. 
%                             (Default: 1e-10).
%           param.MAX_ITER  : Stop criterium. When the number of iterations
%                             becomes greater than MAX_ITER, we quit the 
%                             iterative process. 
%                             (Default: 1000).
%         
%   Output:
%       x       : A N-by-1 vector with the solution to the optimization 
%                 problem (1).
%   Optional Outputs:
%       obj  : A vector with the energies E(x(k-1)), where k is the 
%                 iteration number.
%       x_it : matrix containing in its columns the solution estimate at
%              each iteration.
%       R_it : a vector storing safe sphere radius at each iteration.
%       screen_it : binary matrix containing as its columns the screened 
%                   coordinates ate each iteration.
%
%   Example:
%       x = SPIRAL(g, f, N, x0, param)        
%
%   References:
%       [1]	Z. T. Harmany, R. F. Marcia, and R. M. Willett, 
%           "This is SPIRAL-TAP: Sparse Poisson Intensity Reconstruction 
%           ALgorithms – Theory and Practice," IEEE Transactions on Image 
%           Processing, vol. 21, pp. 1084–1096, Mar. 2012.
%
% Author: Cassio F. Dantas
% Date: 16 April 2020

%% Parse input

assert(nargin >= 2, 'A (matrix), y (vector) and lambda (scalar) must be provided.');
% x0 = A.'*y; % If x0 not provided

% problem dimensions (n,m)
assert(length(size(A)) == 2, 'Input variable A should be a matrix')
[n, m] = size(A);
assert(all(size(y) == [n 1]),'Input variable y must be a (n x 1) vector')
assert(all(y >= 0),'Input variable y must be non-negative')
assert(isscalar(lambda) & lambda >= 0,'Input variable lambda must non-negative scalar')

% x0
if (nargin < 4) || isempty(x0); x0 = ones(m,1); end
assert(all(size(x0) == [m 1]),'x0 must be a (m x 1) vector')

% param
if (nargin < 5); param = []; end
if ~isfield(param, 'verbose'); param.verbose = false; end
if ~isfield(param, 'TOL'); param.TOL = 1e-10; end
if ~isfield(param, 'MAX_ITER'); param.MAX_ITER = 10000; end
if ~isfield(param, 'stop_crit'); param.stop_crit = 'difference'; end
if ~isfield(param, 'save_all'); param.save_all = false; end
if ~isfield(param, 'save_time'); param.save_time = true; end
if ~isfield(param, 'epsilon'); param.epsilon = 1e-10; end %
if ~isfield(param, 'epsilon_y'); param.epsilon_y = 0; end
if ~isfield(param, 'euc_dist'); param.euc_dist = false; end

if (param.epsilon==0) && any(y==0)
    warning('SPIRAL solver does not converge with epsilon=0 when there are zero entries in the input vector.')
end

% Data fidelity (f)
if param.euc_dist % gaussian (Euclidean data fidelity)
    poisson_noise = false;
    f.eval = @(a) 0.5*norm(y - a,2)^2; % Euclidean distance
else % default is poisson (KL data fidelity)
    poisson_noise = true;
    if param.epsilon_y == 0
        f.eval = @(a) sum(y(y~=0).*log(y(y~=0)./(a(y~=0)+ param.epsilon))) + sum(- y + a + param.epsilon - param.epsilon_y); % force 0*log(0) = 0 (instead of NaN) 
    else
        f.eval = @(a) sum((y+param.epsilon_y).*log((y+param.epsilon_y)./(a+param.epsilon)) - y + a + param.epsilon - param.epsilon_y); % KL distance (fixing first variable as y) with optional epsilon regularizer
    end
end
% Regularization (g)
g.eval = @(a) lambda*norm(a, 1); % regularization
g.prox = @(a, tau) nnSoftThresholding(a, lambda*tau);


tStart = tic;
%% Initialization
x = x0; % Signal to find
k = 1; % Iteration number
sqrty = sqrt(y + param.epsilon_y);
rejected_coords = false(m,1);

idx_y0 = (y==0);

Ax = A*x; % For first iteration
rho = y - Ax + param.epsilon_y - param.epsilon;
if poisson_noise, rho = rho./(Ax+param.epsilon); end
grad = -A.'*rho;        

stop_crit = Inf; % Difference between solutions in successive iterations

if (nargin < 6), precalc = KL_GAP_Safe_precalc(A,y,lambda,param.epsilon_y); end % Initialize screening rule, if not given as an input

% ---- Step-size selection ---- 
% Barzilai-Borwein Scheme
alpha = 1; % Initial alpha
alphamin = 1e-30; alphamax = 1e30; 
% Acceptance criterion
eta = 2;         % Increasing multiplier on alpha
sigma = 0.1;     % between 0 and 1
M = 10; % Number of past iterations considered

% ---- Storage variables ----
obj = zeros(1, param.MAX_ITER);
obj(1) = g.eval(x) + f.eval(Ax);
if param.save_all
    R_it = zeros(1, param.MAX_ITER); % Safe region radius by iteration. TODO, reduce to 1 radius only
    screen_it = false(m, param.MAX_ITER); % Safe region radius by iteration
    stop_crit_it = zeros(1, param.MAX_ITER);
    stop_crit_it(1) = inf;
    step_it = zeros(1,param.MAX_ITER);

    %Save all iterates only if not too much memory-demanding (<4GB)
    if m*param.MAX_ITER*8 < 4e9
        x_it = zeros(m, param.MAX_ITER); % Solution estimate by iteration
        x_it(:,1) = x0;
        save_x_it = true;
    else
        x_it = 'Not saved. Too memory-demanding.';
        warning('Not saving all solution iterates, since too memory-demanding.')
        save_x_it = false;
    end 
end
if param.save_time
    time_it = zeros(1,param.MAX_ITER); % Elapsed time until end of each iteration
    time_it(1) = toc(tStart); % time for initializations 
end


%% Iterative steps of SPIRAL
while (stop_crit > param.TOL) && (k < param.MAX_ITER)
    
    k = k + 1;
    if param.verbose, fprintf('%4d,',k); end

    x_old = x; 
    Ax_old = Ax;
    
    past = (max(k-M,1):(k-1));
    maxpastobj = max(obj(past));
    accept = false;
    while ~accept
        % Update x
        x = g.prox(x_old - grad./alpha, 1./alpha);
        
        % Other updates (using new x candidate)
        dx = x - x_old;
        Ax = A*x; % heavy
%         Adx = Ax - Ax_old;
        normsqdx = sum( dx(:).^2 );
        
%         rho = y - Ax + param.epsilon_y - param.epsilon;
%         if poisson_noise, rho = rho./(Ax+param.epsilon); end
%         grad = -A.'*rho; % heavy    

        % Compute the objective function value
        obj(k) = g.eval(x) + f.eval(Ax);

        % Step-size acceptance criterion
        if (obj(k) <= (maxpastobj - sigma*alpha/2*normsqdx)) ...
             || (alpha >= alphamax)
            accept = true;
        else
            alpha = eta*alpha;
        end
    end  

    % Other updates (using new x iterate)
    Adx = Ax - Ax_old;

    rho = y - Ax + param.epsilon_y - param.epsilon;
    if poisson_noise, rho = rho./(Ax+param.epsilon); end
    grad = -A.'*rho; % heavy    

    %Dual point
    ATtheta = -grad./lambda;
    scaling = max(1,max(ATtheta));
    theta = rho/(lambda*scaling); %dual scaling (or: max(ATtheta)))
    if scaling ~= 1  % if scaling = 1, no post-processing is necessary
        theta(idx_y0) = -1/lambda; %forcing entries on yi=0 to optimal value
        ATtheta = ATtheta/scaling - precalc.sumA_zero*(scaling-1)/scaling; %correcting ATtheta accordingly
    end
    if any(theta<-1/lambda), warning('some theta_i < -1/lambda'); end
    
    % Stopping criterion
    if param.epsilon_y == 0
        dual = y(y~=0).'*log(1+lambda*theta(y~=0)) - sum(lambda*param.epsilon*theta); % since 0*log(a) = 0 for all a>=0. Avoids 0*log(0) = NaN
    else
        dual = (y + param.epsilon_y).'*log(1+lambda*theta) - sum(lambda*param.epsilon*theta);
    end         
    primal = obj(k);
    gap = primal - dual; % gap has to be calculated anyway for GAP_Safe
    if ~isreal(gap) || isnan(gap), warning('gap is complex or NaN!'); end
    
    if strcmp(param.stop_crit, 'gap') % Duality Gap
        stop_crit = real(gap);
    else %primal variable difference
        stop_crit = norm(x - x_old, 2);
    end
    if param.verbose, stop_crit, end

    % Screening
    [screen_vec, radius, precalc] = KL_GAP_Safe(precalc, lambda, ATtheta, gap,theta, y, param.epsilon_y);

    % Remove screened coordinates (and corresponding atoms)
%     A = A(:,~screen_vec);
%     x = x(~screen_vec);
%     precalc.normA = precalc.normA(~screen_vec);
    A(:,screen_vec) = []; 
    x(screen_vec) = []; 
    precalc.normA(screen_vec) = [];
    precalc.sumA_zero(screen_vec) = [];    
    grad(screen_vec) = [];
    
    rejected_coords(~rejected_coords) = screen_vec;
    
    % Save intermediate results
    if param.save_all
        % Store safe sphere radius
        R_it(k) = radius;
        % Store screening vector per iteration
%         screen_it(:,k) = screen_vec;
        screen_it(:,k) = rejected_coords;
        % Store iteration values
%         x_it(:, k) = x;
        if save_x_it, x_it(~screen_it(:,k), k) = x; end  
        % Store stopping criterion
        stop_crit_it(k) = stop_crit;
        % Store step-sizes
        step_it(k) = 1./alpha;
    end
    if param.save_time
        % Store iteration time
        time_it(k) = toc(tStart); % total elapsed time since the beginning 
    end
    
    % Update alpha
    if poisson_noise
        alpha = norm(Adx.*sqrty./(Ax + param.epsilon))^2./normsqdx; 
    else % gaussian
        alpha = norm(Adx)^2./normsqdx;
    end
    alpha = min(alphamax, max(alpha, alphamin));
        
    
end

% zero-padding solution
x_old = x;
x = zeros(m,1);
x(~rejected_coords) = x_old;

%reseting alpha
precalc.alpha = precalc.alpha_coord;

if param.save_all
    % Trim down stored results
    obj = obj(1:k);
    R_it = R_it(1:k);
    screen_it = screen_it(:,1:k);    
    stop_crit_it = stop_crit_it(1:k);
    step_it = step_it(1:k);
    if save_x_it, x_it = x_it(:,1:k); end
else
    x_it = []; obj = []; R_it = []; screen_it = []; stop_crit_it = []; step_it = [];
end
if param.save_time
    time_it = time_it(1:k);
else
    time_it = []; 
end

end

