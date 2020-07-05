function [x, obj, x_it, stop_crit_it, time_it, step_it] = SPIRAL(A, y, lambda, x0, param)
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
%           param.TOL       : Stop criterium. When ||x(n) - x(n-1)||_2, 
%                             where n is the iteration number, is less than
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
%       obj  : A vector with the energies E(x(n-1)), where n is the 
%                 iteration number.
%       x_it : matrix containing in its columns the solution estimate at
%              each iteration.
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
    f.eval = @(a) sum((y+param.epsilon).*log((y+param.epsilon)./(a+param.epsilon)) - y + a); % KL distance (fixing first variable as y) with optional epsilon regularizer
end
% Regularization (g)
g.eval = @(a) lambda*norm(a, 1); % regularization
g.prox = @(a, tau) nnSoftThresholding(a, lambda*tau);


tStart = tic;
%% Initialization
x = x0; % Signal to find
n = 1; % Iteration number
sqrty = sqrt(y + param.epsilon);

Ax = A*x; % For first iteration
rho = y - Ax;
if poisson_noise, rho = rho./(Ax+param.epsilon); end
grad = -A.'*rho;        

stop_crit = Inf; % Difference between solutions in successive iterations

% ---- Step-size selection ---- 
% Barzilai-Borwein Scheme
alpha = 1; % Initial alpha
alphamin = 1e-30; alphamax = 1e30; 
% Acceptance criterion
eta = 2;          % Increasing multiplier on alpha
sigma = 0.1;      % between 0 and 1
M = 10; % Number of past iterations considered

% ---- Storage variables ----
obj = zeros(1, param.MAX_ITER);
obj(1) = g.eval(x) + f.eval(Ax);
if param.save_all
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
while (stop_crit > param.TOL) && (n < param.MAX_ITER)
    
    n = n + 1;
    if param.verbose, fprintf('%4d,',n); end

    x_old = x; 
    Ax_old = Ax;
    
    past = (max(n-M,1):(n-1));
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
        
%         rho = y - Ax;
%         if poisson_noise, rho = rho./(Ax+param.epsilon); end
%         grad = -A.'*rho; % heavy    

        % Compute the objective function value
        obj(n) = g.eval(x) + f.eval(Ax);

        % Step-size acceptance criterion
        if (obj(n) <= (maxpastobj - sigma*alpha/2*normsqdx)) ...
             || (alpha >= alphamax)
            accept = true;
        else
            alpha = eta*alpha;
        end
    end  

    % Other updates (using new x iterate)
    Adx = Ax - Ax_old;

    rho = y - Ax;
    if poisson_noise, rho = rho./(Ax+param.epsilon); end
    grad = -A.'*rho; % heavy    
    
    % Stopping criterion
    if strcmp(param.stop_crit, 'gap') % Duality Gap
        %Dual point
        ATtheta = -grad./lambda;
        theta = rho/(lambda*max(1,max(ATtheta))); %dual scaling (or: max(ATtheta)))
        if any(theta<-1/lambda), warning('some theta_i < -1/lambda'); end

        if param.epsilon == 0
            dual = (y(y~=0) + param.epsilon).'*log(1+lambda*theta(y~=0)); % since 0*log(a) = 0 for all a>=0. Avoids 0*log(0) = NaN
        else
            dual = (y + param.epsilon).'*log(1+lambda*theta) - sum(lambda*param.epsilon*theta);
        end        
    	primal = obj(n);
        
    	gap = primal - dual; % gap has to be calculated anyway for GAP_Safe
        if ~isreal(gap) || isnan(gap), warning('gap is complex or NaN!'); end
        stop_crit = real(gap);
    else %primal variable difference
        stop_crit = norm(x - x_old, 2);
    end
    if param.verbose, stop_crit, end

    
    % Save intermediate results
    if param.save_all
        % Store iteration values
%         x_it(:, n) = x;
        if save_x_it, x_it(:, n) = x; end
        % Store stopping criterion
        stop_crit_it(n) = stop_crit;
        % Store step-sizes
        step_it(n) = 1./alpha;
    end
    if param.save_time
        % Store iteration time
        time_it(n) = toc(tStart); % total elapsed time since the beginning 
    end
    
    % Update alpha
    if poisson_noise
        alpha = norm(Adx.*sqrty./(Ax + param.epsilon))^2./normsqdx; 
    else % gaussian
        alpha = norm(Adx)^2./normsqdx;
    end
    alpha = min(alphamax, max(alpha, alphamin));
        
    
end

if param.save_all
    % Trim down stored results
    obj = obj(1:n);
    stop_crit_it = stop_crit_it(1:n);
    step_it = step_it(1:n);
    if save_x_it, x_it = x_it(:,1:n); end
else
    x_it = []; obj = []; stop_crit_it = []; step_it = [];
end
if param.save_time
    time_it = time_it(1:n);
else
    time_it = []; 
end

end

