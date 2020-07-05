function [x, obj, x_it, R_it, screen_it,stop_crit_it, time_it] = CoD_KL_l1_GAPSafe(A, y, lambda, x0, param, precalc)
% KL_l1_MM a Majoration-minimization approach to solve a non-negative 
% l1-regularized problem that uses the Kullback-Leibler divergence as the 
% data-fidelity term :
% 
% (1)                  min_(x>=0) D_KL(y, Ax) + lambda*norm(x,1)
%
% where lambda is a sparsity regularization term D_KL(a,b) is the Kullback 
% Leibler divergence between vectors a and b, which, in turn, can be 
% written in terms of a scalar divergence between each of the entries a_i, 
% b_i of vectors a and b, as follows:
% 
%   d_KL(a_i,b_i) = a_i log(a_i/b_i) - a_i + b_i
%
% This functions therefore seeks a sparse vector x such that y ≈ Ax in the
% Kullback-Leibler sense.
%
%   Required inputs:
%       A   : (n x m) matrix.
%       y   : (n x 1) input vector.
%       lambda   : regularization parameter (see (1)).
%
%   Optional inputs:
%       x0    : Initialization of the vector x of size (m x 1).
%               (Default: ones(m,1))
%       param : Matlab structure with some additional parameters.
%           param.verbose   : Verbose mode (Default: false)
%           param.TOL       : Stop criterium. When ||x(n) - x(n-1)||_2, 
%                             where n is the iteration number, is less than
%                             TOL, we quit the iterative process. 
%                             (Default: 1e-10).
%           param.MAX_ITER  : Stop criterium. When the number of iterations
%                             becomes greater than MAX_ITER, we quit the 
%                             iterative process. 
%                             (Default: 1000).
%           param.constraint: A function handle imposing some constraint on
%                             x. 
%                             (Default: @(x) x).
%           param.save_all  : Flag enabling to return the solution estimate
%                             at every iteration (in the variable x_it)
%                             and the objective function (in variable obj)
%                             (Default: false).
%         
%   Output:
%       x     : (m x 1) vector, solution of (1)
%
%   Optional outputs (only when param.save_all == true)
%       obj  : A vector with the objective function value at each iteration
%       x_it : matrix containing in its columns the solution estimate at
%              each iteration.
%       R_it : a vector storing safe sphere radius at each iteration.
%       screen_it : binary matrix containing as its columns the screened 
%                   coordinates ate each iteration.
%
%   See also:
%
%   References:
%
% Author: Cassio F. Dantas
% Date: 17 Mar 2020

%% Input parsing

assert(nargin >= 2, 'A (matrix), y (vector) and lambda (scalar) must be provided.');
% x0 = A.'*y; % If x0 not provided
A = full(A); y = full(y);%current C implementation does not support sparse matrices

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
if ~isfield(param, 'save_all'); param.save_all = false; end
if ~isfield(param, 'save_time'); param.save_time = true; end
if ~isfield(param, 'stop_crit'); param.stop_crit = 'difference'; end
if ~isfield(param, 'epsilon'); param.epsilon = 0; end
% nb_portions = 10;
% idx_portions = round(linspace(0,m,nb_portions+1));

% objective function
%f.eval = @(a,b) sum(a.*log(a./b) - a + b); % KL distance
if param.epsilon == 0
    f.eval = @(a) sum(y(y~=0).*log(y(y~=0)./a(y~=0))) + sum(- y + a); % force 0*log(0) = 0 (instead of NaN) 
else
    f.eval = @(a) sum((y+param.epsilon).*log((y+param.epsilon)./(a+param.epsilon)) - y + a); % KL distance (fixing first variable as y) with optional epsilon regularizer
end
% f.eval = sum((y(y~=0)+param.epsilon).*log((y(y~=0)+param.epsilon)./(Ax(y~=0)+param.epsilon))) - sum(y) + sum(Ax); %NOT THE SAME AS ABOVE!
g.eval = @(a) lambda*norm(a, 1); % regularization

tStart = tic;
%% Initialization
x = x0 + 0; % Signal to find (+0 avoid x to be just a pointer to x0)
k = 1;  % Iteration number
stop_crit = Inf; % Difference between solutions in successive iterations
% screen_vec = false(size(x));

Ax = A*x; % For first iteration
if (nargin < 6), precalc = KL_GAP_Safe_precalc(A,y,lambda,param.epsilon); end % Initialize screening rule, if not given as an input
% A2 = A.^2; %used for greedy version of CoD. Uncomment this and l.151-160

if param.save_all
    obj = zeros(1, param.MAX_ITER); % Objective function value by iteration
    obj(1) =  f.eval(Ax) + g.eval(x) ;
    R_it = zeros(4, param.MAX_ITER); % Safe region radius by iteration. TODO, reduce to 1 radius only
    screen_it = false(m, param.MAX_ITER); % Safe region radius by iteration
    stop_crit_it = zeros(1, param.MAX_ITER);
    stop_crit_it(1) = inf;
    
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

%% CoD iterations
while (stop_crit > param.TOL) && (k < param.MAX_ITER)
   
    k = k + 1;
    if param.verbose, fprintf('%4d,',k); end
    
    x_old = x + 0; % +0 avoids x_old to be modified within the MEX function
    
    % Update x and Ax(from Hsieh-Dhillon2011 Coord Descent)
    %C implementation. (attention! the value of x is changed inside the MEX function)
%     x = CoD_KL_l1_update_slow(y+param.epsilon, A.', x, Ax+param.epsilon, lambda); Ax = A*x;
%     [x, Ax] = CoD_KL_l1_update_noeps(y+param.epsilon, A, x, Ax+param.epsilon, lambda); Ax = Ax - param.epsilon; %the Ax 
    CoD_KL_l1_update(y, A, x, Ax, lambda, param.epsilon); %implemented in C   
    %Update coordinates by batches (portions)
%     portion = mod(k-2,nb_portions)+1;
%     CoD_KL_l1_update_portion(y, A, x, Ax, lambda, param.epsilon,idx_portions(portion),idx_portions(portion+1)); %implemented in C    
    %Matlab implementation (for loop)
%     for k_coord = 1:m
%         tmp = (y+param.epsilon)./(Ax+param.epsilon);
%         tmp2 = tmp./(Ax+param.epsilon);
%         g1 = A(:,k_coord).'*(1-tmp);
%         g2 = (A(:,k_coord).^2).'*(tmp2);
%         x(k_coord) = max(0, x(k_coord) -(g1+lambda)/g2);
%         Ax = Ax + (x(k_coord)-x_old(k_coord))*A(:,k_coord);
%     end
    %Updating all coordinates simultaneously would require the inverse Hessian
    % CoD with "best-improvement" coordinate choice (naive implementation)
%     for count = 1:round(sqrt(m)) %1:m
%         tmp = (y+param.epsilon)./(Ax+param.epsilon);
%         tmp2 = tmp./(Ax+param.epsilon);
%         g1_all = A.'*(1-tmp);
%         g2_all = (A2).'*(tmp2);
%         [~, idx] = max(abs(x - max(0, x - (g1_all+lambda)./g2_all)));
%         x_old(idx) = x(idx);
%         x(idx) = max(0, x(idx) -(g1_all(idx)+lambda)/g2_all(idx));
%         Ax = Ax + (x(idx)-x_old(idx))*A(:,idx);
%     end

    % Update dual point
    theta = (y - Ax)./(lambda*(Ax+param.epsilon)); % Feasible dual point calculation
    ATtheta = A.'*theta; % /!\HEAVY CALCULATION. Also used for screening
    theta = theta/max(1,max(ATtheta)); %dual scaling (or: max(ATtheta))
%     ATtheta = ATtheta/max(ATtheta);
    if any(theta<-1/lambda), warning('some theta_i < -1/lambda'); end
    
    % Stopping criterion
    primal = f.eval(Ax) + g.eval(x) ;
    if param.epsilon == 0
        dual = (y(y~=0) + param.epsilon).'*log(1+lambda*theta(y~=0)); % since 0*log(a) = 0 for all a>=0. Avoids 0*log(0) = NaN
    else
        dual = (y + param.epsilon).'*log(1+lambda*theta) - sum(lambda*param.epsilon*theta);
    end    
    
    gap = primal - dual; % gap has to be calculated anyway for GAP_Safe
    
    if strcmp(param.stop_crit, 'gap') % Duality Gap
        stop_crit = gap;
    else %primal variable difference
        stop_crit = norm(x - x_old, 2);
    end
    if param.verbose, stop_crit, end
    
    % Screening
    [screen_vec, radius, precalc] = KL_GAP_Safe(precalc, lambda, ATtheta, gap, param.epsilon, theta, y);
    
    %Test! coordinate-wise limit for dual function 
%     theta_max = max(precalc.A_1/lambda,1)*pinvAi_1.';
%     dual_imax = (y + param.epsilon).*log(1+lambda*theta_max) - (lambda*param.epsilon*theta_max);
    

    % Remove screened coordinates (and corresponding atoms)
%     A = A(:,~screen_vec);
%     x = x(~screen_vec);
%     precalc.normA = precalc.normA(~screen_vec);
    A(:,screen_vec) = []; 
    x(screen_vec) = []; 
    precalc.normA(screen_vec) = [];
%     A2(:,screen_vec) = []; %uncomment this for greedy variant of CoD
    
    % Save intermediate results
    if param.save_all
        % Compute the objective function value if necessary
        if ~strcmp(param.stop_crit,'gap'), primal = f.eval(Ax) + g.eval(x); end
        obj(k) =  primal; %f.eval(A*x) + g.eval(x) ;
        % Store safe sphere radius
        R_it(:,k) = radius;
        % Store screening vector per iteration
%         screen_it(:,k) = screen_vec;
        screen_it(:,k) = screen_it(:,k-1);
        screen_it(~screen_it(:,k-1),k) = screen_vec;
        % Store iteration values
%         x_it(:, k) = x;
        if save_x_it, x_it(~screen_it(:,k), k) = x; end
        % Store stopping criterion
        stop_crit_it(k) = stop_crit;
    end
    if param.save_time
        % Store iteration time
        time_it(k) = toc(tStart); % total elapsed time since the beginning 
    end
end


% zero-padding solution
x_old = x;
x = zeros(m,1);
x(~screen_it(:,k)) = x_old;

%reseting alpha
precalc.alpha = precalc.alpha_coord;

if param.save_all
    % Trim down stored results
    obj = obj(1:k);
    R_it = R_it(:,1:k);
    screen_it = screen_it(:,1:k);
    stop_crit_it = stop_crit_it(1:k);
    if save_x_it, x_it = x_it(:,1:k); end
else
    x_it = []; obj = []; R_it = []; screen_it = []; stop_crit_it = [];
end
if param.save_time
    time_it = time_it(1:k);
else
    time_it = []; 
end

end
