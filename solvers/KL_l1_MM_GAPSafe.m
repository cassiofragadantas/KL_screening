function [x, obj, x_it, R_it, screen_it,stop_crit_it, time_it] = KL_l1_MM_GAPSafe(A, y, lambda, x0, param, precalc)
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
if ~isfield(param, 'epsilon_y'); param.epsilon_y = 0; end

% objective function
%f.eval = @(a,b) sum(a.*log(a./b) - a + b); % KL distance
if param.epsilon_y == 0
    f.eval = @(a) sum(y(y~=0).*log(y(y~=0)./(a(y~=0)+ param.epsilon))) + sum(- y + a + param.epsilon - param.epsilon_y); % force 0*log(0) = 0 (instead of NaN) 
else
    f.eval = @(a) sum((y+param.epsilon_y).*log((y+param.epsilon_y)./(a+param.epsilon)) - y + a + param.epsilon - param.epsilon_y); % KL distance (fixing first variable as y) with optional epsilon regularizer
end
g.eval = @(a) lambda*norm(a, 1); % regularization

tStart = tic;
%% Initialization
x = x0; % Signal to find
k = 1;  % Iteration number
stop_crit = Inf; % Difference between solutions in successive iterations
% screen_vec = false(size(x));
rejected_coords = false(m,1);

idx_y0 = (y==0);

Ax = A*x; % For first iteration
sumA = A.'*ones(n,1);
if (nargin < 6), precalc = KL_GAP_Safe_precalc(A,y,lambda,param.epsilon_y); end % Initialize screening rule, if not given as an input

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

%% MM iterations
while (stop_crit > param.TOL) && (k < param.MAX_ITER)
   
    k = k + 1;
    if param.verbose, fprintf('%4d,',k); end
    
    x_old = x;
    
    % Update x
%     x = x.*(A.'*((y+param.epsilon_y)./(Ax+param.epsilon))) ./ (sumA + lambda);
    yAx = (y+param.epsilon_y)./(Ax+param.epsilon);
    ATyAx = A.'*yAx;
    x = x.*ATyAx./(sumA + lambda);

    % Update A*x for next iteration
    % (also used in Gap stopping criterion and in GAP Safe rule)
    Ax = A*x; % avoid recalculating this

    % Update dual point
    ATtheta = (1/lambda)*(ATyAx - sumA);
    scaling =  max(1,max(ATtheta));
    theta = (yAx - ones(n,1))/(scaling*lambda); %dual scaling (or: max(ATtheta))
%     theta = (y + param.epsilon_y - Ax - param.epsilon)./(lambda*(Ax+param.epsilon)*max(1,max(ATtheta)));
    if scaling ~= 1  % if scaling = 1, no post-processing is necessary
        theta(idx_y0) = -1/lambda; %forcing entries on yi=0 to optimal value
        ATtheta = ATtheta/scaling - precalc.sumA_zero*(scaling-1)/scaling; %correcting ATtheta accordingly
    end
    if any(theta<-1/lambda), warning('some theta_i < -1/lambda'); end
    
    % Stopping criterion
    primal = f.eval(Ax) + g.eval(x) ;
    if param.epsilon_y == 0
        dual = y(y~=0).'*log(1+lambda*theta(y~=0)) - sum(lambda*param.epsilon*theta); % since 0*log(a) = 0 for all a>=0. Avoids 0*log(0) = NaN
    else
        dual = (y + param.epsilon_y).'*log(1+lambda*theta) - sum(lambda*param.epsilon*theta);
    end
    
    gap = primal - dual; % gap has to be calculated anyway for GAP_Safe
    
    if strcmp(param.stop_crit, 'gap') % Duality Gap
        stop_crit = gap;
    else %primal variable difference
        stop_crit = norm(x - x_old, 2);
    end
    if param.verbose, stop_crit, end
    
    % Screening
    [screen_vec, radius, precalc] = KL_GAP_Safe(precalc, lambda, ATtheta, gap, theta, y, param.epsilon_y);

    % Remove screened coordinates (and corresponding atoms)
%     A = A(:,~screen_vec);
%     x = x(~screen_vec);
%     precalc.normA = precalc.normA(~screen_vec);
    A(:,screen_vec) = []; 
    x(screen_vec) = []; 
    precalc.normA(screen_vec) = [];
    precalc.sumA_zero(screen_vec) = [];
    sumA(screen_vec) = [];
    
    rejected_coords(~rejected_coords) = screen_vec;
    
    % Save intermediate results
    if param.save_all
        % Compute the objective function value if necessary
        if ~strcmp(param.stop_crit,'gap'), primal = f.eval(Ax) + g.eval(x); end
        obj(k) =  primal; %f.eval(A*x) + g.eval(x) ;
        % Store safe sphere radius
        R_it(:,k) = radius;
        % Store screening vector per iteration
%         screen_it(:,k) = screen_vec;
        screen_it(:,k) = rejected_coords;
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
x(~rejected_coords) = x_old;

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
