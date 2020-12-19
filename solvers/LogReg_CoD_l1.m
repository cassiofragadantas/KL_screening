function [x, obj, x_it, stop_crit_it, time_it] = LogReg_CoD_l1(A, y, lambda, x0, param)
% LogReg_CoD_l1 is a Coordinate Descent approach to solve the
% l1-regularized Binary Logistic regression problem :
% 
% (1)                  min_(x>=0) sum(log(1+e^(Axi)) - yi Axi) + lambda*norm(x,1)
%
% with yi \in {0,1} containing binary labels. 
% The coordinates xj (j \in [1,...,m]) are update sequentially with a 
% proximal gradient update step with a Newton step-size. The product Ax 
% is incrementally updated at each time.
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
%
%   See also:
%
%   References:
%
% Author: Cassio F. Dantas
% Date: 17 Nov 2020

%% Input parsing

assert(nargin >= 2, 'A (matrix), y (vector) and lambda (scalar) must be provided.');
% A = full(A); y = full(y);%current C implementation does not support sparse matrices

% problem dimensions (n,m)
assert(length(size(A)) == 2, 'Input variable A should be a matrix')
[n, m] = size(A);
assert(all(size(y) == [n 1]),'Input variable y must be a (n x 1) vector')
assert(all(y >= 0),'Input variable y must be non-negative')
assert(isscalar(lambda) & lambda >= 0,'Input variable lambda must non-negative scalar')

% x0
if (nargin < 4) || isempty(x0); x0 = zeros(m,1); end
% x0 = A.'*y; % If x0 not provided
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
f.eval = @(a) sum(log(1 + exp(a)) - y.*a); % Binary logistic cost
% f.eval = @(a,b) sum(log(1 + b) - y.*a); % a=Ax, b=exp(Ax)
g.eval = @(a) lambda*norm(a, 1); % regularization


% ST = @(a,b) max( abs(a) - b, zeros(size(a)) ).* sign(a); % Soft-thresholding - slow
h = @(a) sum(-a.*log(a) - (1-a).*log(1-a)); %binary entropy function (used for dual function) 

tStart = tic;
%% Initialization
x = x0 + 0; % Signal to find (+0 avoid x to be just a pointer to x0)
k = 1;  % Iteration number
stop_crit = Inf; % Difference between solutions in successive iterations

% A2 = A.^2; % Used for hessian calculation - doesn't really accelerate
%Initializations: for first iteration
Ax = A*x;
expAx = exp(Ax);
res =  y -  expAx./(1+expAx); %residual

if param.save_all
    obj = zeros(1, param.MAX_ITER); % Objective function value by iteration
    obj(1) =  f.eval(Ax) + g.eval(x) ;
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
    
    % Update x, Ax and residual (inspired by  EugeneNdiaye/Gap_Safe_Rules cd_logreg_fast.pyx function)
    %C implementation. (attention! the value of x is changed inside the MEX function)
%     NOT IMPLEMENTED!! LogReg_CoD_l1_update(y, A, x, Ax, lambda, param.epsilon, param.epsilon_y); %implemented in C    
    %Matlab implementation (for loop)
    for j_coord = 1:m
        
        %calculate gradient and hessian (step-size)
        grad_j = A(:,j_coord).'*res; %g1
        hessian_j = (A(:,j_coord).^2).'*(expAx./(1+expAx).^2); %g2, A2(:,j_coord)
        
        %Update coordinate xj
%         x(j_coord) = ST(x(j_coord) + grad_j/hessian_j, lambda/hessian_j); %Slow... gradient step and Soft-thresholding
        grad_update = x(j_coord) + grad_j/hessian_j; %gradient step
        x(j_coord) = max( abs(grad_update) - lambda/hessian_j, 0 )*sign(grad_update); %proximal step (Soft-thresholding)
        
        %update Ax and residual
        Ax = Ax + (x(j_coord)-x_old(j_coord))*A(:,j_coord);
        expAx = exp(Ax);
%         tmp = expAx./(1+expAx); tmp2 = tmp./(1+expAx);
        res =  y -  expAx./(1+expAx); %residual
    end


    % Stopping criterion
    if strcmp(param.stop_crit, 'gap') % Duality Gap
        %Dual point
        theta = res/lambda; % Feasible dual point calculation
        ATtheta = A.'*theta; % /!\HEAVY CALCULATION. Also used for screening
        theta = theta/max(1,max(ATtheta)); %dual scaling (or: max(ATtheta))
%         ATtheta = ATtheta/max(ATtheta);
        
        primal = f.eval(Ax) + g.eval(x) ;
        dual = h(y-lambda*theta);
        
    	gap = primal - dual; % gap has to be calculated anyway for GAP_Safe
        stop_crit = gap;
    else %primal variable difference
        stop_crit = norm(x - x_old, 2);
    end
    if param.verbose, stop_crit, end
    
    % Save intermediate results
    if param.save_all
        % Compute the objective function value if necessary
        if ~strcmp(param.stop_crit,'gap'), primal = f.eval(Ax) + g.eval(x); end
        obj(k) =  primal; %f.eval(A*x) + g.eval(x) ;
        % Store iteration values
        if save_x_it, x_it(:, k) = x; end
        % Store stopping criterion
        stop_crit_it(k) = stop_crit;
    end
    if param.save_time
        % Store iteration time
        time_it(k) = toc(tStart); % total elapsed time since the beginning 
    end
end

if param.save_all
    % Trim down stored results
    obj = obj(1:k);
    stop_crit_it = stop_crit_it(1:k);
    if save_x_it, x_it = x_it(:,1:k); end
else
    x_it = []; obj = []; stop_crit_it = [];
end
if param.save_time
    time_it = time_it(1:k);
else
    time_it = []; 
end

end
