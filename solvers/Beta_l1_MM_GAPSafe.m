function [x, obj, x_it, R_it, screen_it,stop_crit_it, time_it] = Beta_l1_MM_GAPSafe(A, y, lambda, x0, param, precalc)
% Beta_l1_MM a Majoration-minimization approach to solve a non-negative 
% l1-regularized problem that uses the Beta divergence (beta = 1.5) as the 
% data-fidelity term :
% 
% (1)                  min_(x>=0) D_beta(y, Ax) + lambda*norm(x,1)
%
% where lambda is a sparsity regularization term D_beta(a,b) is the beta
% divergence between vectors a and b, which, in turn, can be 
% written in terms of a scalar divergence between each of the entries a_i, 
% b_i of vectors a and b, as follows for beta \in [0,2]:
% 
%   d_beta(a_i,b_i) = 1/(beta*(beta-1)) (a_i^beta + (beta-1)b_i^beta
%                                             - beta a_i b_i^(beta-1) )
%
% This functions therefore seeks a sparse vector x such that y ≈ Ax in the
% beta divergence sense.
%
% THIS IMPLEMENTATION ONLY WORKS WITH BETA=1.5 DUE TO THE COMPUTATION OF
% THE DUAL OBJECTIVE FUNCTION (which is specific for each beta value).
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
% Date: 12 Nov 2020

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
% if ~isfield(param, 'beta'); param.beta = 1.5; end, beta = param.beta;

% objective function
%generic beta
%%f.eval = @(u,v) 1/(beta*(beta-1)) * sum( u.^beta + (beta-1)*v.^beta - beta*u.*v.^(beta-1)); %beta divergence
% f.eval = @(a) 1/(beta*(beta-1)) * sum( (y+param.epsilon_y).^beta + (beta-1)*(a + param.epsilon).^beta ...
%               - beta*(y+param.epsilon_y).*(a + param.epsilon).^(beta-1)); %beta divergence
%beta = 1.5
f.eval = @(a) 4/3 * sum( (y+param.epsilon_y).^1.5 + (1/2)*(a + param.epsilon).^(1.5) ...
              - (3/2)*(y+param.epsilon_y).*sqrt(a + param.epsilon)); %beta divergence
g.eval = @(a) lambda*norm(a, 1); % regularization

tStart = tic;
%% Initialization
x = x0; % Signal to find
k = 1;  % Iteration number
stop_crit = Inf; % Difference between solutions in successive iterations
% screen_vec = false(size(x));
rejected_coords = false(m,1);

% idx_  y0 = (y+param.epsilon_y==0);

Ax = A*x; % For first iteration

if (nargin < 6), precalc = beta_GAP_Safe_precalc(A,y+param.epsilon_y,lambda,param.epsilon); end % Initialize screening rule, if not given as an input

if param.save_all
    obj = zeros(1, param.MAX_ITER); % Objective function value by iteration
    obj(1) =  f.eval(Ax) + g.eval(x) ;
    R_it = zeros(1, param.MAX_ITER); % Safe region radius by iteration.
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
    %Generic beta divergence (convex only for beta >= 1)
%     Axbeta1 = (Ax+param.epsilon).^(beta-1);
%     yAxbeta2 = (y+param.epsilon_y).*(Ax+param.epsilon).^(beta-2);
    %beta=1.5
    Axbeta1 = sqrt(Ax+param.epsilon);
    yAxbeta2 = (y+param.epsilon_y)./Axbeta1; %valid in this particular case
    
    %All together
    ATAxbeta1 = A.'*Axbeta1;
    ATyAxbeta2 = A.'*(yAxbeta2);
    x = x.*(ATyAxbeta2) ./ (ATAxbeta1 + lambda);
    %
    %spliting y=0 and y~=0 - Slow...
% %     A_y0 = A(idx_y0,:); A_ny0 = A(~idx_y0,:);
%     ATAxbeta1_y0 = A(idx_y0,:).'*Axbeta1(idx_y0); %A_y0
%     ATAxbeta1_ny0 = A(~idx_y0,:).'*Axbeta1(~idx_y0); %A_ny0
%     ATyAxbeta2_y0 = A(idx_y0,:).'*yAxbeta2(idx_y0);
%     ATyAxbeta2_ny0 = A(~idx_y0,:).'*yAxbeta2(~idx_y0);
%     x = x.*(ATyAxbeta2_y0 + ATyAxbeta2_ny0) ./ (ATAxbeta1_y0 + ATAxbeta1_ny0 + lambda);
    

    % Update A*x for next iteration
    % (also used in Gap stopping criterion and in GAP Safe rule)
    Ax = A*x; % avoid recalculating this

    % Update dual point     
    %All together
    ATtheta = (1/lambda)*(ATyAxbeta2 - ATAxbeta1);
    theta = (yAxbeta2 - Axbeta1)/lambda;
%     scaling = max( [1, max(ATtheta), max(theta(y~=0)*lambda./precalc.b)] ); %previous implementation
    scaling = max([1, max(ATtheta)]);
    ATtheta = ATtheta/scaling;
    theta = theta/scaling; %dual scaling (or: max(ATtheta))
    %
    %Make sure min(theta(idx_y0))< -sqrt(param.epsilon)/lambda) and b/lambda (theoretically, unecessary on the adaptive approach)
    theta = min(theta, precalc.b/lambda); %this min are almost never change theta
%     theta(~idx_y0) = min(theta(~idx_y0),precalc.b/lambda); %these mins are almost never change theta
%     theta(idx_y0) = min(theta(idx_y0),-sqrt(param.epsilon)/lambda);
    %
%     theta(idx_y0) = theta(idx_y0)*scaling; %previous implementation
    %
    %Recompute ATtheta (since theta might have changed) for screening test
    %Slow!we don't do it, and that's ok because the new ATtheta is necessarily
    %smaller or equals to the previous one and it doesn't seem to affect screening performance.
%     ATtheta = A.'*theta;

    
    % Stopping criterion
    primal = f.eval(Ax) + g.eval(x) ;
    %/!\ only true for beta = 1.5
    dual = sum( (4*(y+param.epsilon_y).^1.5 + 0.5*(lambda*theta).^3 - 0.5*((lambda*theta).^2 + 4*(y+param.epsilon_y)).^1.5 + 3*(lambda*theta).*(y+param.epsilon_y))/3 - param.epsilon*(lambda*theta) );
    
    gap = primal - dual; % gap has to be calculated anyway for GAP_Safe
    
    if strcmp(param.stop_crit, 'gap') % Duality Gap
        stop_crit = gap;
    else %primal variable difference
        stop_crit = norm(x - x_old, 2);
    end
    if param.verbose, stop_crit, end
    
    % Screening
    [screen_vec, radius, precalc] = Beta_GAP_Safe(precalc, lambda, ATtheta, gap, theta, y+param.epsilon_y);

    % Remove screened coordinates (and corresponding atoms)
%     A = A(:,~screen_vec);
%     x = x(~screen_vec);
%     precalc.normA = precalc.normA(~screen_vec);
    A(:,screen_vec) = []; 
    x(screen_vec) = []; 
    precalc.normA(screen_vec) = [];
    
    rejected_coords(~rejected_coords) = screen_vec;
    
    % Save intermediate results
    if param.save_all
        % Compute the objective function value if necessary
        if ~strcmp(param.stop_crit,'gap'), primal = f.eval(Ax) + g.eval(x); end
        obj(k) =  primal; %f.eval(A*x) + g.eval(x) ;
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

if param.save_all
    % Trim down stored results
    obj = obj(1:k);
    R_it = R_it(1:k);
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
