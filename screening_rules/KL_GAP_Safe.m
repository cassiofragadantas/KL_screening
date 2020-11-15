function [screen_vec, radius, precalc] = KL_GAP_Safe(precalc, lambda, ATtheta, gap, theta, y, epsilon_y)
% KL_GAP_Safe implements a GAP Safe Screening rule for the l1-regularized 
% problem that uses Kullback-Leibler divergence as the data-fidelity term :
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
% This function returns a binary vector of the same size as x,
% containing 1 (true) in all screened entries (those identified not 
% belonging to the solution support by the screening test).
% The implemented GAP Safe sphere region is an extension to the KL-problem
% of the standard GAP Safe rule proposed in [1].
%
%   Required inputs:
%       x   : (m x 1) vector, primal solution estimate.
%       A      : (n x m) dictionary matrix.
%       y      : (n x 1) input vector.
%       lambda   : regularization parameter (see (1)).
%
%   Optional inputs:
%       param : Matlab structure with some additional parameters.
%           param.verbose   : Verbose mode (Default: false)
%         
%   Output:problems
%       screen_vec : (m x 1) binary vector (true for screened coordinates)
%
%   Optional outputs (only when param.save_all == true)
%       radius : safe sphere radius
%
%   See also: KL_l1_MM_GAPSafe.m for a solver of problem (1) using this GAP
%             Safe screening rule.
%
%   References:
%       [1] Fercoq, O., Gramfort, A. & Salmon, J. Mind the duality gap: 
%             safer rules for the Lasso. 32nd ICML (2015).
%
% Author: Cassio F. Dantas
% Date: 30 Mar 2020

if (nargin < 7), epsilon_y = 0; end

% Prevent errors
if gap <= 0, screen_vec = false(size(ATtheta)); radius = 0; return; end

%% Safe sphere definition
%==== Other - wrong ====
%radius(1) = sqrt(2*gap)*(1+lambda*precalc.pinvA_1)/(lambda*sqrt(precalc.sumy)); %Wrong! But seems to work!
%radius(1) = gap*(1+lambda*precalc.pinvA_1)/(lambda*precalc.miny); %Wrong! dual "inverse-Lipschitz"
%radius(1) = sqrt(2*gap*(precalc.maxy*precalc.A_inf*precalc.A_1))/(lambda*epsilon); %0) primal Lispchitz gradient (direct extension of GAP journal paper)

%==== Hessian based ====
%Wrong! Does not consider negative theta entries
% radius(1) = sqrt(2*gap/precalc.alpha_all_old);
% radius(2) = sqrt(2*gap/precalc.alpha_coord_old);

%Correct! Try to go back to the old ones, as experimentally they seem safe
% radius(1) = sqrt(2*gap/precalc.alpha_all); %Separated min (not optimal)
radius = sqrt(2*gap/precalc.alpha);


%Teste: feedback loop alpha_r <--> r
improving = precalc.improving; k=0;  %true for adaptive local screening!
while improving
    denominator_r = (1 + lambda*(theta + radius)).^2 ; denominator_r = denominator_r(y~=0);
%     denominator_r = min( denominator_r , precalc.denominator );
    alpha_r = lambda^2 * min( (y(y~=0)+epsilon_y)./(denominator_r) );
    radius_new = sqrt(2*gap/alpha_r);
    improvement = (radius - radius_new);
    if improvement/radius > 1e-1
       radius = radius - improvement;
       precalc.alpha = alpha_r;
       k = k+1;
    else
       improving = false;
    end
end

%% Screening test
screen_vec = (ATtheta + radius*precalc.normA < 1); % min(radius)

% if k == 0, fprintf('.'), else, fprintf('%d ',k); end
end

