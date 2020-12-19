function [screen_vec, radius, precalc] = LogReg_GAP_Safe(precalc, lambda, ATtheta, gap, theta, y, epsilon_y)
% LogReg_GAP_Safe implements a GAP Safe Screening rule for the l1-regularized 
% Logistic Regression problem.
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
%       lambda   : regularization parameter.
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
%   See also: LogReg_CoD_l1_GAPSafe.m for a solver of the aimed problem 
%             using this GAP Safe screening rule.
%
%   References:
%       [1] Fercoq, O., Gramfort, A. & Salmon, J. Mind the duality gap: 
%             safer rules for the Lasso. 32nd ICML (2015).
%       [2] Dantas, C.F., Soubies, E. & Fevotte, C. Expanding Boundaries of
%             GAP Safe Screenig (2020).
%
% Author: Cassio F. Dantas
% Date: 16 Nov 2020

if (nargin < 7), epsilon_y = 0; end
y = y + epsilon_y;

% Prevent errors
if gap <= 0, screen_vec = false(size(ATtheta)); radius = 0; return; end

%% Safe sphere definition

radius = sqrt(2*gap/precalc.alpha);


%Teste: feedback loop alpha_r <--> r
improving = precalc.improving; k=0;  %true for adaptive local screening!
while improving
    tmp =  max(0, min(abs(lambda*theta - y + 1/2)) - lambda*radius);
    alpha_r = 4*lambda^2/(1 - 4*tmp^2);

    radius_new = sqrt(2*gap/alpha_r);
    improvement = (radius - radius_new);
    if improvement > 0
       radius = radius - improvement;
       precalc.alpha = alpha_r;
       k = k+1;
    end
    %stopping criterion
    if improvement/radius < 1e-1, improving = false; end
end

%% Screening test
screen_vec = (abs(ATtheta) + radius*precalc.normA < 1); % min(radius)

% if k == 0, fprintf('.'), else, fprintf('%d ',k); end
end

