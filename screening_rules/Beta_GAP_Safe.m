function [screen_vec, radius, precalc] = Beta_GAP_Safe(precalc, lambda, ATtheta, gap, theta, y) % idx_y0, epsilon)
% Beta_GAP_Safe implements a GAP Safe Screening rule for the l1-regularized 
% problem that uses beta divergence (beta=1.5) as the data-fidelity term :
% 
% (1)                  min_(x>=0) D_beta(y, Ax) + lambda*norm(x,1)
%
% where lambda is a sparsity regularization term D_beta(a,b) is the beta
% divergence (beta=1.5) between vectors a and b, which, in turn, can be 
% written in terms of a scalar divergence between each of the entries a_i, 
% b_i of vectors a and b, as follows taking beta=1.5:
% 
%   d_beta(a_i,b_i) = 1/(beta*(beta-1)) (a_i^beta + (beta-1)b_i^beta
%                                             - beta a_i b_i^(beta-1) )
%
% This functions therefore seeks a sparse vector x such that y ≈ Ax in the
% beta divergence sense.
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
% Date: 16 Nov 2020


% Prevent errors
if gap <= 0, screen_vec = false(size(ATtheta)); radius = 0; return; end

%% Safe sphere definition

radius = sqrt(2*gap/precalc.alpha);


%Teste: feedback loop alpha_r <--> r
improving = precalc.improving; k=0;  %true for adaptive local screening!
while improving
    d = lambda*(theta+radius);
    d = min(d, precalc.b);
%     d(idx_y0) = min(-sqrt(epsilon), d(idx_y0)); %intersecting with SO
%     d(~idx_y0) = min(precalc.b, d(~idx_y0)); %negligeable practical relevance
    alphai = lambda^2*( (d.^2 + 2*y)./sqrt(d.^2 + 4*y) - d );
    % In theory, alphai is always positive, but there can be numerical instabilites (see detailed comment in Beta_GAP_Safe_precalc.m)
    if any(alphai<=0), alphai = lambda^2*(2*y.^2./(d.^3+4*y.*d)); end
    alpha_r = min(alphai);

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
screen_vec = (ATtheta + radius*precalc.normA < 1); % min(radius)

% if k == 0, fprintf('.'), else, fprintf('%d ',k); end
end

