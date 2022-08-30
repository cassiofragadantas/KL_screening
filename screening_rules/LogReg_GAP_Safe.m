function [screen_vec, radius, precalc, trace] = LogReg_GAP_Safe(precalc, lambda, ATtheta, gap, theta, y)
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

% Prevent errors
if gap <= 0, screen_vec = false(size(ATtheta)); radius = 0; trace.nb_it = 0; return; end

%% Safe sphere definition

radius = sqrt(2*gap/precalc.alpha);

%Teste: feedback loop alpha_r <--> r
improving = precalc.improving; k=0;  %true for adaptive local screening!
trace.nb_it_all = 0; %TO DELETE
if improving == 2 % Analytic variant
%     radius_an = radius; precalc.alpha_an = precalc.alpha; precalc.alpha_old = precalc.alpha; %to delete
    t =  min(abs(lambda*theta - y + 1/2));
    if gap < 2*t^2 % t - lambda*radius > 0
        numerator = -4*t*lambda*sqrt(2*gap) + 2*lambda*sqrt(2*gap + 1 -4*t^2); %apparently numerator is always positive
        if numerator < 0, error('WEIRD, numerator<0. This should not happen, theoretically.'), end
        alpha_star = (numerator/(1 -4*t^2))^2;
%     if t - lambda*sqrt(2*gap/alpha_star) > 0
        precalc.alpha = alpha_star;
        precalc.alpha = max(alpha_star,precalc.alpha);
        radius = sqrt(2*gap/precalc.alpha);
    end
%     end
else % Iterative variant %TODO uncomment
    radius = sqrt(2*gap/precalc.alpha);
    t =  min(abs(lambda*theta - y + 1/2));
    while improving
        tmp =  max(0, t - lambda*radius);
        alpha_r = 4*lambda^2/(1 - 4*tmp^2);

        radius_new = sqrt(2*gap/alpha_r);
        improvement = (radius - radius_new);
        if improvement > 0
           radius = radius - improvement;
           precalc.alpha = alpha_r;
           k = k+1;
        end
        trace.nb_it_all = trace.nb_it_all + 1; %TO DELETE
        %stopping criterion
        if improvement/radius < 1e-1, improving = false; end %1e-1
    end
% TO DELETE
% t - lambda*radius > 0
% radius - radius_an
% precalc.alpha - precalc.alpha_an
% precalc.alpha - precalc.alpha_old
end

%% Screening test
screen_vec = (abs(ATtheta) + radius*precalc.normA < 1); % min(radius)

% if k == 0, fprintf('.'), else, fprintf('%d ',k); end
trace.nb_it = k;
end

