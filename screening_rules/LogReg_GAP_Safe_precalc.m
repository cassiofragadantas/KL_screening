function precalc = LogReg_GAP_Safe_precalc(A, y, lambda, epsilon, epsilon_y, precalc)
% This function performs some precalculation required as input in function 
% LogReg_GAP_Safe
%
%   Required inputs (problem parameters):
%       A      : (n x m) dictionary matrix.
%       y      : (n x 1) input vector.
%       lambda : regularization parameter (see (1)).
%
%   Output:
%       precal : struct containing the required precalculations as its fields
%           .primal : function handle calculating the primal objective value.
%           .dual   : function handle calculating the primal objective value.
%           .pinv_At_inf : infinity norm of the pseudo-inverse of A.'
%           .normA  : vector containing the l2-norms of the columns of A.
%
% Author: Cassio F. Dantas
% Date: 16 Nov 2020

if (nargin < 6) || isempty(precalc) % recompute everything

    % || pinv(A) ||_1 = || pinv(A.') ||_inf
    % try %there might be a memory issue
    if numel(A)<2.6e8
        pinvA = pinv(full(A)); %A.'*inv(A*A.');

        precalc.pinvA_1 = norm(pinvA,1); % faster than max(sum(abs(pinvA),1))
        %precalc.pinvAi_1 = sum(abs(pinvA),1); %column by column
    % catch
    else
        %Sub-optimal bound on || pinv(A) ||_1 
        warning('Computation of pinv(A) demands too much memory. Using an upper bound for norm(pinv(A),1) instead');
        precalc.pinvA_1 = sqrt(size(A,2))/svds(A,1,'smallest');
        %precalc.pinvAi_1 = repmat(precalc.pinvA_1,1,size(A,1)); %not computed, just replicating the bound
    end
       
    % Norm of the columns of A
    precalc.normA = sqrt(sum(A.^2)).';

end

% alpha (strong concavity constant)
precalc.alpha_global = 4*lambda^2;
precalc.alpha0 = 4*lambda^2/(1 - 4*(min(lambda*precalc.pinvA_1,1/2)-1/2)^2);
precalc.alpha = precalc.alpha0;

precalc.improving = false; %'true' for adaptive alpha, 'false' for fixed alpha

end