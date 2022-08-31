function precalc = KL_GAP_Safe_precalc(A, y, lambda, epsilon_y, precalc)
% This function performs some precalculation required as input in function 
% KL_GAP_Safe
%
%   Required inputs (problem parameters):
%       A      : (n x m) dictionary matrix.
%       y      : (n x 1) input vector.
%       lambda : regularization parameter (see (1)).
%
%   Output:
%       precal : struct containing the required precalculations as its fields
%           .alpha : local strong concavity bound 
%           .normA : vector containing the l2-norms of the columns of A.
%           .sumA_zero : sum of columns of A restricted to lines i where yi==0
%
% Author: Cassio F. Dantas
% Date: 30 Mar 2020

idx_y0 = (y + epsilon_y == 0);

if (nargin < 5) || isempty(precalc) % recompute everything
    
    % || pinv(A) ||_1 = || pinv(A.') ||_inf - DEPRECATED (used on previous alpha)
%     % try %there might be a memory issue
%     if numel(A)<2.6e8
%         pinvA = pinv(full(A)); %A.'*inv(A*A.');
% 
%         precalc.pinvA_1 = norm(pinvA,1); % faster than max(sum(abs(pinvA),1))
%         precalc.pinvAi_1 = sum(abs(pinvA),1); %column by column
%     % catch
%     else
%         %Sub-optimal bound on || pinv(A) ||_1 
%         warning('Computation of pinv(A) demands too much memory. Using an upper bound for norm(pinv(A),1) instead');
%         precalc.pinvA_1 = sqrt(size(A,2))/svds(A,1,'smallest');
%         precalc.pinvAi_1 = repmat(precalc.pinvA_1,1,size(A,1)); %not computed, just replicating the bound
%     end
% 
%     % || A ||_inf, || A ||_1
%     precalc.A_1 = norm(A, 1); % max(sum(A))

    % Norm of the columns of A
%     precalc.normA = sqrt(sum(A.^2)).';
    precalc.normA = sqrt(sum(A(~idx_y0,:).^2,1)).'; %improved screening! ignoring lines of A s.t. yi = 0
    
    % Sum of columns of A restricted to lines i where yi==0
    precalc.sumA_zero = sum(A(idx_y0,:)).'/lambda;

end

precalc.min_y = min(y(~idx_y0)+epsilon_y);
precalc.sqrt_y = sqrt(y(~idx_y0)+epsilon_y);

% alpha (strong concavity constant)
% precalc.denominator = (1 + max(precalc.A_1,lambda)*precalc.pinvAi_1(~idx_y0).').^2; %old alpha (worse than the one below)
precalc.denominator = min((lambda+sum(A(~idx_y0,:),1))./A(~idx_y0,:),[],2).^2;
precalc.alpha_coord = lambda^2 * min( (y(~idx_y0)+epsilon_y)./precalc.denominator ); %coordinate-wise min
precalc.alpha = precalc.alpha_coord; %This one will be updated over the iterations

precalc.improving = false; %'true' for adaptive alpha, 'false' for fixed alpha

end