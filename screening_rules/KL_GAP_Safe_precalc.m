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
%           .primal : function handle calculating the primal objective value.
%           .dual   : function handle calculating the primal objective value.
%           .pinv_At_inf : infinity norm of the pseudo-inverse of A.'
%           .normA  : vector containing the l2-norms of the columns of A.
%
% Author: Cassio F. Dantas
% Date: 30 Mar 2020

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

    % || A ||_inf, || A ||_1
    %precalc.A_inf = norm(A, inf); % max(sum(A,2))
    precalc.A_1 = norm(A, 1); % max(sum(A))

    % Norm of the columns of A
%     precalc.normA = sqrt(sum(A.^2)).';
    precalc.normA = sqrt(sum(A(y~=0,:).^2)).'; %improved screening! ignoring lines of A s.t. yi = 0
    
    % Sum of columns of A restricted to lines i where yi==0
    precalc.sumA_zero = sum(A(y==0,:)).'/lambda;

    if epsilon_y == 0 %neglecting zero entries in y
        y=y(y~=0); 
%         precalc.pinvAi_1 = precalc.pinvAi_1(y~=0); 
    end 

    % Sum of yi
    %precalc.sumy = sum(y+epsilon_y); %or abs(y+epsilon_y) just for precaution

    % Max of yi
    %precalc.maxy = max(y+epsilon_y);

    % Min of yi
    precalc.miny = min(y+epsilon_y);
end

% if epsilon_y == 0, y=y(y~=0); end %neglecting zero entries in y
y=y(y~=0);

% alpha (strong concavity constant)
% precalc.alpha_all = lambda^2 * min(y+epsilon_y)/(1 + max(precalc.A_1,lambda)*precalc.pinvA_1).^2;  %separated min (less performant)
% precalc.denominator = (1 + max(precalc.A_1,lambda)*precalc.pinvAi_1(y~=0).').^2; %old alpha (better than alpha_all but worse than the one below)
precalc.denominator = min((lambda+sum(A(y~=0,:),1))./A(y~=0,:),[],2).^2;
precalc.alpha_coord = lambda^2 * min( (y+epsilon_y)./precalc.denominator ); %coordinate-wise min
precalc.alpha = precalc.alpha_coord; %This one will be updated over the iterations

precalc.improving = false; %'true' for adaptive alpha, 'false' for fixed alpha

% Primal and Dual objective functions handle
% Used only if not already calculated by Gap stopping criterion
% precalc.primal = @(Ax,x) sum((y+epsilon_y).*log((y+epsilon_y)./(Ax+epsilon)) - y + Ax) + lambda*norm(x,1);
% precalc.dual = @(theta) (y+epsilon_y).'*log(1+lambda*theta) - sum(lambda*epsilon*theta);
end