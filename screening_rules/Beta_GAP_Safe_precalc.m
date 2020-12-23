function precalc = Beta_GAP_Safe_precalc(A, y, lambda, epsilon, precalc)
% This function performs some precalculation required as input in function 
% Beta_GAP_Safe
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

if (nargin < 5) || isempty(precalc) % recompute everything
    
    % || a_j ||_1
    precalc.a_1 = sum(A); %l1-norm (sum) of columns of A.
    
    % Norm of the columns of A
    precalc.normA = sqrt(sum(A.^2)).';

    % || y ||_1.5
    precalc.normy15 = norm(y,1.5);

end

%else: only lambda has changed
% y = y+epsilon_y; %y=y(y~=0);
c = - ((4*precalc.normy15^1.5 + 2*(size(y,1)-1)*epsilon^1.5 + 3*epsilon)/(1-3*epsilon))^(1/3); %corresponds to lambda*c in the paper
b = min( (lambda - c.*precalc.a_1)./A, [], 2) + c; %A(y~=0,:)
b = min(b, (y-epsilon)/sqrt(epsilon));

% alpha (strong concavity constant)
alphai = lambda^2*( (b.^2 + 2*y)./sqrt(b.^2 + 4*y) - b );
if any(alphai<=0) % If b is too big compared to y, there can be numerical instabilities even if in theory alphai is always positive.
    % To handle it, we use the Taylor expansion sqrt(1+x)≃ 1 + x/2 for (x<<1) as follows:
    % alphai = lambda^2*b.*( (1 + 2*y./b^2)./sqrt(1 + 4*y./b^2) - 1 );
    % Setting u = (1 + 2*y./b^2)./sqrt(1 + 4*y./b^2)
    % Then u = sqrt(u^2) = sqrt(1 + 4*y.^2/(b.^4+4*y.*b.^2)) = 1 + 2*y.^2/(b.^4+4*y.*b.^2)
    % Therefore alphai = lambda^2*b.*(u-1)
    alphai = lambda^2*(2*y.^2./(b.^3+4*y.*b));
end
% if any(y==0), alphai = [alphai; 2*lambda^2*sqrt(epsilon)]; end %only when epsilon_y=0

precalc.alpha0 = min(alphai);
precalc.alpha = precalc.alpha0;

precalc.b = b; %useful in the adaptive approach

precalc.improving = false; %'true' for adaptive alpha, 'false' for fixed alpha

end