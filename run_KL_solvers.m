%%%%%%%%%%%%%%%%%%%%% KL divergence : running solvers %%%%%%%%%%%%%%%%%%%%%
% This script is called by the main.m script.
% It requires all simulation parameters to stored on 'param' variable. 
% Variables A, y, lambda and x0 also need to be set.

%Coordinate Descent - Hsieh2011
fprintf('CoD solver KL...\n')
% profile on    
[x_CoD, obj_CoD, x_it_CoD, stop_crit_it_CoD, time_it_CoD] ...
    = CoD_KL_l1(A,y,lambda,x0_CoD,param);
% profile off
% profsave(profile('info'),'./Results/new_Profile_CoD')

%CoD + Screening
fprintf('CoD solver KL + Screening...\n')
[x_CoDscr, obj_CoDscr, x_it_CoDscr, R_it_CoDscr, screen_it_CoDscr, stop_crit_it_CoDscr, time_it_CoDscr] ...
    = CoD_KL_l1_GAPSafe(A,y,lambda,x0_CoDscr,param,precalc);

precalc.improving = true;
fprintf('CoD solver KL + Screening w/ refinement...\n')
[x_CoDscr_adap, obj_CoDscr_adap, x_it_CoDscr_adap, R_it_CoDscr_adap, screen_it_CoDscr_adap, stop_crit_it_CoDscr_adap, time_it_CoDscr_adap, alpha_redef_CoDscr_adap(k_lambda,k_mc)] ...
    = CoD_KL_l1_GAPSafe(A,y,lambda,x0_CoDscr,param,precalc);
precalc.improving = false;
precalc.alpha = precalc.alpha_coord;

%MM
fprintf('MM solver KL...\n')
[x_MM, obj_MM, x_it_MM, stop_crit_it_MM, time_it_MM] ...
    = KL_l1_MM(A,y,lambda,x0_MM,param);

%MM + Screening
fprintf('MM solver KL + Screening...\n')
[x_MMscr, obj_MMscr, x_it_MMscr, R_it_MMscr, screen_it_MMscr, stop_crit_it_MMscr, time_it_MMscr] ...
    = KL_l1_MM_GAPSafe(A,y,lambda,x0_MMscr,param,precalc);

precalc.improving = true;
fprintf('MM solver KL + Screening w/ refinement...\n')
[x_MMscr_adap, obj_MMscr_adap, x_it_MMscr_adap, R_it_MMscr_adap, screen_it_MMscr_adap, stop_crit_it_MMscr_adap, time_it_MMscr_adap, alpha_redef_MMscr_adap(k_lambda,k_mc)] ...
    = KL_l1_MM_GAPSafe(A,y,lambda,x0_MMscr,param,precalc);
precalc.improving = false;
precalc.alpha = precalc.alpha_coord;

%SPIRAL
fprintf('SPIRAL solver KL...\n')
[x_SPIRAL, obj_SPIRAL, x_it_SPIRAL, stop_crit_it_SPIRAL, time_it_SPIRAL, step_it_SPIRAL] ...
    = SPIRAL(A, y, lambda, x0_SPIRAL, param);

%SPIRAL + Screening
fprintf('SPIRAL solver KL + Screening...\n')
[x_SPIRALscr, obj_SPIRALscr, x_it_SPIRALscr, R_it_SPIRALscr, screen_it_SPIRALscr, stop_crit_it_SPIRALscr, time_it_SPIRALscr] ...
    = SPIRAL_GAPSafe(A,y,lambda,x0_SPIRALscr,param,precalc);

precalc.improving = true;
fprintf('SPIRAL solver KL + Screening w/ refinement...\n')
[x_SPIRALscr_adap, obj_SPIRALscr_adap, x_it_SPIRALscr_adap, R_it_SPIRALscr_adap, screen_it_SPIRALscr_adap, stop_crit_it_SPIRALscr_adap, time_it_SPIRALscr_adap, alpha_redef_SPIRALscr_adap(k_lambda,k_mc)] ...
    = SPIRAL_GAPSafe(A,y,lambda,x0_SPIRALscr,param,precalc);
precalc.improving = false;
precalc.alpha = precalc.alpha_coord;

%FISTA
% fprintf('FISTA solver...\n')
% g.eval = @(x) lambda*norm(x, 1);
% g.prox = @(x, tau) nnSoftThresholding(x, lambda*tau);
% f.eval = @(x) sum((y+epsilon).*log((y+epsilon)./(A*x+epsilon)) - y + A*x);
% f.grad = @(x) A.'*((A*x-y)./(A*x+epsilon)); % sum(A).' - A.'*((y+epsilon)./(A*x+epsilon));
% % f.grad = @(a) a.^(beta-2).*(a-y); % for generic beta-divergence
% if strcmp(param.stop_crit,'gap') % function handle for dual objective must be provided
%     %Dual point and objective
%     param.theta = @(x) ((y - A*x)./(lambda*(A*x+epsilon)))/max(A.'* ((y - A*x)./(lambda*(A*x+epsilon)))); %dual scaling 
% %         param.dual = @(x) (y+epsilon).'*log(1+lambda*x) - sum(lambda*x*epsilon);
%     param.dual = @(x) (y(y~=0) + param.epsilon).'*log(1+lambda*x(y~=0)) - sum(lambda*param.epsilon*x(y~=0)); % since 0*log(a) = 0 for all a>=0. Avoids 0*log(0) = NaN
% end
% 
% if param.euc_dist
%     f.eval = @(x) 0.5*norm(y - A*x,2)^2; % Euclidean distance
%     f.grad = @(x) A.'*(A*x-y);
% end
% 
% [x_FISTA, obj_FISTA, x_it_FISTA, stop_crit_it_FISTA, time_it_FISTA ] ...
%     = FISTA(g, f, m, x0_FISTA, param);
% %     assert(~isnan(stop_crit_it_FISTA(end)), 'FISTA diverged') %FISTA diverges sometimes

% Empirical verification of screening safety
if any(x_MM(screen_it_MMscr(:,end)) > 10*m*max(stop_crit_it_MM(end),eps)), warning('SCREENING FAILED!'); end
% if any(screen_it_MMscr(:,end)) % estimating factor (about 1.25*m in 100 runs)
%     gap_factor(k_lambda) = max(gap_factor(k_lambda), max(x_MM(screen_it_MMscr(:,end))) / stop_crit_it_MM(end));
% end
if any(x_SPIRAL(screen_it_SPIRALscr(:,end)) > 10*m*max(stop_crit_it_SPIRAL(end),eps)), warning('SCREENING FAILED!'); end

%dual
theta = (1/lambda)  * (y-A*x_SPIRAL)./(A*x_SPIRAL + param.epsilon);
if param.epsilon == 0
    dual = (y(y~=0) + param.epsilon).'*log(1+lambda*theta(y~=0)) - sum(lambda*param.epsilon*theta(y~=0)); % since 0*log(a) = 0 for all a>=0. Avoids 0*log(0) = NaN
else
    dual = (y + param.epsilon).'*log(1+lambda*theta) - sum(lambda*param.epsilon*theta);
end    
%TO EXPLORE!! %hist(theta./y) % close to constant
%assert(all(A.'*theta >= -1),'ASSUMPTION THAT (At*theta >= -1) FAILED'); %IT FAILS at small lambda (e.g. lambda_rel = 1e-12)
%assert(all(A(y~=0,:).'*theta(y~=0) >= -1),'ASSUMPTION THAT (At*theta >= -1) FAILED (even excluding y==0 coordinates)'); %IT FAILS at small lambda (e.g. lambda_rel = 1e-12)
