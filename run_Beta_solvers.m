%%%%%%%%%%%%%%%%%%%%%%% BETA 1.5 : running solvers %%%%%%%%%%%%%%%%%%%%%%%
% This script is called by the main.m script.
% It requires all simulation parameters to stored on 'param' variable. 
% Variables A, y, lambda and x0 also need to be set.

% Multiplicative Update solver

fprintf('MM solver BETA DIV...\n')
% profile on    
[x_MM, obj_MM, x_it_MM, stop_crit_it_MM, time_it_MM] ...
    = Beta_l1_MM(A,y,lambda,x0_MM,param);
% profile off
% profsave(profile('info'),'./Results/new_Profile_Beta_MM')

% profile on
fprintf('MM solver BETA DIV + Screening ...\n')
[x_MMscr, obj_MMscr, x_it_MMscr, R_it_MMscr, screen_it_MMscr, stop_crit_it_MMscr, time_it_MMscr] ...
    = Beta_l1_MM_GAPSafe(A,y,lambda,x0_MMscr,param,precalc);
% profile off
% profsave(profile('info'),'./Results/new_Profile_Beta_MMscr')

precalc.improving = true;
fprintf('MM solver BETA DIV + Screening w/ refinement...\n')
[x_MMscr_adap, obj_MMscr_adap, x_it_MMscr_adap, R_it_MMscr_adap, screen_it_MMscr_adap, stop_crit_it_MMscr_adap, time_it_MMscr_adap, alpha_redef_MMscr_adap(k_lambda,k_mc)] ...
    = Beta_l1_MM_GAPSafe(A,y,lambda,x0_MMscr,param,precalc);
precalc.improving = false;
precalc.alpha = precalc.alpha0;


x_CoD = 0; x_CoDscr = 0; x_SPIRAL = 0; x_SPIRALscr = 0;