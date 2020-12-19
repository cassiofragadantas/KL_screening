%%%%%%%%%%%%%%%%%% Logistic Regression : running solvers %%%%%%%%%%%%%%%%%%
% This script is called by the main.m script.
% It requires all simulation parameters to stored on 'param' variable. 
% Variables A, y, lambda and x0 also need to be set.

%Running times for Leukemia (lambda/lambda_max - time): 
% 0.1 - 8s, 0.01 - 20s , 0.001 - 20s. Total estimated for 100 lambdas
% from 1 to 1e-3 : 100*15s = 1500s. Quite close to the 1250s from GAP
% Safe paper (Cython code).

%Coordinate Descent

fprintf('CoD solver LogReg...\n')
% profile on    
[x_CoD, obj_CoD, x_it_CoD, stop_crit_it_CoD, time_it_CoD] ...
    = LogReg_CoD_l1(A,y,lambda,x0_CoD,param);
% profile off
% profsave(profile('info'),'./Results/new_Profile_LogReg_CoD')

% profile on
fprintf('CoD solver LogReg + Screening ...\n')
[x_CoDscr, obj_CoDscr, x_it_CoDscr, R_it_CoDscr, screen_it_CoDscr, stop_crit_it_CoDscr, time_it_CoDscr] ...
    = LogReg_CoD_l1_GAPSafe(A,y,lambda,x0_CoDscr,param,precalc);
% profile off
% profsave(profile('info'),'./Results/new_Profile_LogReg_CoDscr')

precalc.improving = true;
fprintf('CoD solver LogReg + Screening w/ refinement ...\n')
[x_CoDscr_adap, obj_CoDscr_adap, x_it_CoDscr_adap, R_it_CoDscr_adap, screen_it_CoDscr_adap, stop_crit_it_CoDscr_adap, time_it_CoDscr_adap] ...
    = LogReg_CoD_l1_GAPSafe(A,y,lambda,x0_CoDscr,param,precalc);    
precalc.improving = false;

precalc.alpha = precalc.alpha_global;
fprintf('CoD solver LogReg + Screening global ...\n')
[x_CoDscr_global, obj_CoDscr_global, x_it_CoDscr_global, R_it_CoDscr_global, screen_it_CoDscr_global, stop_crit_it_CoDscr_global, time_it_CoDscr_global] ...
    = LogReg_CoD_l1_GAPSafe(A,y,lambda,x0_CoDscr,param,precalc); 
precalc.alpha = precalc.alpha0;


x_MM = 0; x_MMscr = 0; x_SPIRAL = 0; x_SPIRALscr = 0;