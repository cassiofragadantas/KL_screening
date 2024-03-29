clear all
addpath('./solvers/')
addpath('./screening_rules/')
addpath('./datasets/')
rng_seed = 10; % 0 for no seed
if rng_seed, rng(rng_seed), fprintf('\n\n /!\\/!\\ RANDOM SEED ACTIVATED /!\\/!\\\n\n'); end

%==== User parameters ====
mc_it = 1; % Number of noise realizations
param.save_all = true; %Solvers will store all intermediate variables
param.verbose = false;
param.stop_crit = 'gap'; 
param.TOL_rel = 1e-7; %-1 to run until max iteration (1e-8*min(nnz(y),n-nnz(y))/n)
normalized_TOL = false; %Normalizes to the data scaling
param.MAX_ITER = 1e5;
epsilon = 1e-6; % >0 to avoid singularity at 0 on KL divergence.
param.epsilon = epsilon;
param.epsilon_y = 0; %epsilon
%CHECK STANDARD REGULARIZATION PATH IN GAP SAFE JOURNAL PAPER (100 POINTS FROM 10-3 TO 1)
lambdas_rel = 1e-2;  %logspace(-3,0,22);  %[1e-1 1e-2 1e-3];  %logspace(-3,0,100);  % regularization (relative to lambda_max) 

warm_start = false;

param.euc_dist = false; %If true, runs SPIRAL and FISTA in standard lasso problem (w/ Euclidean distance)

save_coordinates_evolution = false; %True for MM synthetic experiment

%==== Problem parameters ====
problem_type = 'logistic'; % Options: 'logistic', 'KL', 'beta15'
noise_type = 'none'; % Options: 'poisson', 'gaussian_std', 'gaussian_snr', otherwise: no noise.
exp_type = 'synthetic'; % Options: 'synthetic', or some real dataset 
                         % Count data (KL): 
                         %     'TasteProfile', '20newsgroups', 'NIPSpapers', 'Encyclopedia', 'MNIST'
                         % Classification (Logistic): 
                         %     'Leukemia', 'Colon-cancer'
                         %     'Leukemia_mod', 'Colon-cancer_mod'
                         % Hyperspectral (beta-div): 
                         %     'Urban', 'Urban_subsampled'
                         %     'Cuprite', 'Cuprite_subsampled', 'Moffett', 'Madonna'
                         %     'Cuprite_USGS-lib', 'Urban_USGS-lib' (using USGS spectral library as dictionary)
param.screen_period = 1; % Screening is performed every screen_period iterations of the solver (default is 1)
param.normalizeA = true;
            
if strcmp(noise_type,'gaussian_std')
    sigma = 0.1; %noise standard deviation
elseif strcmp(noise_type,'gaussian_snr')
    snr_db = 15;
end

if strcmp(exp_type,'synthetic')
    n = 100; %10;
    m = 1000; %20;
    
    regenerate_A = false; % Used in the synthetic experiments only
    sp_ratio = 0.1; %Non-zero entries ratio in x_golden
    nnz_x = sp_ratio*m; % number of nonzero entries of the solution x             
    
    %generating A
    if strcmp(problem_type,'logistic')
        A = randn(n,m);
    else %KL and beta=1.5
        A = abs(randn(n,m));
%         A = abs(sprand(n,m,0.02,0.01)); assert(~any(sum(A) == 0)) %density, condition number
    end
    A = A./repmat(sqrt(sum(A.^2)),n,1);% normalizing columns  

else
    sp_ratio = 'UnDef'; %Non-zero entries ratio
    load_dataset
end

% Intilization x0
x0_CoD = ones(m,1); %zeros(m,1);
x0_CoDscr = ones(m,1); %zeros(m,1);
x0_MM = ones(m,1);
x0_MMscr = ones(m,1);
x0_SPIRAL = ones(m,1);
x0_SPIRALscr = ones(m,1);
%x0_FISTA = ones(m,1);
if strcmp(problem_type,'logistic')
    x0_CoD = zeros(m,1);x0_CoDscr = zeros(m,1);
end

%gap_factor = zeros(size(lambdas_rel)); % ration between duality gap and the biggest screened entry in MM

%==== Storage variables ====
if strcmp(problem_type,'KL')
    CoD = true; MM = true; PG = true;
elseif strcmp(problem_type,'beta15')
    CoD = false; MM = true; PG = false;
elseif strcmp(problem_type,'logistic')
    CoD = true; MM = false; PG = false;
else
    error('\nType of problem not implemented! Check problem_type variable.')
end

if CoD
    l1_norm_CoD = zeros(size(lambdas_rel)); %L1 norm
    sparsity_ratio_CoD = zeros(size(lambdas_rel)); %Sparsity
    %Screening ratio
    screen_ratio_CoD = zeros(size(lambdas_rel)); 
    screen_ratio_CoD_adap = zeros(size(lambdas_rel)); 
    if strcmp(problem_type,'logistic'), screen_ratio_CoD_global = zeros(size(lambdas_rel)); end
    if true %mc_it == 1 %Screening ratio per iteration
        screen_ratio_it_CoD = zeros(param.MAX_ITER,length(lambdas_rel)); 
        screen_ratio_it_CoD_adap = zeros(param.MAX_ITER,length(lambdas_rel));
        if strcmp(problem_type,'logistic'), screen_ratio_it_CoD_global = zeros(param.MAX_ITER,length(lambdas_rel)); end
    end
    %Nb iterations
    nb_iter_CoD = zeros(length(lambdas_rel),mc_it); 
    nb_iter_CoDscr = zeros(length(lambdas_rel),mc_it);
    nb_iter_CoDscr_adap = zeros(length(lambdas_rel),mc_it);
    if strcmp(problem_type,'logistic'), nb_iter_CoDscr_global = zeros(length(lambdas_rel),mc_it); end
    %Time
    time_CoD = zeros(length(lambdas_rel),mc_it);
    time_CoDscr = zeros(length(lambdas_rel),mc_it);
    time_CoDscr_adap = zeros(length(lambdas_rel),mc_it);
    if strcmp(problem_type,'logistic'), time_CoDscr_global = zeros(length(lambdas_rel),mc_it); end
    %time: various convergence thresholds
    time_CoD_various = zeros(5, length(lambdas_rel),mc_it);
    time_CoDscr_various = zeros(5, length(lambdas_rel),mc_it);
    time_CoDscr_adap_various = zeros(5, length(lambdas_rel),mc_it);    
    if strcmp(problem_type,'logistic'), time_CoDscr_global_various = zeros(5, length(lambdas_rel),mc_it); end
    %time: per iteration
    if mc_it == 1 
        time_it_CoD_all = cell(size(lambdas_rel));
        time_it_CoDscr_all = cell(size(lambdas_rel));
        time_it_CoDscr_adap_all = cell(size(lambdas_rel));
        if strcmp(problem_type,'logistic'), time_it_CoDscr_global_all = cell(size(lambdas_rel)); end
    end
    %convergence per iteration
    if mc_it == 1 
        stop_crit_it_CoD_all = cell(size(lambdas_rel));
        stop_crit_it_CoDscr_all = cell(size(lambdas_rel));
        stop_crit_it_CoDscr_adap_all = cell(size(lambdas_rel));
        if strcmp(problem_type,'logistic'), stop_crit_it_CoDscr_global_all = cell(size(lambdas_rel)); end
    end
    %Other - Screening
    alpha_redef_CoDscr_adap = zeros(length(lambdas_rel),mc_it);
    screen_nb_iter_CoDscr_adap = zeros(length(lambdas_rel),mc_it);
    screen_time_CoDscr = zeros(length(lambdas_rel),mc_it);
    screen_time_CoDscr_adap = zeros(length(lambdas_rel),mc_it);
    if strcmp(problem_type,'logistic'), screen_time_CoDscr_global = zeros(length(lambdas_rel),mc_it); end    
end
if MM
    l1_norm_MM = zeros(size(lambdas_rel)); %L1 norm
    sparsity_ratio_MM = zeros(size(lambdas_rel)); %Sparsity
    %Screening ratio
    screen_ratio_MM = zeros(size(lambdas_rel)); 
    screen_ratio_MM_adap = zeros(size(lambdas_rel)); 
    if true %mc_it == 1 %Screening ratio per iteration
        screen_ratio_it_MM = zeros(param.MAX_ITER,length(lambdas_rel)); 
        screen_ratio_it_MM_adap = zeros(param.MAX_ITER,length(lambdas_rel)); 
    end 
    nb_iter_MM = zeros(length(lambdas_rel),mc_it);
    nb_iter_MMscr = zeros(length(lambdas_rel),mc_it);
    nb_iter_MMscr_adap = zeros(length(lambdas_rel),mc_it);
    time_MM = zeros(length(lambdas_rel),mc_it);
    time_MMscr = zeros(length(lambdas_rel),mc_it);
    time_MMscr_adap = zeros(length(lambdas_rel),mc_it);
    time_MM_various = zeros(5, length(lambdas_rel),mc_it);
    time_MMscr_various = zeros(5, length(lambdas_rel),mc_it);
    time_MMscr_adap_various = zeros(5, length(lambdas_rel),mc_it);
    if mc_it == 1 %time: per iteration
        time_it_MM_all = cell(size(lambdas_rel));
        time_it_MMscr_all = cell(size(lambdas_rel));
        time_it_MMscr_adap_all = cell(size(lambdas_rel));
    end
    if mc_it == 1 %convergence per iteration
        stop_crit_it_MM_all = cell(size(lambdas_rel));
        stop_crit_it_MMscr_all = cell(size(lambdas_rel));
        stop_crit_it_MMscr_adap_all = cell(size(lambdas_rel));
    end
    %Other - Screening
    alpha_redef_MMscr_adap = zeros(length(lambdas_rel),mc_it);
    screen_nb_iter_MMscr_adap = zeros(length(lambdas_rel),mc_it);
    screen_time_MMscr = zeros(length(lambdas_rel),mc_it);
    screen_time_MMscr_adap = zeros(length(lambdas_rel),mc_it);
end
if PG   
    l1_norm_SPIRAL = zeros(size(lambdas_rel)); %L1 norm
    sparsity_ratio_SPIRAL = zeros(size(lambdas_rel)); %Sparsity
    %Screening ratio
    screen_ratio_SPIRAL = zeros(size(lambdas_rel)); 
    screen_ratio_SPIRAL_adap = zeros(size(lambdas_rel)); 
    if true %mc_it == 1 %Screening ratio per iteration
        screen_ratio_it_SPIRAL = zeros(param.MAX_ITER,length(lambdas_rel));
        screen_ratio_it_SPIRAL_adap = zeros(param.MAX_ITER,length(lambdas_rel));
    end
    nb_iter_SPIRAL = zeros(length(lambdas_rel),mc_it);
    nb_iter_SPIRALscr = zeros(length(lambdas_rel),mc_it);
    nb_iter_SPIRALscr_adap = zeros(length(lambdas_rel),mc_it);
    time_SPIRAL = zeros(length(lambdas_rel),mc_it);
    time_SPIRALscr = zeros(length(lambdas_rel),mc_it);
    time_SPIRALscr_adap = zeros(length(lambdas_rel),mc_it);
    %time_FISTA = zeros(size(lambdas_rel));
    time_SPIRAL_various = zeros(5, length(lambdas_rel),mc_it);
    time_SPIRALscr_various = zeros(5, length(lambdas_rel),mc_it);  
    time_SPIRALscr_adap_various = zeros(5, length(lambdas_rel),mc_it);  
    if mc_it == 1 %time: per iteration
        time_it_SPIRAL_all = cell(size(lambdas_rel));
        time_it_SPIRALscr_all = cell(size(lambdas_rel));
        time_it_SPIRALscr_adap_all = cell(size(lambdas_rel));
    end
    if mc_it == 1 %convergence per iteration
        stop_crit_it_SPIRAL_all = cell(size(lambdas_rel));
        stop_crit_it_SPIRALscr_all = cell(size(lambdas_rel));
        stop_crit_it_SPIRALscr_adap_all = cell(size(lambdas_rel));
    end
    %Reconstruction error - TODO
    rec_err_euc_SPIRAL = zeros(size(lambdas_rel));
    rec_err_KL_SPIRAL = zeros(size(lambdas_rel));
    %Other - Screening
    alpha_redef_SPIRALscr_adap = zeros(length(lambdas_rel),mc_it);
    screen_nb_iter_SPIRALscr_adap = zeros(length(lambdas_rel),mc_it);
    screen_time_SPIRALscr = zeros(length(lambdas_rel),mc_it);
    screen_time_SPIRALscr_adap = zeros(length(lambdas_rel),mc_it);    
end

input_error_euc = 0; input_error_KL = 0;
time_precalc = zeros(length(lambdas_rel),mc_it);
normy2 = zeros(1,mc_it);

% ==== Main loop ====
for k_mc = 1:mc_it
k_mc, tic

%Generate problem and input data
if strcmp(exp_type,'synthetic')    
    if strcmp(problem_type,'logistic')
        if regenerate_A
            A = randn(n,m);
            A = A./repmat(sqrt(sum(A.^2)),n,1);% normalizing columns
        end
        y_orig = round(rand(n,1));
    else %KL and beta=1.5
        if regenerate_A
            A = abs(randn(n,m));
            A = A./repmat(sqrt(sum(A.^2)),n,1);% normalizing columns
        end
        param.ymult = 5; % a low value (~5) will lead y with lots of zero entries on a synthetic experiment with poisson noise.
        x_golden = param.ymult*sprand(m,1,sp_ratio); %entries of x_golden are uniformly distributed on [0 ymult]
        y_orig = A*abs(full(x_golden)); %y = abs(randn(n,1));
        y_orig = y_orig/norm(y_orig);
    end    
else %Real dataset
    if any(strcmp(exp_type,{'Cuprite_USGS-lib', 'Urban_USGS-lib'}))
        y_orig = Y_orig(:,k_mc);
        y_orig = y_orig./norm(y_orig); %normalize columns
    elseif ~any(strcmp(exp_type,{'Leukemia','Leukemia_mod','Colon-cancer','Colon-cancer_mod'}))
        %Draw a random entry to be the input signal
        if param.normalizeA, y_orig = y_orig/norm(y_orig); end
        A = [A(:,1:(idx_y-1)) y_orig A(:,(idx_y):end)];
        idx_y = randi(m+1);
        y_orig = A(:,idx_y);
        A = A(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
        % Uncomment line below for not normalizing y (so that it has integer entries)
%         if param.normalizeA && strcmp(problem_type,'KL'), y_orig = round(y_orig*normA(idx_y)); end
    end
end

%Add noise
if strcmp(noise_type,'poisson')
    y = poissrnd(full(y_orig));
elseif strcmp(noise_type,'gaussian_std') % gaussian with std
    y = y_orig + sigma*randn(size(y_orig));
elseif strcmp(noise_type,'gaussian_snr') % gaussin with snr
    y = y_orig + (10^(-snr_db/20)*norm(y_orig)/sqrt(n))*randn(n,1);
else % no noise
    y = y_orig;
end
assert(all(y>=0),'Input signal should only have positive entries.')
normy2(k_mc) = norm(y,2)^2;

%Keep only non-zero entries in y
% idx = (y~=0); y=y(idx); A = A(idx,:); n = length(y);

precalc = [];  % To ensure that it will be fully recalculated

%Maximum regularization
if strcmp(problem_type,'KL')
    lambda_max = max(A.'*(y+param.epsilon_y-param.epsilon))/epsilon;
elseif strcmp(problem_type,'beta15')
    lambda_max = max(A.'*(y+param.epsilon_y-param.epsilon))/sqrt(epsilon);
elseif strcmp(problem_type,'logistic')
    lambda_max = max(abs(A.'*(y-0.5)));
else
    error('\nType of problem not implemented! Check problem_type variable.')
end

if lambda_max == Inf %case epsilon=0 with KL or beta=1.5 (not recommended, since some solvers are not guaranteed to converge)
    lambda_max = max(A.'*y); %In case of epsilon=0, lambda_max actually makes no sense. So, just using that of the standard lasso
end


lambdas = lambdas_rel*lambda_max; 

%Stopping criterion
if normalized_TOL 
    if strcmp(problem_type,'logistic')
        param.TOL = param.TOL_rel*min(sum(y),n-sum(y))/n; 
    else 
        param.TOL = param.TOL_rel*norm(y,2)^2; 
    end
else
    param.TOL = param.TOL_rel;
end


for k_lambda = 1:length(lambdas)
    lambda = lambdas(k_lambda);
    fprintf('\n ---- Regularization parameter %d / %d ----\n',k_lambda, length(lambdas))

    %Precalculations for Screening tests
    tPrecalc = tic;
    if strcmp(problem_type,'KL')
        precalc = KL_GAP_Safe_precalc(A,y,lambda,param.epsilon_y, precalc);
    elseif strcmp(problem_type,'beta15')
        precalc = Beta_GAP_Safe_precalc(A,y+param.epsilon_y,lambda,param.epsilon);
    elseif strcmp(problem_type,'logistic')
        precalc = LogReg_GAP_Safe_precalc(A,y,lambda);
    else
        error('\nType of problem not implemented! Check problem_type variable.')
    end
    time_precalc(k_lambda,k_mc) = toc(tPrecalc);
   
    %Run solvers
    if strcmp(problem_type,'KL')
        run_KL_solvers
    elseif strcmp(problem_type,'beta15')
        run_Beta_solvers
    elseif strcmp(problem_type,'logistic')
        run_LogReg_solvers
    else
        error('\nType of problem not implemented! Check problem_type variable.')
    end
    

    % Warm start
    if warm_start
        if CoD
            x0_CoD = x_CoD;
            x0_CoDscr = x_CoDscr;
        end
        if MM
            x0_MM = x_MM;
            x0_MMscr = x_MMscr;
        end
        if PG
            x0_SPIRAL = x_SPIRAL;
            x0_SPIRALscr = x_SPIRALscr;
            %x0_FISTA = x_FISTA;
        end
    end
    
    %% Storing results
    %L1-norm
    if CoD, l1_norm_CoD(k_lambda) = l1_norm_CoD(k_lambda) + norm(x_CoD,1)/mc_it; end
    if MM, l1_norm_MM(k_lambda) = l1_norm_MM(k_lambda) + norm(x_MM,1)/mc_it; end
    if PG, l1_norm_SPIRAL(k_lambda) = l1_norm_SPIRAL(k_lambda) + norm(x_SPIRAL,1)/mc_it; end

    % Sparsity
    if CoD, sparsity_ratio_CoD(k_lambda) = sparsity_ratio_CoD(k_lambda) + sum(x_CoD < eps)/(m*mc_it); end    
    if MM, sparsity_ratio_MM(k_lambda) = sparsity_ratio_MM(k_lambda) + sum(x_MM< 1e-10)/(m*mc_it); end %x_MM<=max(1e-10*max(x_MM),eps)
    if PG, sparsity_ratio_SPIRAL(k_lambda) = sparsity_ratio_SPIRAL(k_lambda) + sum(x_SPIRAL< eps)/(m*mc_it); end %x_SPIRAL<=1e-10*max(x_SPIRAL)
    
    % Screen ratio
    if CoD
        screen_ratio_CoD(k_lambda) = screen_ratio_CoD(k_lambda) + sum(screen_it_CoDscr(:,end))/(m*mc_it); 
        screen_ratio_CoD_adap(k_lambda) = screen_ratio_CoD_adap(k_lambda) + sum(screen_it_CoDscr_adap(:,end))/(m*mc_it); 
        if strcmp(problem_type,'logistic'), screen_ratio_CoD_global(k_lambda) = screen_ratio_CoD_global(k_lambda) + sum(screen_it_CoDscr_global(:,end))/(m*mc_it); end
    end
    if MM 
        screen_ratio_MM(k_lambda) = screen_ratio_MM(k_lambda) + sum(screen_it_MMscr(:,end))/(m*mc_it); 
        screen_ratio_MM_adap(k_lambda) = screen_ratio_MM_adap(k_lambda) + sum(screen_it_MMscr_adap(:,end))/(m*mc_it); 
    end
    if PG
        screen_ratio_SPIRAL(k_lambda) = screen_ratio_SPIRAL(k_lambda) + sum(screen_it_SPIRALscr(:,end))/(m*mc_it); 
        screen_ratio_SPIRAL_adap(k_lambda) = screen_ratio_SPIRAL_adap(k_lambda) + sum(screen_it_SPIRALscr_adap(:,end))/(m*mc_it); 
    end
    %per iteration - padding array until MAX_ITER
    if true %mc_it == 1
        if CoD
            screen_ratio_it_CoD(:,k_lambda) = screen_ratio_it_CoD(:,k_lambda) + padarray(sum(screen_it_CoDscr).',size(screen_ratio_it_CoD,1)-length(stop_crit_it_CoDscr),'replicate','post')/(m*mc_it); 
            screen_ratio_it_CoD_adap(:,k_lambda) = screen_ratio_it_CoD_adap(:,k_lambda) + padarray(sum(screen_it_CoDscr_adap).',size(screen_ratio_it_CoD_adap,1)-length(stop_crit_it_CoDscr_adap),'replicate','post')/(m*mc_it); 
            if strcmp(problem_type,'logistic'), screen_ratio_it_CoD_global(:,k_lambda) = screen_ratio_it_CoD_global(:,k_lambda) + padarray(sum(screen_it_CoDscr_global).',size(screen_ratio_it_CoD_global,1)-length(stop_crit_it_CoDscr_global),'replicate','post')/(m*mc_it); end
        end
        if MM
            screen_ratio_it_MM(:,k_lambda) = screen_ratio_it_MM(:,k_lambda) + padarray(sum(screen_it_MMscr).',size(screen_ratio_it_MM,1)-length(stop_crit_it_MMscr),'replicate','post')/(m*mc_it); 
            screen_ratio_it_MM_adap(:,k_lambda) = screen_ratio_it_MM_adap(:,k_lambda) + padarray(sum(screen_it_MMscr_adap).',size(screen_ratio_it_MM_adap,1)-length(stop_crit_it_MMscr_adap),'replicate','post')/(m*mc_it); 
        end
        if PG
            screen_ratio_it_SPIRAL(:,k_lambda) = screen_ratio_it_SPIRAL(:,k_lambda) + padarray(sum(screen_it_SPIRALscr).',size(screen_ratio_it_SPIRAL,1)-length(stop_crit_it_SPIRALscr),'replicate','post')/(m*mc_it); 
            screen_ratio_it_SPIRAL_adap(:,k_lambda) = screen_ratio_it_SPIRAL_adap(:,k_lambda) + padarray(sum(screen_it_SPIRALscr_adap).',size(screen_ratio_it_SPIRAL_adap,1)-length(stop_crit_it_SPIRALscr_adap),'replicate','post')/(m*mc_it); 
        end
    end
    
    % Other screening results
    if CoD
        alpha_redef_CoDscr_adap(k_lambda,k_mc) = trace_CoDscr_adap.count_alpha;
        screen_nb_iter_CoDscr_adap(k_lambda,k_mc) = sum(trace_CoDscr_adap.screen_nb_it) + sum(trace_CoDscr_adap.screen_nb_it==0);
        screen_time_CoDscr(k_lambda,k_mc) = sum(trace_CoDscr.screen_time_it);
        screen_time_CoDscr_adap(k_lambda,k_mc) = sum(trace_CoDscr_adap.screen_time_it);
        if strcmp(problem_type,'logistic'), screen_time_CoDscr_global(k_lambda,k_mc) = sum(trace_CoDscr_global.screen_time_it); end
    end
    if MM
        alpha_redef_MMscr_adap(k_lambda,k_mc) = trace_MMscr_adap.count_alpha;
        screen_nb_iter_MMscr_adap(k_lambda,k_mc) = sum(trace_MMscr_adap.screen_nb_it) + sum(trace_MMscr_adap.screen_nb_it==0);
        screen_time_MMscr(k_lambda,k_mc) = sum(trace_MMscr.screen_time_it);
        screen_time_MMscr_adap(k_lambda,k_mc) = sum(trace_MMscr_adap.screen_time_it);     
    end
    if PG
        alpha_redef_SPIRALscr_adap(k_lambda,k_mc) = trace_SPIRALscr_adap.count_alpha;
        screen_nb_iter_SPIRALscr_adap(k_lambda,k_mc) = sum(trace_SPIRALscr_adap.screen_nb_it) + sum(trace_SPIRALscr_adap.screen_nb_it==0);
        screen_time_SPIRALscr(k_lambda,k_mc) = sum(trace_SPIRALscr.screen_time_it);
        screen_time_SPIRALscr_adap(k_lambda,k_mc) = sum(trace_SPIRALscr_adap.screen_time_it);        
    end
    
    % Nb Iterations
    if CoD
        nb_iter_CoD(k_lambda,k_mc) = length(stop_crit_it_CoD);
        nb_iter_CoDscr(k_lambda,k_mc) = length(stop_crit_it_CoDscr);
        nb_iter_CoDscr_adap(k_lambda,k_mc) = length(stop_crit_it_CoDscr_adap);
        if strcmp(problem_type,'logistic'), nb_iter_CoDscr_global(k_lambda,k_mc) = length(stop_crit_it_CoDscr_global); end
    end
    if MM 
        nb_iter_MM(k_lambda,k_mc) = length(stop_crit_it_MM);
        nb_iter_MMscr(k_lambda,k_mc) = length(stop_crit_it_MMscr);
        nb_iter_MMscr_adap(k_lambda,k_mc) = length(stop_crit_it_MMscr_adap);
    end
    if PG
        nb_iter_SPIRAL(k_lambda,k_mc) = length(stop_crit_it_SPIRAL);
        nb_iter_SPIRALscr(k_lambda,k_mc) = length(stop_crit_it_SPIRALscr);
        nb_iter_SPIRALscr_adap(k_lambda,k_mc) = length(stop_crit_it_SPIRALscr_adap);
    end
    
    % Time
    if CoD
        time_CoD(k_lambda,k_mc) = time_it_CoD(end);
        time_CoDscr(k_lambda,k_mc) = time_it_CoDscr(end);
        time_CoDscr_adap(k_lambda,k_mc) = time_it_CoDscr_adap(end);
        if strcmp(problem_type,'logistic'), time_CoDscr_global(k_lambda,k_mc) = time_it_CoDscr_global(end); end
    end
    if MM 
        time_MM(k_lambda,k_mc) = time_it_MM(end);
        time_MMscr(k_lambda,k_mc) = time_it_MMscr(end);
        time_MMscr_adap(k_lambda,k_mc) = time_it_MMscr_adap(end);
    end
    if PG
        time_SPIRAL(k_lambda,k_mc) = time_it_SPIRAL(end);
        time_SPIRALscr(k_lambda,k_mc) = time_it_SPIRALscr(end);
        time_SPIRALscr_adap(k_lambda,k_mc) = time_it_SPIRALscr_adap(end);
    end
    %time_FISTA(k_lambda) = time_it_FISTA(end)/mc_it;
    
    % Time for various convergence thresholds
    for kk = 1:5
        if CoD
            time_CoD_various(kk,k_lambda,k_mc) = min([inf time_it_CoD(find(stop_crit_it_CoD<10^(5-kk)*param.TOL,1))]);
            time_CoDscr_various(kk,k_lambda,k_mc) = min([inf time_it_CoDscr(find(stop_crit_it_CoDscr<10^(5-kk)*param.TOL,1))]);
            time_CoDscr_adap_various(kk,k_lambda,k_mc) = min([inf time_it_CoDscr_adap(find(stop_crit_it_CoDscr_adap<10^(5-kk)*param.TOL,1))]);
            if strcmp(problem_type,'logistic'), time_CoDscr_global_various(kk,k_lambda,k_mc) = min([inf time_it_CoDscr_global(find(stop_crit_it_CoDscr_global<10^(5-kk)*param.TOL,1))]); end
        end
        if MM
            time_MM_various(kk,k_lambda,k_mc) = min([inf time_it_MM(find(stop_crit_it_MM<10^(5-kk)*param.TOL,1))]);
            time_MMscr_various(kk,k_lambda,k_mc) = min([inf time_it_MMscr(find(stop_crit_it_MMscr<10^(5-kk)*param.TOL,1))]);
            time_MMscr_adap_various(kk,k_lambda,k_mc) = min([inf time_it_MMscr_adap(find(stop_crit_it_MMscr_adap<10^(5-kk)*param.TOL,1))]);
        end
        if PG
            time_SPIRAL_various(kk,k_lambda,k_mc) = min([inf time_it_SPIRAL(find(stop_crit_it_SPIRAL<10^(5-kk)*param.TOL,1))]);
            time_SPIRALscr_various(kk,k_lambda,k_mc) = min([inf time_it_SPIRALscr(find(stop_crit_it_SPIRALscr<10^(5-kk)*param.TOL,1))]);
            time_SPIRALscr_adap_various(kk,k_lambda,k_mc) = min([inf time_it_SPIRALscr_adap(find(stop_crit_it_SPIRALscr_adap<10^(5-kk)*param.TOL,1))]);
        end
    end
    
    %time: per iteration
    if mc_it == 1
        if CoD
            time_it_CoD_all{k_lambda} = time_it_CoD;
            time_it_CoDscr_all{k_lambda} = time_it_CoDscr;
            time_it_CoDscr_adap_all{k_lambda} = time_it_CoDscr_adap;
            if strcmp(problem_type,'logistic'), time_it_CoDscr_global_all{k_lambda} = time_it_CoDscr_global; end
        end
        if MM
            time_it_MM_all{k_lambda} = time_it_MM;
            time_it_MMscr_all{k_lambda} = time_it_MMscr;
            time_it_MMscr_adap_all{k_lambda} = time_it_MMscr_adap;            
        end
        if PG
            time_it_SPIRAL_all{k_lambda} = time_it_SPIRAL;
            time_it_SPIRALscr_all{k_lambda} = time_it_SPIRALscr;
            time_it_SPIRALscr_adap_all{k_lambda} = time_it_SPIRALscr_adap;            
        end
    end
    
    %convergence: per iteration
    if mc_it == 1
        if CoD
            stop_crit_it_CoD_all{k_lambda} = stop_crit_it_CoD;
            stop_crit_it_CoDscr_all{k_lambda} = stop_crit_it_CoDscr;
            stop_crit_it_CoDscr_adap_all{k_lambda} = stop_crit_it_CoDscr_adap;
            if strcmp(problem_type,'logistic'), stop_crit_it_CoDscr_global_all{k_lambda} = stop_crit_it_CoDscr_global; end
        end
        if MM
            stop_crit_it_MM_all{k_lambda} = stop_crit_it_MM;
            stop_crit_it_MMscr_all{k_lambda} = stop_crit_it_MMscr;
            stop_crit_it_MMscr_adap_all{k_lambda} = stop_crit_it_MMscr_adap;            
        end
        if PG
            stop_crit_it_SPIRAL_all{k_lambda} = stop_crit_it_SPIRAL;
            stop_crit_it_SPIRALscr_all{k_lambda} = stop_crit_it_SPIRALscr;
            stop_crit_it_SPIRALscr_adap_all{k_lambda} = stop_crit_it_SPIRALscr_adap;            
        end
    end
    
    % Reconstruction error
    KL_err = @(x) sum((y_orig+epsilon).*log((y_orig+epsilon)./(x+epsilon)) - y_orig + x);
    if PG, rec_err_euc_SPIRAL(k_lambda) = rec_err_euc_SPIRAL(k_lambda) + norm(y_orig - A*x_SPIRAL)/mc_it;end
    if PG, rec_err_KL_SPIRAL(k_lambda) = rec_err_KL_SPIRAL(k_lambda) +  KL_err(A*x_SPIRAL)/mc_it;end
        
end
input_error_euc = input_error_euc + norm(y_orig - y)/mc_it;
input_error_KL = input_error_KL + KL_err(y)/mc_it;
toc
end

% Saving results
clear A
if ~exist('./Results/', 'dir'), mkdir('./Results/'); end
if save_coordinates_evolution %save coordinates evolution separately (heavy)
    save(['./Results/new_main_' problem_type '_screening_test_mcIt' num2str(mc_it) '_' num2str(length(lambdas)) 'lambdas' num2str(lambdas_rel(1)) '-' num2str(lambdas_rel(end)) '_tol' num2str(param.TOL) '_eps' num2str(epsilon) '_n' num2str(n) 'm' num2str(m) '_sp' num2str(sp_ratio) '_wstart' num2str(warm_start) '_seed' num2str(rng_seed) '_CoordEvolution.mat'], 'x_it_MM', 'x_it_MMscr', 'x_it_MMscr_adap', 'screen_it_MMscr', 'screen_it_MMscr_adap')
end

clear x_it_CoD x_it_CoDscr x_it_CoDscr_adap x_it_CoDscr_global x_it_MM x_it_MMscr x_it_MMscr_adap x_it_SPIRAL x_it_SPIRALscr x_it_SPIRALscr_adap % cleaning variables not used for plots
if length(lambdas) == 1 && mc_it ==  1 % Medium weight (keep all but screening coordinates evolution)
    save(['./Results/new_main_' problem_type '_screening_test_mcIt' num2str(mc_it) '_' num2str(length(lambdas)) 'lambdas' num2str(lambdas_rel(1)) '-' num2str(lambdas_rel(end)) '_tol' num2str(param.TOL) '_eps' num2str(epsilon) '_n' num2str(n) 'm' num2str(m) '_sp' num2str(sp_ratio) '_wstart' num2str(warm_start) '_seed' num2str(rng_seed) '.mat'])
else %Light save
    clear screen_it_CoDscr screen_it_CoDscr_adap screen_it_CoDscr_global screen_it_MMscr screen_it_MMscr_adap screen_it_SPIRALscr screen_it_SPIRALscr_adap % cleaning variables not used for plots
    clear R_it_CoDscr R_it_CoDscr_adap R_it_CoDscr_global R_it_MMscr R_it_MMscr_adap R_it_SPIRALscr R_it_SPIRALscr_adap    
    clear obj_CoD obj_CoDscr obj_CoDscr_adap obj_CoD_global obj_MM obj_MMscr obj_MMscr_adap obj_SPIRAL obj_SPIRALscr obj_SPIRALscr_adap
    clear stop_crit_it_CoD stop_crit_it_CoDscr stop_crit_it_CoDscr_adap stop_crit_it_CoD_global stop_crit_it_MM stop_crit_it_MMscr stop_crit_it_MMscr_adap stop_crit_it_SPIRAL stop_crit_it_SPIRALscr stop_crit_it_SPIRALscr_adap
    clear time_it_CoD time_it_CoDscr time_it_CoDscr_adap time_it_CoD_global time_it_MM time_it_MMscr time_it_MMscr_adap time_it_SPIRAL time_it_SPIRALscr time_it_SPIRALscr_adap
    clear trace_CoDscr trace_CoDscr_adap trace_CoDscr_global trace_MMscr trace_MMscr_adap trace_SPIRALscr  trace_SPIRALscr_adap 
    save(['./Results/new_main_' problem_type '_screening_test_mcIt' num2str(mc_it) '_' num2str(length(lambdas)) 'lambdas' num2str(lambdas_rel(1)) '-' num2str(lambdas_rel(end)) '_tol' num2str(param.TOL) '_eps' num2str(epsilon) '_n' num2str(n) 'm' num2str(m) '_sp' num2str(sp_ratio) '_wstart' num2str(warm_start) '_seed' num2str(rng_seed) '.mat'])
end

plot_results
