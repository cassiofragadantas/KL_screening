addpath('./solvers/')
addpath('./screening_rules/')
addpath('./datasets/')
rng(10), fprintf('\n\n /!\\/!\\ RANDOM SEED ACTIVATED /!\\/!\\\n\n');

%==== User parameters ====
mc_it = 2; %100 % Number of noise realizations
plots = false;
param.save_all = true;
param.verbose = false;
param.stop_crit = 'gap'; 
param.TOL = 1e-9; %-1 to run until max iteration
param.MAX_ITER = 1e6;
epsilon = 1e-6; % >0 to avoid singularity at 0 on KL divergence.
param.epsilon = epsilon;
param.epsilon_y = 0; %epsilon
%CHECK STANDARD REGULARIZATION PATH IN GAP SAFE JOURNAL PAPER (100 POINTS FROM 10-3 TO 1)
lambdas_rel = 1E-6; %logspace(-6,-10,9); %logspace(-1,-12,100); %logspace(-8,-12,100); %logspace(-2,0,100); % regularization (relative to lambda_max) %(0.01:0.01:1.1)

warm_start = false;

param.euc_dist = false; %If true, runs SPIRAL and FISTA in standard lasso problem (w/ Euclidean distance)

%==== Problem parameters ====
noise_type = 'poisson'; % Options: 'poisson', 'gaussian_std', 'gaussian_snr', otherwise: no noise.
exp_type = 'synthetic'; % Options: 'synthetic', or some real dataset 
            % (e.g. 'TasteProfile', '20newsgroups', 'NIPSpapers', 'Encyclopedia', 'MNIST')
param.normalizeA = true;
param.ymult = 5; %~5: for y with lots of zero entries. Synthetic experiment only.
            
if strcmp(noise_type,'gaussian_std')
    sigma = 0.1; %noise standard deviation
elseif strcmp(noise_type,'gaussian_snr')
    snr_db = 15;
end

if strcmp(exp_type,'synthetic')
    n = 200; %20; 
    m = 400; %40;
    sp_ratio = 0.1; %Non-zero entries ratio in x_golden
    nnz_x = sp_ratio*m; % number of nonzero entries of the solution x             
    regenerate_A = false; % Used in the synthetic experiments only
    A = abs(randn(n,m));
%     A = abs(sprand(n,m,0.02,0.01)); assert(~any(sum(A) == 0)) %density, condition number
    A = A./repmat(sqrt(sum(A.^2)),n,1);% normalizing columns
else
    sp_ratio = 'UnDef'; %Non-zero entries ratio
    load_dataset
end

x0_CoD = zeros(m,1);
x0_CoDscr = zeros(m,1);
x0_MM = ones(m,1);
x0_MMscr = ones(m,1);
x0_SPIRAL = ones(m,1);
x0_SPIRALscr = ones(m,1);
%x0_FISTA = ones(m,1);

%gap_factor = zeros(size(lambdas_rel)); % ration between duality gap and the biggest screened entry in MM

%==== Storage variables ====
l1_norm_MM = zeros(size(lambdas_rel));
l1_norm_SPIRAL = zeros(size(lambdas_rel));
% sparsity_ratio_MM = zeros(size(lambdas_rel));
sparsity_ratio_MM_direct = zeros(size(lambdas_rel));
sparsity_ratio_SPIRAL_direct = zeros(size(lambdas_rel));
screen_ratio = zeros(size(lambdas_rel));
screen_ratio_it = zeros(param.MAX_ITER,length(lambdas_rel));
rec_err_euc_SPIRAL = zeros(size(lambdas_rel));
rec_err_KL_SPIRAL = zeros(size(lambdas_rel));
input_error_euc = 0; input_error_KL = 0;
%
%time
time_CoD = zeros(size(lambdas_rel));
time_CoDscr = zeros(size(lambdas_rel));
time_MM = zeros(size(lambdas_rel));
time_MMscr = zeros(size(lambdas_rel));
time_SPIRAL = zeros(size(lambdas_rel));
time_SPIRALscr = zeros(size(lambdas_rel));
%time_FISTA = zeros(size(lambdas_rel));
time_precalc = zeros(size(lambdas_rel));
%
%time: various convergence thresholds
time_CoD_various = zeros(5, length(lambdas_rel));
time_CoDscr_various = zeros(5, length(lambdas_rel));
time_MM_various = zeros(5, length(lambdas_rel));
time_MMscr_various = zeros(5, length(lambdas_rel));
time_SPIRAL_various = zeros(5, length(lambdas_rel));
time_SPIRALscr_various = zeros(5, length(lambdas_rel));

% ==== Main loop ====
for k_mc = 1:mc_it
k_mc

%Generate problem and input data
if strcmp(exp_type,'synthetic')
    if regenerate_A
        A = abs(randn(n,m));
        A = A./repmat(sqrt(sum(A.^2)),n,1);% normalizing columns
    end
    x_golden = param.ymult*sprand(m,1,sp_ratio);
    y_orig = A*abs(full(x_golden)); %y = abs(randn(n,1));
else
    %Draw a random entry to be the input signal
    if param.normalizeA, y_orig = y_orig/normA(idx_y); end
    A = [A(:,1:(idx_y-1)) y_orig A(:,(idx_y):end)];
    idx_y = randi(m+1);
    y_orig = A(:,idx_y);
    A = A(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
    if param.normalizeA, y_orig = round(y_orig*normA(idx_y)); end
end

%Add noise
if strcmp(noise_type,'poisson')
    y = poissrnd(y_orig);
elseif strcmp(noise_type,'gaussian_std') % gaussian with std
    y = y_orig + sigma*randn(size(y_orig));
elseif strcmp(noise_type,'gaussian_snr') % gaussin with snr
    y = y_orig + (10^(-snr_db/20)*norm(y_orig)/sqrt(n))*randn(n,1);
else % no noise
    y = y_orig;
end
assert(all(y>=0),'Input signal should only have positive entries.')

%Keep only non-zero entries in y
% idx = (y~=0); y=y(idx); A = A(idx,:); n = length(y);

precalc = [];  % To ensure that it will be fully recalculated

if epsilon>0
    lambda_max = max(A.'*y)/epsilon;
else
    lambda_max = max(A.'*y); %In case of epsilon=0, lambda_max actually makes no sense. So, just using that of the standard lasso
end

lambdas = lambdas_rel*lambda_max; 

for k_lambda = 1:length(lambdas)
    lambda = lambdas(k_lambda);

    %Precalculations for Screening tests
    tPrecalc = tic;
    precalc = KL_GAP_Safe_precalc(A,y,lambda,param.epsilon_y, precalc);
    time_precalc(k_lambda) = time_precalc(k_lambda) + toc(tPrecalc)/mc_it;

    %Coordinate Descent - Hsieh2011
    fprintf('CoD solver...\n')
%     profile on    
    [x_CoD, obj_CoD, x_it_CoD, stop_crit_it_CoD, time_it_CoD] ...
        = CoD_KL_l1(A,y,lambda,x0_CoD,param);
%     profile off
%     profsave(profile('info'),'./Results/new_Profile_CoD')
    
    %CoD + Screening
    fprintf('CoD solver + Screening...\n')
    [x_CoDscr, obj_CoDscr, x_it_CoDscr, R_it_CoDscr, screen_it_CoDscr, stop_crit_it_CoDscr, time_it_CoDscr] ...
        = CoD_KL_l1_GAPSafe(A,y,lambda,x0_CoDscr,param,precalc);
    
    %MM
    fprintf('MM solver...\n')
    [x_MM, obj_MM, x_it_MM, stop_crit_it_MM, time_it_MM] ...
        = KL_l1_MM(A,y,lambda,x0_MM,param);
    
    %MM + Screening
    fprintf('MM solver + Screening...\n')
    [x_MMscr, obj_MMscr, x_it_MMscr, R_it_MMscr, screen_it_MMscr, stop_crit_it_MMscr, time_it_MMscr] ...
        = KL_l1_MM_GAPSafe(A,y,lambda,x0_MMscr,param,precalc);

    %SPIRAL
    fprintf('SPIRAL solver...\n')
    [x_SPIRAL, obj_SPIRAL, x_it_SPIRAL, stop_crit_it_SPIRAL, time_it_SPIRAL, step_it_SPIRAL] ...
        = SPIRAL(A, y, lambda, x0_SPIRAL, param);

    %SPIRAL + Screening
    fprintf('SPIRAL solver + Screening...\n')
    [x_SPIRALscr, obj_SPIRALscr, x_it_SPIRALscr, R_it_SPIRALscr, screen_it_SPIRALscr, stop_crit_it_SPIRALscr, time_it_SPIRALscr] ...
        = SPIRAL_GAPSafe(A,y,lambda,x0_SPIRALscr,param,precalc);
    
    %FISTA
%     fprintf('FISTA solver...\n')
%     g.eval = @(x) lambda*norm(x, 1);
%     g.prox = @(x, tau) nnSoftThresholding(x, lambda*tau);
%     f.eval = @(x) sum((y+epsilon).*log((y+epsilon)./(A*x+epsilon)) - y + A*x);
%     f.grad = @(x) A.'*((A*x-y)./(A*x+epsilon)); % sum(A).' - A.'*((y+epsilon)./(A*x+epsilon));
%     % f.grad = @(a) a.^(beta-2).*(a-y); % for generic beta-divergence
%     if strcmp(param.stop_crit,'gap') % function handle for dual objective must be provided
%         %Dual point and objective
%     	param.theta = @(x) ((y - A*x)./(lambda*(A*x+epsilon)))/max(A.'* ((y - A*x)./(lambda*(A*x+epsilon)))); %dual scaling 
% %         param.dual = @(x) (y+epsilon).'*log(1+lambda*x) - sum(lambda*x*epsilon);
%         param.dual = @(x) (y(y~=0) + param.epsilon).'*log(1+lambda*x(y~=0)) - sum(lambda*param.epsilon*x(y~=0)); % since 0*log(a) = 0 for all a>=0. Avoids 0*log(0) = NaN
%     end
%       
%     if param.euc_dist
%         f.eval = @(x) 0.5*norm(y - A*x,2)^2; % Euclidean distance
%         f.grad = @(x) A.'*(A*x-y);
%     end
%      
%     [x_FISTA, obj_FISTA, x_it_FISTA, stop_crit_it_FISTA, time_it_FISTA ] ...
%         = FISTA(g, f, m, x0_FISTA, param);
% %     assert(~isnan(stop_crit_it_FISTA(end)), 'FISTA diverged') %FISTA diverges sometimes


    % Warm start
    if warm_start
        x0_CoD = x_CoD;
        x0_CoDscr = x_CoDscr;
        x0_MM = x_MM;
        x0_MMscr = x_MMscr;
        x0_SPIRAL = x_SPIRAL;
        x0_SPIRALscr = x_SPIRALscr;
        %x0_FISTA = x_FISTA;
    end

    % Empirical verification of screening safety
    if any(x_MM(screen_it_MMscr(:,end)) > 10*m*max(stop_crit_it_MM(end),eps)), warning('SCREENING FAILED!'); end
%     if any(screen_it_MMscr(:,end)) % estimating factor (about 1.25*m in 100 runs)
%         gap_factor(k_lambda) = max(gap_factor(k_lambda), max(x_MM(screen_it_MMscr(:,end))) / stop_crit_it_MM(end));
%     end
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
    
    %% Storing results
    %L1-norm
    l1_norm_MM(k_lambda) = l1_norm_MM(k_lambda) + norm(x_MM,1)/mc_it;
    l1_norm_SPIRAL(k_lambda) = l1_norm_SPIRAL(k_lambda) + norm(x_SPIRAL,1)/mc_it;

    % Sparsity
%     theta = (1/lambda)  * (y-A*x_MM)./(A*x_MM + param.epsilon);
%     certificate = A.'*theta;
%     sparsity_ratio_MM(k_lambda) = sparsity_ratio_MM(k_lambda) + sum(certificate<1-1e-10)/(m*mc_it);
    sparsity_ratio_MM_direct(k_lambda) = sparsity_ratio_MM_direct(k_lambda) + sum(x_MM< 1e-10)/(m*mc_it); %x_MM<=max(1e-10*max(x_MM),eps)
    sparsity_ratio_SPIRAL_direct(k_lambda) = sparsity_ratio_SPIRAL_direct(k_lambda) + sum(x_SPIRAL<1e-10)/(m*mc_it); %x_SPIRAL<=1e-10*max(x_SPIRAL)
    
    % Screen ratio
    screen_ratio(k_lambda) = screen_ratio(k_lambda) + sum(screen_it_MMscr(:,end))/(m*mc_it);
    screen_ratio_it(:,k_lambda) = screen_ratio_it(:,k_lambda) + padarray(sum(screen_it_MMscr).',size(screen_ratio_it,1)-length(stop_crit_it_MMscr),'replicate','post')/(m*mc_it);
    
    % Time
    time_CoD(k_lambda) = time_CoD(k_lambda) + time_it_CoD(end)/mc_it;
    time_CoDscr(k_lambda) = time_CoDscr(k_lambda) + time_it_CoDscr(end)/mc_it;    
    time_MM(k_lambda) = time_MM(k_lambda) + time_it_MM(end)/mc_it;
    time_MMscr(k_lambda) = time_MMscr(k_lambda) + time_it_MMscr(end)/mc_it;
    time_SPIRAL(k_lambda) = time_SPIRAL(k_lambda) + time_it_SPIRAL(end)/mc_it;
    time_SPIRALscr(k_lambda) = time_SPIRALscr(k_lambda) + time_it_SPIRALscr(end)/mc_it;
    %time_FISTA(k_lambda) = time_FISTA(k_lambda) + time_it_FISTA(end)/mc_it;
    
    % Time for various convergence thresholds
    for kk = 1:5
        time_CoD_various(kk,k_lambda) = time_CoD_various(kk,k_lambda) + min([inf time_it_CoD(find(stop_crit_it_CoD<10^(5-kk)*param.TOL,1))/mc_it]);
        time_CoDscr_various(kk,k_lambda) = time_CoDscr_various(kk,k_lambda) + min([inf time_it_CoDscr(find(stop_crit_it_CoDscr<10^(5-kk)*param.TOL,1))/mc_it]);
        time_MM_various(kk,k_lambda) = time_MM_various(kk,k_lambda) + min([inf time_it_MM(find(stop_crit_it_MM<10^(5-kk)*param.TOL,1))/mc_it]);
        time_MMscr_various(kk,k_lambda) = time_MMscr_various(kk,k_lambda) + min([inf time_it_MMscr(find(stop_crit_it_MMscr<10^(5-kk)*param.TOL,1))/mc_it]);
        time_SPIRAL_various(kk,k_lambda) = time_SPIRAL_various(kk,k_lambda) + min([inf time_it_SPIRAL(find(stop_crit_it_SPIRAL<10^(5-kk)*param.TOL,1))/mc_it]);
        time_SPIRALscr_various(kk,k_lambda) = time_SPIRALscr_various(kk,k_lambda) + min([inf time_it_SPIRALscr(find(stop_crit_it_SPIRALscr<10^(5-kk)*param.TOL,1))/mc_it]);
    end
    
    % Reconstruction error
    KL_err = @(x) sum((y_orig+epsilon).*log((y_orig+epsilon)./(x+epsilon)) - y_orig + x);
    rec_err_euc_SPIRAL(k_lambda) = rec_err_euc_SPIRAL(k_lambda) + norm(y_orig - A*x_SPIRAL)/mc_it;
    rec_err_KL_SPIRAL(k_lambda) = rec_err_KL_SPIRAL(k_lambda) +  KL_err(A*x_SPIRAL)/mc_it;
        
end
input_error_euc = input_error_euc + norm(y_orig - y)/mc_it;
input_error_KL = input_error_KL + KL_err(y)/mc_it;
end

% Saving results
if length(lambdas) == 1 && mc_it ==  1 %save coordinates evolution separately (heavy)
    save(['./Results/preliminary/new_main_KL_screening_test_mcIt' num2str(mc_it) '_lambdas' num2str(lambdas_rel(1)) '-' num2str(lambdas_rel(end)) '_tol' num2str(param.TOL) '_eps' num2str(epsilon) '_n' num2str(n) 'm' num2str(m) '_sp' num2str(sp_ratio) '_wstart' num2str(warm_start) '_CoordEvolution.mat'],'x_it_MMscr')
end
clear x_it_CoD x_it_CoDscr x_it_MM x_it_MMscr x_it_SPIRAL x_it_SPIRALscr % cleaning variables not used for plots
save(['./Results/preliminary/new_main_KL_screening_test_mcIt' num2str(mc_it) '_lambdas' num2str(lambdas_rel(1)) '-' num2str(lambdas_rel(end)) '_tol' num2str(param.TOL) '_eps' num2str(epsilon) '_n' num2str(n) 'm' num2str(m) '_sp' num2str(sp_ratio) '_wstart' num2str(warm_start) '.mat'])

%% Plots
if plots 
    set(0,'DefaultTextInterpreter','latex'), set(0,'DefaultLegendInterpreter','latex');
    
    if lambda == 1 && mc_it ==  1 % single run plots
        % Convergence vs iteration
        % Gap evolution
        figure,
        subplot(2,1,1)
        semilogy(stop_crit_it_CoD), hold on;
        semilogy(stop_crit_it_CoDscr),
        semilogy(stop_crit_it_MM),
        semilogy(stop_crit_it_MMscr),
        semilogy(stop_crit_it_SPIRAL), 
        semilogy(stop_crit_it_SPIRALscr), 
        %semilogy(stop_crit_it_FISTA);
        ylabel('Duality GAP'), xlabel('Iteration number')      
        title('Convergence')
        % Primal objective function values
%         subplot(2,1,2)
%         min_obj = min([min(obj_MM) min(obj_FISTA) min(obj_MMscr)]);
%         semilogy(obj_MM(2:end)-min_obj); hold on; 
%         semilogy(obj_MMscr(2:end)-min_obj); semilogy(obj_FISTA(2:end)-min_obj);
%         ylabel('Primal objective - min(Primal)'), xlabel('Iteration number')
        legend('CoD','CoD + screening', 'MM', 'MM + screeening', 'SPIRAL', 'SPIRAL + screening')

        % Convergence vs time
        % Gap evolution
        subplot(2,1,2)
        semilogy(time_it_CoD, stop_crit_it_CoD); hold on;
        semilogy(time_it_CoDscr, stop_crit_it_CoDscr);        
        semilogy(time_it_MM, stop_crit_it_MM),
        semilogy(time_it_MMscr, stop_crit_it_MMscr),
        semilogy(time_it_SPIRAL, stop_crit_it_SPIRAL);
        semilogy(time_it_SPIRALscr, stop_crit_it_SPIRALscr);
        %semilogy(time_it_FISTA, stop_crit_it_FISTA);
        ylabel('Duality GAP'), xlabel('Time [s]')
        
        % Iteration time (ugly, since it varies...)
%         figure, hold on
%         plot(time_it_MM), plot(time_it_MMscr)
%         figure, hold on
%         plot(time_it_MM(15:end) - time_it_MM(14:end-1)), plot(time_it_MMscr(15:end)-time_it_MMscr(14:end-1))
        
        % Evolution by coordinate
        figure,
        %set(gca, 'ColorOrder', hot(size(x_it_MM,1)), 'NextPlot', 'replacechildren'); %lines, hot, winter, jet
        semilogy(x_it_MMscr.')
        title('Coordinates evolution')
        xlabel('Iteration number'), ylabel('Coordinate value')

        % Screening per iteration
        figure
        subplot(2,1,1)
        imagesc(screen_it_MMscr), colormap gray
        xlabel('Iteration number'), ylabel('Coordinate number')
        subplot(2,1,2)
        plot(sum(screen_it_MMscr))
        xlabel('Iteration number'), ylabel('\# of screened coordinates')
        
        figure
        subplot(2,1,1)
        imagesc(screen_it_SPIRALscr), colormap gray
        xlabel('Iteration number'), ylabel('Coordinate number')
        subplot(2,1,2)
        plot(sum(screen_it_SPIRALscr))
        xlabel('Iteration number'), ylabel('\# of screened coordinates')


        %See solutions
        if strcmp(exp_type,'synthetic')
            figure, hold on
            stem(x_CoD), stem(x_MM,'x'), stem(x_SPIRAL,'x'), stem(x_golden)
            legend('CoD','MM','SPIRAL','Golden')
            xlabel('Coordinate index'), ylabel('Coordinate value')
            title('Sparse solution vector')
            fprintf(['Sparsity ratio: (1-nnz)/m = ' num2str(sparsity_ratio) '\n'])
        end
        
        % Reconstruction
        figure, hold on
        plot(y_orig), plot(y), plot(A*x_CoD,'--'), plot(A*x_MM,'--'), plot(A*x_SPIRAL, '-.' )
        legend('Original', 'Noisy', 'CoD reconstructed', 'MM reconstructed','SPIRAL reconstructed')
        xlabel('Coordinate index'), ylabel('Coordinate value')
        title('Input vector reconstruction')

        figure, hold on
        plot(y-y_orig), plot(A*x_MM-y_orig,'--'), plot(A*x_SPIRAL-y_orig, '-.' )
        legend('Noisy (- Original)', 'MM rec. (- Original)','SPIRAL rec. (- Original)')
        ylabel('Error w.r.t clean signal') %xlabel('Coordinate index')
        title('Reconstruction error per coordinate (the closer to zero, the better!)')
        ax= gca, ax.XAxisLocation = 'origin';
    
    else % full path plots
        
        set(0,'DefaultTextInterpreter','latex'), set(0,'DefaultLegendInterpreter','latex')
        % Sparsity ratio vs. lambda
        figure, hold on,title('Sparsity ratio')
        plot(lambdas_rel,sparsity_ratio_MM_direct)
        plot(lambdas_rel,sparsity_ratio_SPIRAL_direct), 
        plot(lambdas_rel, screen_ratio)
        legend('MM', 'SPIRAL', 'screen ratio')
        xlabel('Regularization ($\lambda / \lambda_{max}$)','Interpreter','latex')
        ylabel('Sparsity (\% of zeros)')
        % L1-norm ratio vs. lambda
        figure, plot(lambdas_rel,l1_norm_MM)
        xlabel('Regularization ($\lambda / \lambda_{max}$)','Interpreter','latex')
        ylabel('$\ell_1$-norm of the solution')
        % Colormap
        figure,
        imagesc(lambdas_rel,1:param.MAX_ITER,screen_ratio_it,[0 1]), colorbar
        title('Screening ratio (\% of coordinates)')
        xlabel('Regularization ($\lambda / \lambda_{max}$)','Interpreter','latex')
        ylabel('Iteration number')
        
        % Reconstruction error
        %SNR
        figure, hold on
        semilogx(lambdas_rel,20*log10(norm(y_orig)./rec_err_euc_SPIRAL))
        ax = gca; hline = refline(ax,[0 20*log10(norm(y_orig)./norm(y_orig - y))]); hline.Color = 'k';
        xlabel('Regularization ($\lambda / \lambda_{max}$)','Interpreter','latex')
        ylabel('Reconstruction error (euclidean) [SNR]')
        legend('SPIRAL','Noisy')
        %Error
        figure
        subplot(2,1,1), hold on
        loglog(lambdas_rel,rec_err_euc_SPIRAL)
        ax = gca; hline = refline(ax,[0 input_error_euc]); hline.Color = 'k';
        xlabel('Regularization ($\lambda / \lambda_{max}$)','Interpreter','latex')
        ylabel('Reconstruction error (euclidean)')
        legend('SPIRAL','Noisy')
        subplot(2,1,2), hold on
        loglog(lambdas_rel,rec_err_KL_SPIRAL)
        ax = gca; hline = refline(ax,[0 input_error_KL]); hline.Color = 'k';
        xlabel('Regularization ($\lambda / \lambda_{max}$)','Interpreter','latex')
        ylabel('Reconstruction error (KL)')
        legend('SPIRAL','Noisy')
        %Time gain screening
        %TODO
    end
end
