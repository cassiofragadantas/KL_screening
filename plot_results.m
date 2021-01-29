%% Plots
set(0,'DefaultTextInterpreter','latex'), set(0,'DefaultLegendInterpreter','latex');

if length(lambda) == 1 && mc_it ==  1 % single run plots
        
    if CoD
        % 1) Convergence vs time
        figure, subplot(2,1,1), legends = {'Alg. 2','Alg. 3'};
        title(['$CoD solver, \lambda / \lambda_{max} = 10^{' num2str(log10(lambdas_rel(1))) '} $'])
        semilogy(time_it_CoD, stop_crit_it_CoD,'k'); hold on
        if strcmp(problem_type,'logistic') 
            semilogy(time_it_CoDscr_global, stop_crit_it_CoDscr_global,'g');
            legends = ['Alg. 1' legends];
        end
        semilogy(time_it_CoDscr, stop_crit_it_CoDscr);
        semilogy(time_it_CoDscr_adap, stop_crit_it_CoDscr_adap);
        legends = ['CoD' legends];
        ylabel('Duality gap'), xlabel('Time [s]')
        legend(legends); grid on
        
        % 2) Screening ratio per iteration
        subplot(2,1,2), hold on, legends = {'Alg. 2','Alg. 3'}; 
        if strcmp(problem_type,'logistic') 
            plot(screen_ratio_it_CoD_global(1:nb_iter_CoDscr_global),'g');
            legends = ['Alg. 1' legends];
        end
        plot(screen_ratio_it_CoD(1:nb_iter_CoDscr));
        plot(screen_ratio_it_CoD_adap(1:nb_iter_CoDscr_adap));
        xlabel('Iteration number'), ylabel('Screening ratio') %(\% of screened coordinates)
        legend(legends,'Location', 'southeast'); grid on
    end
    if MM
        % 1) Convergence vs time
        figure, subplot(2,1,1), legends = {'Alg. 2','Alg. 3'};
        title(['$MU solver, \lambda / \lambda_{max} = 10^{' num2str(log10(lambdas_rel(1))) '} $'])
        semilogy(time_it_MM, stop_crit_it_MM, 'k'); hold on
        if strcmp(problem_type,'logistic') 
            semilogy(time_it_MMscr_global, stop_crit_it_MMscr_global, 'g');
            legends = ['Alg. 1' legends];
        end
        semilogy(time_it_MMscr, stop_crit_it_MMscr);
        semilogy(time_it_MMscr_adap, stop_crit_it_MMscr_adap);
        legends = ['MU' legends];
        ylabel('Duality gap'), xlabel('Time [s]')
        legend(legends); grid on
        
        % 2) Screening ratio per iteration
        subplot(2,1,2), hold on, legends = {'Alg. 2','Alg. 3'}; 
        if strcmp(problem_type,'logistic') 
            plot(screen_ratio_it_MM_global(1:nb_iter_MMscr_global),'g');
            legends = ['Alg. 1' legends];
        end
        plot(screen_ratio_it_MM(1:nb_iter_MMscr));
        plot(screen_ratio_it_MM_adap(1:nb_iter_MMscr_adap));
        xlabel('Iteration number'), ylabel('Screening ratio') %(\% of screened coordinates)
        legend(legends,'Location', 'southeast'); grid on
    end
    if PG
        % 1) Convergence vs time
        figure, subplot(2,1,1), legends = {'Alg. 2','Alg. 3'};
        title(['$SPIRAL solver, \lambda / \lambda_{max} = 10^{' num2str(log10(lambdas_rel(1))) '} $'])
        semilogy(time_it_SPIRAL, stop_crit_it_SPIRAL,'k'); hold on
        if strcmp(problem_type,'logistic') 
            semilogy(time_it_SPIRALscr_global, stop_crit_it_SPIRALscr_global, 'g');
            legends = ['Alg. 1' legends];
        end
        semilogy(time_it_SPIRALscr, stop_crit_it_SPIRALscr);
        semilogy(time_it_SPIRALscr_adap, stop_crit_it_SPIRALscr_adap);
        legends = ['SPIRAL' legends];
        ylabel('Duality gap'), xlabel('Time [s]')
        legend(legends); grid on

        % 2) Screening ratio per iteration
        subplot(2,1,2), hold on, legends = {'Alg. 2','Alg. 3'}; 
        if strcmp(problem_type,'logistic') 
            plot(screen_ratio_it_SPIRAL_global(1:nb_iter_SPIRALscr_global),'g');
            legends = ['Alg. 1' legends];
        end
        plot(screen_ratio_it_SPIRAL(1:nb_iter_SPIRALscr));
        plot(screen_ratio_it_SPIRAL_adap(1:nb_iter_SPIRALscr_adap));
        xlabel('Iteration number'), ylabel('Screening ratio') %(\% of screened coordinates)
        legend(legends,'Location', 'southeast'); grid on
    end     
    
elseif false % single run plots (other)
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

    figure
    subplot(2,1,1)
    imagesc(screen_it_CoDscr), colormap gray
    xlabel('Iteration number'), ylabel('Coordinate number')
    subplot(2,1,2)
    plot(sum(screen_it_CoDscr))
    xlabel('Iteration number'), ylabel('\# of screened coordinates')

    %radius and alpha per iteration
    figure
    subplot(1,2,1), hold on
    semilogy(R_it_CoDscr_global(1,2:end))
    semilogy(R_it_CoDscr(1,2:end),'-.') %fixed local
    semilogy(R_it_CoDscr_adap(1,2:end),'--')
    xlabel('Iteration number'), ylabel('Radius')
    subplot(1,2,2), hold on
    plot(2*stop_crit_it_CoDscr_global(2:end)./R_it_CoDscr_global(1,2:end).^2)        
    plot(2*stop_crit_it_CoDscr(2:end)./R_it_CoDscr(1,2:end).^2,'-.')
    plot(2*stop_crit_it_CoDscr_adap(2:end)./R_it_CoDscr_adap(1,2:end).^2,'--')
    xlabel('Iteration number'), ylabel('alpha')

    figure
    subplot(1,2,1), hold on
    semilogy(R_it_SPIRALscr(1,2:end))
    semilogy(R_it_SPIRALscr_adap(1,2:end),'--')
    set(gca, 'YScale', 'log')
    xlabel('Iteration number'), ylabel('Radius')
    subplot(1,2,2), hold on
    plot(2*stop_crit_it_SPIRALscr(2:end)./R_it_SPIRALscr(1,2:end).^2)
    plot(2*stop_crit_it_SPIRALscr_adap(2:end)./R_it_SPIRALscr_adap(1,2:end).^2,'--')
    xlabel('Iteration number'), ylabel('alpha')

    figure
    subplot(1,2,1), hold on
    semilogy(R_it_MMscr(1,2:end))
    semilogy(R_it_MMscr_adap(1,2:end),'--')
    set(gca, 'YScale', 'log')
    xlabel('Iteration number'), ylabel('Radius')
    subplot(1,2,2), hold on
    plot(2*stop_crit_it_MMscr(2:end)./R_it_MMscr(1,2:end).^2)
    plot(2*stop_crit_it_MMscr_adap(2:end)./R_it_MMscr_adap(1,2:end).^2,'--')
    xlabel('Iteration number'), ylabel('alpha')

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

elseif false % full path plots

    set(0,'DefaultTextInterpreter','latex'), set(0,'DefaultLegendInterpreter','latex')
    % Sparsity ratio vs. lambda
    figure, hold on,title('Sparsity ratio')
    plot(lambdas_rel,sparsity_ratio_CoD),
    plot(lambdas_rel,sparsity_ratio_MM),
    plot(lambdas_rel,sparsity_ratio_SPIRAL), 
    plot(lambdas_rel, screen_ratio_CoD),
    plot(lambdas_rel, screen_ratio_MM),
    plot(lambdas_rel, screen_ratio_SPIRAL),
    legend('CoD', 'MM', 'SPIRAL', 'screen ratio CoD', 'screen ratio MM', 'screen ratio SPIRAL')
    xlabel('Regularization ($\lambda / \lambda_{max}$)','Interpreter','latex')
    ylabel('Sparsity (\% of zeros)')
    % L1-norm ratio vs. lambda
    figure, hold on, title('L1 norm vs. Regularization level')
    plot(lambdas_rel,l1_norm_CoD)
    plot(lambdas_rel,l1_norm_MM)
    plot(lambdas_rel,l1_norm_SPIRAL)
    legend('CoD', 'MM', 'SPIRAL', 'screen ratio')
    xlabel('Regularization ($\lambda / \lambda_{max}$)','Interpreter','latex')
    ylabel('$\ell_1$-norm of the solution')
    % Colormap Screening ratio
    if mc_it == 1
        figure,
        subplot(1,3,1), imagesc(lambdas_rel,1:param.MAX_ITER,screen_ratio_it_CoD,[0 1]), colorbar
        subplot(1,3,2), imagesc(lambdas_rel,1:param.MAX_ITER,screen_ratio_it_MM,[0 1]), colorbar
        subplot(1,3,3), imagesc(lambdas_rel,1:param.MAX_ITER,screen_ratio_it_SPIRAL,[0 1]), colorbar
        title('Screening ratio (\% of coordinates)')
        xlabel('Regularization ($\lambda / \lambda_{max}$)','Interpreter','latex')
        ylabel('Iteration number')
    end

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
