switch exp_type
    case 'TasteProfile'
        % Check if the dataset is available locally
        if exist('TasteProfile_train_triplets_reduced5k50k.csv','file') ~= 2
            error( ['\nMISSING DATASET!\n', ...
                   'Please download the TasteProfile dataset at:\n' ...
                   'http://millionsongdataset.com/tasteprofile/ \n' ...
                   'Then, run the notebook processTasteProfile.ipynb ' ...
                   'to generate the required csv dataset file%s'],'.')
        end
        
%         T = readtable('TasteProfile_train_triplets_reduced10k100k.csv'); %~9000 x 70000
        T = readtable('TasteProfile_train_triplets_reduced5k50k.csv'); %~4800 users x 44000 songs
        song_vs_user_mtx = sparse(T{:,2}+1,T{:,1}+1,T{:,3});
        clear T
        
        n = size(song_vs_user_mtx,1);
        m = size(song_vs_user_mtx,2) - 1;

        % Pick one random user to be the input signal 
        % (to be sparsely approximated as a function of the others)
        idx_y = randi(m+1);
        y_orig = song_vs_user_mtx(:,idx_y);
        A = song_vs_user_mtx(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
        clear song_vs_user_mtx
        
    case '20newsgroups'
        % Check if the dataset is available locally
        if exist('20newsgroups_Count_100words.mat','file') ~= 2
            error( '\nMISSING DATASET!\n Please run the python notebook 20newsgoups_vectorized.ipynb first.')
        end
        
        load('20newsgroups_Count_100words.mat'); X = X.'; % 102 x 18840, condition ~3e3
%         load('20newsgroups_Count.mat'); X = X.';
    
        %Remove all-zero columns or rows
        X(:,sum(X)==0) = [];
        X(sum(X,2)==0,:) = [];

        n = size(X,1);
        m = size(X,2) - 1;
        sp_ratio = nnz(X)/numel(X);

        % Pick one random user to be the input signal 
        % (to be sparsely approximated as a function of the others)
        idx_y = randi(m+1);
        y_orig = X(:,idx_y);
        A = X(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
        clear X words
        
    case 'NIPSpapers'
        % Check if the dataset is available locally
        if exist('nips_1-17.mat','file') ~= 2
            error( ['\nMISSING DATASET!\n', ...
                   'Please download the NIPSpaper dataset (mat file) at:\n' ...
                   'http://ai.stanford.edu/~gal/data.html'],' ')
        end
        
        load('nips_1-17') % 2484 x 14035
        
        %Remove all-zero columns or rows
        counts(:,sum(counts)==0) = [];
        counts(sum(counts,2)==0,:) = [];  % 2483 x 14035
        
        n = size(counts,2);
        m = size(counts,1) - 1;
        sp_ratio = nnz(counts)/numel(counts);

        % Pick one random user to be the input signal 
        % (to be sparsely approximated as a function of the others)
        idx_y = randi(m+1);
        y_orig = counts(idx_y,:).';
        A = counts([1:(idx_y-1) (idx_y+1):(m+1)],:).';
        clear counts words docs_names authors_names docs_authors aw_counts
        
    case 'Encyclopedia'
        % Check if the dataset is available locally
        if exist('grolier7500.mat','file') ~= 2
            error( ['\nMISSING DATASET!\n', ...
                   'Please download the Grolier encyclopedia dataset (mat file) at:\n' ...
                   'https://cs.nyu.edu/~roweis/data.html\n', ...
                   'Then run Encyclopedia_reduce.m script in ./datasets/ folder'
                   ],' ')
        end
        
        load('grolier7500.mat') % 7501 x 30990, condition ~3e3

        %Remove all-zero columns or rows
        grolier(:,sum(grolier)==0) = [];
        grolier(sum(grolier,2)==0,:) = []; % 7501 x 29515
         
        n = size(grolier,1);
        m = size(grolier,2) - 1;
        sp_ratio = nnz(grolier)/numel(grolier);

        % Pick one random user to be the input signal 
        % (to be sparsely approximated as a function of the others)
        idx_y = randi(m+1);
        y_orig = grolier(:,idx_y);
        A = grolier(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
        clear grolier words
        
    case 'MNIST'
        % Check if the dataset is available locally
        if exist('mnist_train.csv','file') ~= 2
            error( ['\nMISSING DATASET!\n', ...
                   'Please download the Grolier encyclopedia dataset (csv file) at:\n' ...
                   'https://www.python-course.eu/neural_network_mnist.php'],' ')
        end
        
        %Training digits
        T = readtable('mnist_train.csv');
        digits = sparse(T{:,:}).'; % 784 pixels x 60000 digits
%         digits = digits(2:end,full(digits(1,:))==6); %remove labels and keep only one digit
        digits = digits(2:end,:); %removing the label
        clear T
        %Append Testing digits
%         T = readtable('mnist_test.csv');
%         digits = [digits sparse(T{:,2:end}).'];
%         clear T
        
        n = size(digits,1);
        m = size(digits,2) - 1;
        sp_ratio = nnz(digits)/numel(digits);

        % Pick one random user to be the input signal 
        % (to be sparsely approximated as a function of the others)
        idx_y = randi(m+1);
        y_orig = digits(:,idx_y);
        A = digits(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
        clear digits
    
    case {'Leukemia', 'Leukemia_mod'}
        % Check if the dataset is available locally
        if exist('Leukemia.mat','file') ~= 2
            error( ['\nMISSING DATASET!\n', ...
                   'Please download the Leukemia dataset at LIBSVM:\n' ...
                   'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#leukemia\n', ...
                   'Python script ./datasets/LogisticRegressionData.ipynb' ... 
                   'can be used to load the data and save as .mat format.'
                   ],' ')
        end
        
        load('Leukemia.mat') % 72 x 7129 (38 training / 34 testing)

        A = full([A_train; A_test]);
        y_orig = [y_train.'; y_test.'];
        if strcmp(exp_type,'Leukemia_mod'), A(17,:)=[]; y_orig(17)=[]; end %Remove sample 17 (and maybe 20) to improve matrix conditioning
        
        clear A_test A_train y_test y_train
        
        n = size(A,1);
        m = size(A,2);
        sp_ratio = nnz(A)/numel(A);
        idx_y = 1;

    case {'Colon-cancer', 'Colon-cancer_mod'}
        % Check if the dataset is available locally
        if exist('Colon-cancer.mat','file') ~= 2
            error( ['\nMISSING DATASET!\n', ...
                   'Please download the Colon cancer dataset at LIBSVM:\n' ...
                   'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#colon-cancer\n', ...
                   'Python script ./datasets/LogisticRegressionData.ipynb' ... 
                   'can be used to load the data and save as .mat format.'
                   ],' ')
        end
        
        load('Colon-cancer.mat') % 60 x 2000

        A = full(A); y_orig = y.';
        if strcmp(exp_type,'Colon-cancer_mod'), A(57,:)=[]; y_orig(57)=[]; end %Remove sample 57 (and maybe 3) to improve matrix conditioning
        clear y
        
        n = size(A,1);
        m = size(A,2);
        sp_ratio = nnz(A)/numel(A);
        idx_y = 1; 
        
    case 'rcv1'
        % Check if the dataset is available locally
        if exist('rcv1_rankdef.mat','file') ~= 2
            error( ['\nMISSING DATASET!\n', ...
                   'Please download the Colon cancer dataset at LIBSVM:\n' ...
                   'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary\n', ...
                   'Python script ./datasets/LogisticRegressionData.ipynb' ... 
                   'can be used to load the data and save as .mat format.'
                   ],' ')
        end
        
        load('rcv1.mat')
%         load('rcv1_rankdef.mat') % Entire training dataset 20242 x 44504 (rank deficient)
%         load('rcv1_fullrank_n1000_m9553.mat') % 1000 randomly selected samples from testing dataset, size 1000 x 9553 (full rank)
%         load('rcv1_fullrank_n2000_m15308.mat') % 2000 randomly selected samples from testing dataset, size 2000 x 15308 (full rank)

        A = full(A); y_orig = y.';
        clear y
        
        n = size(A,1);
        m = size(A,2);
        sp_ratio = nnz(A)/numel(A);
        idx_y = 1;  
        
    case 'Moffett'
        
        load('Aviris_Moffet.mat')
        A = double(reshape(im(:,:,:),size(im,1)*size(im,2),size(im,3))');
        
        % Il y a du bruit sur les données qui crée des valeurs négatives; dans le
        % papier TIP on les a seuillé brutalement (sans en être fier)
        if min(A(:)) < 0,  A = A - min(A(:)); end %avoid negative entries
%         A(A<0) = 0;
        % Suppression de bandes de fréquences sans énergie (pre-processing utilisé par Nicolas)
        mask = [1:4 104:115 151:175 205:222]; 
        A(mask,:) = [];

        n = size(A,1);
        m = size(A,2) - 1;     
        sp_ratio = nnz(A)/numel(A);

        % Pick one random user to be the input signal 
        % (to be sparsely approximated as a function of the others)
        idx_y = randi(m+1);
        y_orig = A(:,idx_y);
        A = A(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
        
        clear im
        
    case 'Madonna'

        load Hyspex_Madonna
        A = double(reshape(im(:,:,:),size(im,1)*size(im,2),size(im,3))');
        
        n = size(A,1);
        m = size(A,2) - 1;     
        sp_ratio = nnz(A)/numel(A);

        % Pick one random user to be the input signal 
        % (to be sparsely approximated as a function of the others)
        idx_y = randi(m+1);
        y_orig = A(:,idx_y);
        A = A(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
        
        clear im        

    case {'Urban', 'Urban_subsampled'}
        % Check if the dataset is available locally
        if exist('Urban_R162.mat','file') ~= 2
            error( ['\nMISSING DATASET!\n', ...
                   'Please download the Urban image (mat file) at:\n' ...
                   'https://rslab.ut.ac.ir/data'
                   ],' ')
        end

        load('Urban_R162.mat') % 162 x 94249 (307x307 pixels)
        A = Y;

        n = size(A,1);
        m = size(A,2) - 1;
        
        %randomly subsample pixels from the image
        if strcmp(exp_type,'Urban_subsampled')
            new_m = 5000;
            idx = randsample(m+1,new_m+1); %uniformly
            A = A(:,idx);
            m = new_m;
        end
        
        sp_ratio = nnz(A)/numel(A);

        % Pick one random user to be the input signal 
        % (to be sparsely approximated as a function of the others)
        idx_y = randi(m+1);
        y_orig = A(:,idx_y);
        A = A(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
        
        clear Y nCol nRow nBand maxValue new_m idx
        
    case {'Cuprite', 'Cuprite_subsampled'}
        % Check if the dataset is available locally
        if exist('cuprite_ref.mat','file') ~= 2
            error( ['\nMISSING DATASET!\n', ...
                   'Please download the (cropped) Cuprite image (mat file) at:\n' ...
                   'http://www.lx.it.pt/~bioucas/code.htm', ...
                   'Also available at (requires simple variable renaming in code below):', ...
                   'https://rslab.ut.ac.ir/data'
                   ],' ')
        end
        
        load('cuprite_ref.mat') % 188 x 47750, condition ~9.9e3
        if min(x(:)) < 0,  A = x - min(x(:)); end %avoid negative entries
%         A = x; A(A<0) = 0;

        n = size(A,1);
        m = size(A,2) - 1;
        
        %randomly subsample pixels from the image
        if strcmp(exp_type,'Cuprite_subsampled')
            new_m = 10000;
            idx = randsample(m+1,new_m+1); %uniformly
            A = A(:,idx);
            m = new_m;
        end
        
        sp_ratio = nnz(A)/numel(A);

        % Pick one random user to be the input signal 
        % (to be sparsely approximated as a function of the others)
        idx_y = randi(m+1);
        y_orig = A(:,idx_y);
        A = A(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
        
        clear x Columns L Lines new_m idx

    case {'Cuprite_USGS-lib', 'Urban_USGS-lib'}
        
        load USGS_1995_Library.mat % 224 x 501

        A = datalib(:,4:end); %3 first columns contain: 1) wavelenghts 2) resolution 3) channel number

        % Order bands by increasing wavelength
%         [~, index] = sort(datalib(:,1));
%         A =  datalib(index,4:end);
%         names = names(4:end,index);

        % Prune the library so that minimum angle (in degres) between any two 
        % signatures is larger than min_angle. The larger min_angle the easier 
        % the sparse regression problem is.
        min_angle = 3; % value used in Bioucas-Dias et al. 2012 Hyperspectral Unmixing Overview
        [A, ~] = prune_library(A,min_angle); % A becomes: 224 x 342
%         names = names(index',:); %list of material names

        % order  the columns of A by decreasing angles 
%         [A, index, angles] = sort_library_by_angle(A);
%         names = names(index',:);

        if strcmp(exp_type,'Cuprite_USGS-lib')
            % Check if the dataset is available locally
            if exist('cuprite_ref.mat','file') ~= 2
                error( ['\nMISSING DATASET!\n', ...
                       'Please download the (cropped) Cuprite image (mat file) at:\n' ...
                       'http://www.lx.it.pt/~bioucas/code.htm'
                       ],' ')
            end      

            load('cuprite_ref.mat') % 188 x 47750 (250x191 pixels)
            if min(x(:)) < 0,  x = x - min(x(:)); end %avoid negative entries

            % Randomly pick mc_it pixels from the image to be unmixed (by being
            % sparsely decomposed w.r.t. the pure signatures from USGS library)
            if ~exist('mc_it', 'var'), mc_it = 1; end
            idx_pixels = randsample(size(x,2),mc_it); %uniformly
            Y_orig = x(:,idx_pixels); %matrix containing the input signals as columns
            y_orig = Y_orig(:,1); idx_y = 1;
            
            SlectBands = BANDS.'; clear BANDS

        elseif strcmp(exp_type,'Urban_USGS-lib')
            % Check if the dataset is available locally
            if exist('Urban_R162.mat','file') ~= 2
                error( ['\nMISSING DATASET!\n', ...
                       'Please download the Urban image (mat file) at:\n' ...
                       'https://rslab.ut.ac.ir/data'
                       ],' ')
            end
            
            load('Urban_R162.mat') % 162 x 94249 (307x307 pixels)

            % Randomly pick mc_it pixels from the image to be unmixed (by being
            % sparsely decomposed w.r.t. the pure signatures from USGS library)
            if ~exist('mc_it', 'var'), mc_it = 1; end
            idx_pixels = randsample(size(Y,2),mc_it); %uniformly
            Y_orig = Y(:,idx_pixels); %matrix containing the input signals as columns
            y_orig = Y_orig(:,1); idx_y = 1;
            
            wavlen = 400:10:2500; %wavlengths (in nanometers) in the image
            
        end

        % Select bands in the library with wavelenght matching those in the image
%         A = A(SlectBands,:);
        %find band-by-band correspondance (gives basically the same result)
        idx_bands = zeros(1,length(SlectBands));
        for k = 1:length(SlectBands)
            [~, idx_bands(k)] = min(abs(wavlen(SlectBands(k))*1e-3 - datalib(:,1)));
        end
        A = A(idx_bands,:);

        n = size(A,1); % Final dimensions: 188 x 342 (Cuprite) or 162 x 342 (Urban)
        m = size(A,2) - 1;
        sp_ratio = nnz(A)/numel(A);
        
        clear x Columns Lines L datalib names idx_bands min_angle Y nCol nRow nBand maxValue
        
    otherwise
        error('\nType of experiment not implemented! Check exp_type variable.')    
end

%Normalize A if necessary
if param.normalizeA
    A = [A(:,1:(idx_y-1)) y_orig A(:,(idx_y):end)];
    normA = sqrt(sum(A.^2));
    A = A./repmat(normA,n,1);
    A = A(:,[1:(idx_y-1) (idx_y+1):(m+1)]);
end
