switch exp_type
    case 'TasteProfile'
        % Check if the dataset is available locally
        if exist('TasteProfile_train_triplets_reduced10k100k.csv','file') ~= 2
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
        if exist('nips_1-17.mat','file') ~= 2
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
                   'Then run Grolier_encyclopedia.m script'
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