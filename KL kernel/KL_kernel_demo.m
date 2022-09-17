path = './';
addpath(genpath(path));
dataset_name = 'kl_caltech_matrix.mat'; 
load(dataset_name);
model_gmm = 'GMM'; %'GMM
if strcmp(model_gmm,'GMM')
    K_train = GMM.train;
    K_train2test = GMM.train2test;
    K_test2train = GMM.test2train;
end
AA_train_val =K_train;
AA_test2train = K_test2train;
AA_train2test = K_train2test;
label_train_val = double(train_label)';
label_test = double(test_label)';
m = max(label_train_val);

lambda =1;
k_fold = 10;
gamma_list = [0.5, 1, 5, 10, 20, 100];
sigma_list = [1e-4 1e-3 1e-2 1e-1 1];
bias_list = [0];
bias = 0;
iter_total = 10;
num_2class = 0;

y_test_pre_total = zeros(size(label_test,1),iter_total);
y_test_total = zeros(size(label_test,1),iter_total);
for iters = 1:iter_total
    fprintf('The loop is on%6.2f\n',iters/iter_total)
    [shuffle_index] = shuffle_ind(label_train_val,dataset_name,num_2class); % shuffle data
    M_train_val = AA_train_val(shuffle_index,shuffle_index);
    M_test2train = AA_test2train(:, shuffle_index);
    M_train2test = AA_train2test(shuffle_index, :);
    Y_train_val = label_train_val(shuffle_index);
    Y_test = label_test;
    F1_score_list = zeros(length(gamma_list),length(sigma_list));
    for ind_sigma = 1:length(sigma_list)
        for ind_gamma = 1:length(gamma_list)
           fprintf('fitst loop is on %6.2f, second loop is on %6.2f \n',ind_sigma/length(sigma_list), ind_gamma/length(gamma_list))
            gamma = gamma_list(ind_gamma);
            sigma = sigma_list(ind_sigma);
            [G_train_val,~, ~] = kl_kernel(M_train_val,[],[],sigma,bias);
            F1_list = zeros(k_fold,1);
            num_periter = floor(size(G_train_val,1) / k_fold);
            for iter =1:k_fold
                if iter==k_fold
                    ind_val = ((k_fold-1) * num_periter + 1):size(G_train_val,1);
                else
                    ind_val = (iter-1) * num_periter + (1:num_periter);
                end
                ind_train = setdiff((1:size(G_train_val,1)),ind_val);
                G = G_train_val(ind_train,ind_train);
                Y_train = Y_train_val(ind_train,1);
                Y_val = Y_train_val(ind_val,1);
                G_val(:,:,1) = G_train_val(ind_val,ind_train); G_val(:,:,2) = G_train_val(ind_train,ind_val).';
                % n: data number / m: classes number 
                n = size(G,1); 
                X = zeros(2*m*(1+n),1);
                Y = zeros(m*n,1);
                for i = 1:m
                    % generate one-side labels
                    y = -1 * ones(n,1); y(Y_train==i)=1;
                    K = G   .* (y * y.');
                    A = [0,           0,             y.',            zeros(1,n); ...
                         0,           0,             zeros(1,n),     y.'; ...
                         y,           zeros(n,1),    eye(n)/gamma,   K; ...
                         zeros(n,1),  y,             K.',         eye(n)/gamma];
                    b = [0; 0; ones(n,1);ones(n,1)];
                    %change to least square?
                    if rcond(A)>1e-16
                        x = linsolve(A,b);
                    else
                        x = (A.'*A + lambda*eye(size(A)))^(-1)*A.'*b;     
                    end 
                    X(double(2*(1+n)*(i-1))+(1:2*(1+n)),1)=x;
                    Y(double(n*(i-1))+(1:n),1)=y;
                end
                % acc in val set
                n_val = size(G_val,1);
                class_val_node = zeros(n_val,m);
                for i = 1:m
                    b_1 = X(double(2*(1+n)*(i-1))+1,1); b_2 = X(double(2*(1+n)*(i-1))+2,1);
                    alpha = X(double(2*(1+n)*(i-1))+(3:n+2),1); beta = X(double(2*(1+n)*(i-1))+(n+3:2*n+2),1);
                    y_train = Y(double(n*(i-1))+(1:n),1);
                    f_s = G_val(:,:,1) * (beta .* y_train)  + b_1;
                    f_t = G_val(:,:,2) * (alpha .* y_train)  + b_2;
                    class_val_node(:,i) = f_s+f_t;
                end
                % max value of each column & row index of each column
                [MA,y_val_pre]=max(class_val_node.');  y_val_pre = y_val_pre.';
                y_val_pre(y_val_pre<0) = inf;
                acc_val = sum(y_val_pre==Y_val)/n_val;
                F1_list(iter,1) = acc_val;
                clear G_val Y_val
            end
            F1_score_list(ind_gamma,ind_sigma) = mean(F1_list);
        end
    end
    % max value of each column & row index of each column
    [MA,IA]=max(F1_score_list); 
     % max value of matrix & column index of that value
    [mVal,mInd]=max(MA);
    maxRow=IA(mInd); maxCol=mInd;
    [gamma_list(maxRow) sigma_list(maxCol) mVal]
    gamma = gamma_list(maxRow); sigma = sigma_list(maxCol);

%     gamma = 0.5; sigma=0.01;
    [G_train_val,G_test2train, G_train2test] = kl_kernel(M_train_val,M_test2train,M_train2test,sigma,bias);
    G = G_train_val; Y_train = Y_train_val;
    n = size(G,1);
    X = zeros(2*m*(1+n),1);
    Y = zeros(m*n,1);
    for i = 1:m
        % generate one-side labels
        y = -1 * ones(n,1); y(Y_train==i)=1;
        K = G .* (y * y.');
        A = [0,           0,             y.',            zeros(1,n); ...
             0,           0,             zeros(1,n),     y.'; ...
             y,           zeros(n,1),    eye(n)/gamma,   K; ...
             zeros(n,1),  y,             K.',         eye(n)/gamma];
        b = [0; 0; ones(n,1);ones(n,1)];
        %change to least square?
        if rcond(A)>1e-16
            x = linsolve(A,b);
        else
            x = (A.'*A + lambda*eye(size(A)))^(-1)*A.'*b;
        end 
        
        X(double(2*(1+n)*(i-1))+(1:2*(1+n)),1)=x;
        Y(double(n*(i-1))+(1:n),1)=y;
    end
    % train process
    n_train = size(G,1);
    class_test_node = zeros(n_train,m);
    class_test_node_s = zeros(n_train,m);
    class_test_node_t = zeros(n_train,m);
    for i = 1:m
        b_1 = X(double(2*(1+n)*(i-1))+1,1); b_2 = X(double(2*(1+n)*(i-1))+2,1);
        alpha = X(double(2*(1+n)*(i-1))+(3:n+2),1); beta = X(double(2*(1+n)*(i-1))+(n+3:2*n+2),1);
        y_train = Y(double(n*(i-1))+(1:n),1);
        f_s = G * (beta .* y_train)  + b_1;
        f_t = G.' * (alpha .* y_train)  + b_2;
        class_test_node(:,i) = f_s+f_t;
        class_test_node_s(:,i) = f_s;
        class_test_node_t(:,i) = f_t;
    end
    % max value of each column & row index of each column
    [MA,y_test_pre]=max(class_test_node.');  y_test_pre = y_test_pre.';
    y_test_pre(y_test_pre<0) = inf;
    acc_test = sum(y_test_pre==Y_train)/n_train;

    [MA,y_test_pre_s]=max(class_test_node_s.');  y_test_pre_s = y_test_pre_s.';
    y_test_pre_s(y_test_pre_s<0) = inf;
    acc_test_s = sum(y_test_pre_s==Y_train)/n_train;
    
    [MA,y_test_pre_t]=max(class_test_node_t.');  y_test_pre_t = y_test_pre_t.';
    y_test_pre_t(y_test_pre_t<0) = inf;
    acc_test_t = sum(y_test_pre_t==Y_train)/n_train;
    acc_train_s_t = [acc_test acc_test_s acc_test_t]
    
    % test process
    n_test = size(G_test2train,1);
    class_test_node = zeros(n_test,m);
    class_test_node_s = zeros(n_test,m);
    class_test_node_t = zeros(n_test,m);
    for i = 1:m
        b_1 = X(double(2*(1+n)*(i-1))+1,1); b_2 = X(double(2*(1+n)*(i-1))+2,1);
        alpha = X(double(2*(1+n)*(i-1))+(3:n+2),1); beta = X(double(2*(1+n)*(i-1))+(n+3:2*n+2),1);
        y_train = Y(double(n*(i-1))+(1:n),1);
        f_s = G_test2train * (beta .* y_train)  + b_1;
        f_t = G_train2test.' * (alpha .* y_train)  + b_2;
        class_test_node(:,i) = f_s+f_t;
        class_test_node_s(:,i) = f_s;
        class_test_node_t(:,i) = f_t;
    end
    % max value of each column & row index of each column
    [MA,y_test_pre]=max(class_test_node.');  y_test_pre = y_test_pre.';
    y_test_pre(y_test_pre<0) = inf;
    acc_test = sum(y_test_pre==Y_test)/n_test;

    [MA,y_test_pre_s]=max(class_test_node_s.');  y_test_pre_s = y_test_pre_s.';
    y_test_pre_s(y_test_pre_s<0) = inf;
    acc_test_s = sum(y_test_pre_s==Y_test)/n_test;
    
    [MA,y_test_pre_t]=max(class_test_node_t.');  y_test_pre_t = y_test_pre_t.';
    y_test_pre_t(y_test_pre_t<0) = inf;
    acc_test_t = sum(y_test_pre_t==Y_test)/n_test;
    
    acc_test_s_t = [acc_test acc_test_s acc_test_t]
    
    y_test_pre_total(:,iters) = y_test_pre;
    y_test_total(:,iters) = Y_test;
end
save('KL_kernel.mat','y_test_pre_total','y_test_total')