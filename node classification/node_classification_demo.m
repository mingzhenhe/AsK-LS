path = './';
addpath(genpath(path));
% load dataset
dataset_name = 'data/cora.mat'; % cora citeseer pubmed pubmed_zihuan am_photo
                                         % am_computer
load(dataset_name);
AA = double(G);

[M_train_val, YY_train_val, M_test, YY_test] = split_datasets_cv(AA,label);
lambda =1;
k_fold = 10;
num_periter = floor(size(M_train_val,1) / k_fold);

gamma = 1; 
iter_total = 5;
if strcmp(dataset_name,'data/pubmed_zihuan.mat') 
    iter_total = 5;
end
y_test_pre_total = zeros(size(YY_test,1),iter_total);
y_test_total = zeros(size(YY_test,1),iter_total);
for iter = 1:iter_total
    fprintf('The loop is on %6.5f\n',iter/iter_total)
    % n: data number / m: classes number 
    [G_train_val, Y_train_val, G_test, Y_test] = split_datasets_cv(AA,label);
    
    m = max(label);
    n_test = size(G_test,1);
    class_test_node = zeros(n_test,m);
    for i = 1:m
        fprintf('The class loop is on %d-%d\n',i,m)
        X_feature = zeros(size(G_train_val,1),2);
        test_feature = zeros(size(G_test,1),2);
        Y_feature = -1 * ones(size(G_train_val,1),1);
        Y_feature(Y_train_val==i) = 1;
        
        for ind_stack = 1:k_fold
            if ind_stack==k_fold
                ind_val = ((k_fold-1) * num_periter + 1):size(G_train_val,1);
            else
                ind_val = (ind_stack-1) * num_periter + (1:num_periter);
            end
            ind_train = setdiff((1:size(G_train_val,1)),ind_val);
            G = G_train_val(ind_train,ind_train);
            Y_train = Y_train_val(ind_train,1);
            G_val(:,:,1) = G_train_val(ind_val,ind_train); G_val(:,:,2) = G_train_val(ind_train,ind_val).';
            
            G_test_tmp1 = G_test(:,ind_train,1); 
            G_test_tmp2 = G_test(:,ind_train,2); 
            n = size(G,1);
            % generate one-side labels
            y = -1 * ones(n,1); y(Y_train==i)=1;
            K = G .* (y * y.');
            A = [0,           0,             y.',            zeros(1,n); ...
                 0,           0,             zeros(1,n),     y.'; ...
                 y,           zeros(n,1),    eye(n)/gamma,   K; ...
                 zeros(n,1),  y,             K.',         eye(n)/gamma];
            b = [0; 0; ones(n,1);ones(n,1)];
            %change to least square?
            if rcond(A)>1e-8
                x = linsolve(A,b);
            else
                x = (A.'*A + lambda*eye(size(A)))^(-1)*A.'*b;
            end 
            % gerneate output of first classfier
            b_1 = x(1,1); b_2 = x(2,1);
            alpha = x((3:n+2),1); beta = x((n+3:2*n+2),1);
            y_train = y;
            f_s = G_val(:,:,1) * (beta .* y_train)  + b_1;
            f_t = G_val(:,:,2) * (alpha .* y_train)  + b_2;
            X_feature(ind_val,1) = f_s; X_feature(ind_val,2) = f_t;
            clear f_s f_t
            f_s = G_test_tmp1 * (beta .* y_train)  + b_1;
            f_t = G_test_tmp2 * (alpha .* y_train)  + b_2;
            test_feature(:,1) = test_feature(:,1) + f_s; 
            test_feature(:,2) = test_feature(:,2) + f_t;
            clear G_val Y_val
        end
        test_feature = test_feature/k_fold;
        % keyi kaolv qudiao bias
        Y_feature(Y_feature==-1) = 0;
        theta = glmfit(X_feature, Y_feature, 'binomial', 'link', 'logit');
        p = glmval(theta, test_feature, 'logit');
        class_test_node(:,i) = p;
    end

    % max value of each column & row index of each column
    [MA,y_test_pre]=max(class_test_node.');  y_test_pre = y_test_pre.';
    y_test_pre(y_test_pre<0) = inf;
    acc_test = sum(y_test_pre==Y_test)/n_test

    y_test = Y_test;
    y_test_pre_total(:,iter) = y_test_pre;
    y_test_total(:,iter) = y_test;
    
end
% save('F:\Directed graph data\Directed graph data\Graph-Embedding-master\pre.mat','y_test_pre','y_test')
save('node_classification.mat','y_test_pre_total','y_test_total')