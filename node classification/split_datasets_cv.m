function [G_train, y_train, G_test, y_test] = split_datasets_cv(G,label)
m=max(label);n=size(G,1);
trian_proportion = 0.6;
num_train = floor(trian_proportion*n);
% G_val_1 = K(x,X); G_val_2 = K(X,x).'
% f_s = G_val_1 * (beta.*label) + b1
% f_t = G_val_2 * (alpha.*label) + b2
total_ind = 1:n;

ind = randperm(n);
y_train = label(ind(1:num_train),1);
train_ind = ind(1:num_train).';

% get val/test ind
ind_val_test = setdiff(total_ind,train_ind);
test_ind = ind_val_test.';

y_test = label(test_ind);
G_train = G(train_ind,train_ind);
G_test(:,:,1) = G(test_ind,train_ind); G_test(:,:,2) = G(train_ind,test_ind).'; 
end