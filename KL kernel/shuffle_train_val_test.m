function [G_train_val,G_test2train,G_train2test,Y_train_val, Y_test] = shuffle_train_val_test(G,label)
ratio = 0.3;
num_total = size(G,1);

ind_test = [];
for i = 1:max(label)
    where_i = find(label==i);
    num_i = size(where_i,1);
    randind = randperm(num_i);
    ind_test_i = where_i(randind(1:floor(ratio*num_i)));
    ind_test = [ind_test ind_test_i.'];
end

ind_train_val = setdiff(1:num_total,ind_test);
G_train_val = G(ind_train_val,ind_train_val);
G_test2train = G(ind_test, ind_train_val);
G_train2test = G(ind_train_val, ind_test);

Y_train_val = label(ind_train_val);
Y_test = label(ind_test);
end

