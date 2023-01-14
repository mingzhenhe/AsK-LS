function [shuffle_index] = shuffle_ind(train_label,dataset_name,num_2class)
    num_train = size(train_label,1);
    shuffle_index = randperm(num_train);
    if strcmp(dataset_name,'kl_2class.mat') || strcmp(dataset_name,'kl_2_cifar.mat')
        where_1 = find(train_label==1);
        num_1 = size(where_1,1);
        shuffle_1_ind = randperm(num_1);
        shuffle_1 = where_1(shuffle_1_ind(1:num_2class));
        
        where_2 = find(train_label==2);
        num_2 = size(where_2,1);
        shuffle_2_ind = randperm(num_2);
        shuffle_2 = where_2(shuffle_2_ind(1:num_2class));
        shuffle_index = [shuffle_1; shuffle_2];
        shuffle_index = sort(shuffle_index);
    end
end

