%divides data 70% into train, 10%val, 20%test
function [train_data, test_data, val_data] = divide_data(class_data)
    n = size(class_data, 1);
    rand_perm = randperm(n);
    end_index_for_train = round(0.7*n);
    train_index = rand_perm(1:end_index_for_train);
    end_index_for_val = round(0.8*n);
    val_index = rand_perm(end_index_for_train+1 : end_index_for_val);
    test_index = rand_perm(end_index_for_val+1 : n);
    
    train_data = class_data(train_index, :);
    test_data = class_data(test_index, :);
    val_data = class_data(val_index, : );
end