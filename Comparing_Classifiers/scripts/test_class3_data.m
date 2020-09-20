 
% %class3_data = importdata('./datasets/group23/linearly_separable/class3_train.txt');
% means = mean(class3_data);
% %means
% %c = calc_covariance(class3_data, means);
% %c
% q = 20;
% [idx, means] = kmeans(class3_data, q);
% for c_index = 1:q
%         p = idx == c_index;
%         single_cluster_data = class3_data(p,:);
%         cov_single_cluster = calc_covariance(single_cluster_data, means(c_index,:));
%         cov_single_cluster;
%         inv(cov_single_cluster);
% end
A = [1 2; 3 4];
B = [1 2];
try_this(2, {A,B});


function c = calc_covariance(A, means)
    n = size(A,1);
    one_vector(1:n) = 1;
    %duplicate mean vec
    mean_mat = means(one_vector, :);
    A_sub_mean = A - mean_mat;
    c = (A_sub_mean'*A_sub_mean)/n;
end
 
function p = try_this(n, cell_array)
A = cell_array{1,1}
B = cell_array{1,2}
p = 1;
end