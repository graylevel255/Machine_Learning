function global_var = finding_global_variance(covariance_matrix,dimension)

sum = 0;
for i = 1:dimension
    sum = sum + covariance_matrix(i,i);
end

global_var = sum/dimension;
