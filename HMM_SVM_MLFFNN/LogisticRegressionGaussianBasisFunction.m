
% Dataset 2(a)

[linear_data,nc] = readCategoryFile('linear','train');
d = 2;
n = size(linear_data,1);
gaussian_logisticRegression_LinearData(linear_data,d,nc,n);

% Dataset 2(b)

[non_linear_data, nlc] = readCategoryFile('nonlinear','train');
nl = size(non_linear_data,1);
gaussian_logisticRegression_nonlinearData(non_linear_data,d,nlc,nl);