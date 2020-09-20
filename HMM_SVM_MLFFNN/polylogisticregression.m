function [trainaccuracy,wnew] = polylogisticregression(data,d,nc,M)
% M is degree
% d is dimension
D = factorial(M+d)/(factorial(M)*factorial(d));
ntrain = size(data,1);
x1_train = data(:,2);
x2_train = data(:,3);
wold = randn(nc,D);
phi = zeros(ntrain,D);

d_index = 1;
for m = 0:M
    for i = 0:m
       j = m-i;
       phi(:,d_index) = (x1_train.^j).*(x2_train.^i);
       d_index = d_index+1;
    end
end

a = phi*wold';
y = logistic_a_to_y(a);
%t = construct_t_matrix('linear',y);
t = construct_t_matrix(y);
eta = 0.01;
yt = y-t;
wnew = wold - eta*yt'*phi;
w_norm = norm(wnew-wold);
iter = 1;
while w_norm>0.01 && iter<3000
    iter = iter + 1
    wold = wnew;
    a = phi*wold';
    y = logistic_a_to_y(a);
    yt = y-t;
    wnew = wold - eta*yt'*phi;
    w_norm = norm(wnew-wold)
end
a_final = phi*wnew';
y_final = logistic_a_to_y(a_final);
iter
trainaccuracy = logistic_accuracy(y_final);
end