%Author: Madhura Pande
A = importdata('../data_assign1_group28/train1000.txt',' ');
B = importdata('../data_assign1_group28/val.txt',' ');
C = importdata('../data_assign1_group28/test.txt');
n_train = size(A,1);
n_val = size(B,1);
n_test = size(C,1);
%n_train
%n_val
%n_test

x1_train = A(:,1);
x2_train = A(:,2);
t_train = A(:,3);

x1_val = B(:,1);
x2_val = B(:,2);
t_val = B(:,3);


x1_test = C(:,1);
x2_test = C(:,2);
t_test = C(:,3);

figure;
tri = delaunay(x1_train,x2_train);
trisurf(tri,x1_train,x2_train,t_train);
title('Surface plot of training data');
xlabel('x1');
ylabel('x2');
zlabel('t');

figure;
scatter3(x1_train,x2_train,t_train);
xlabel('x1');
title('Scatter plot of training data');
ylabel('x2');
zlabel('t');

degrees = [12, 15, 20];
etrain = zeros(1,size(degrees,2));
etest = zeros(1,size(degrees,2));
eval = zeros(1, size(degrees,2));
d = 2;

for k = 1:size(degrees,2) 
    M = degrees(k);
    D = factorial(M+d)/(factorial(M)*factorial(d));
    phi = zeros(n_train,D);
    phi_test = zeros(n_test, D);
    phi_val = zeros(n_val, D);

    d_index = 1;
    for m = 0:M
        for i = 0:m
            j = m-i;
            phi(:,d_index) = (x1_train.^j).*(x2_train.^i);
            d_index = d_index+1;
        end
    end
    lambda = exp(-18); %replace this with different values to get lambda
    w = inv(phi'*phi + (eye(D).*lambda))*phi'*t_train;
    %Predicting for train data
    y_train = phi*w;
    error_train = calcerror(y_train, t_train, n_train);
    etrain(k) = error_train;
    
    %calculating phi for prediction for validation set
    d_index = 1;
    for m = 0:M
        for i = 0:m
            j = m-i;
            phi_val(:,d_index) = (x1_val.^j).*(x2_val.^i);
            d_index = d_index+1;
        end
    end
    y_val = phi_val*w;
    error_val = calcerror(y_val,t_val,n_val);
    eval(k) = error_val;
    
    %calculating phi for prediction for test set
    d_index = 1;
    for m = 0:M
        for i = 0:m
            j = m-i;
            phi_test(:,d_index) = (x1_test.^j).*(x2_test.^i);
            d_index = d_index+1;
        end
    end
    y_test = phi_test*w;
    error_test = calcerror(y_test,t_test,n_test);
    etest(k) = error_test;
end

figure;
plot(degrees,etrain); hold on;
xlabel('M');
plot(degrees, etest); 
legend({'Train Error','Test Error'},'Location','northeast');
hold off;

disp('ERMS for various values of M, with Regularization with lambda = exp(-18) ');
T = array2table([degrees; etrain; eval; etest],'RowNames',{'M','E_RMS_Train', 'E_RMS_Val','E_RMS_Test'})



%Generating scatter plots y/t for best model, M=8
M_best = 8;
d_index = 1;
   for m = 0:M_best
       for i = 0:m
           j = m-i;
           phi_test(:,d_index) = (x1_test.^j).*(x2_test.^i);
           d_index = d_index+1;
       end
   end
y_test = phi_test*w;

d_index = 1;
   for m = 0:M_best
       for i = 0:m
           j = m-i;
           phi(:,d_index) = (x1_train.^j).*(x2_train.^i);
           d_index = d_index+1;
       end
   end
y_train = phi*w;
figure;
tri = delaunay(x1_train,x2_train);
trisurf(tri,x1_train,x2_train,y_train);
xlabel('x1');
ylabel('x2');
zlabel('y');
title('Surface plot for the best model');

figure;
subplot(1,2,1);
scatter(t_train,y_train);
xlabel('t');
ylabel('y');
title('Scatter Plot of Training data for best model for M=8');
subplot(1,2,2);
scatter(t_test,y_test);
xlabel('t');
ylabel('y');
title('Scatter Plot of Test data for best model for M=8');

function e_rms = calcerror(y,t,n)
 e_rms = (sum((t-y).^2)/n).^0.5;
end
