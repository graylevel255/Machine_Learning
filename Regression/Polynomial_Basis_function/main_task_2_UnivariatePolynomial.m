%Author: Madhura Pande
% This code tries different degrees of polynomials, different training set
% sizes, with(and w/o) regularization and generates plots, to fit the best model of the data. 
%Relevant E_RMS values are printed in console in table format.

%Generate 100 samples(Training set by adding random noise)
x = 0:0.01:1;
N = size(x,2);
t = exp(cos(2*3.14*x))+x;
r = normrnd(0,0.5,1,N);
t = t + r;

%Divide data into training, test and validation set randomly.
rand_perm = randperm(N);
train_ind = sort(rand_perm(1:70));
val_ind = sort(rand_perm(71:80));
test_ind = sort(rand_perm(81:end));
x_train = x(train_ind);
t_train = t(train_ind);
x_val = x(val_ind);
t_val = t(val_ind);
x_test = x(test_ind);
t_test = t(test_ind);
n = size(x_train,2);

%Just plotting to get a sense of data
figure;
plot(x,exp(cos(2*3.14*x))+x);
hold on;
scatter(x_train,t_train);
title('Training Data');
legend('y = \ite^{cos(2\pix)}+x');
text(0.5,0.9,'N=70','Units','normalized');
xlabel('x');
ylabel('t');
hold off;

%x_train

%Need to fit parameters by least sqaured error method.
M = [0,1,3,6,9,10,20,68]; %This is for various degrees of M
etrain = zeros(1,size(M,2));
etest = zeros(1,size(M,2));
eval = zeros(1, size(M,2));

for index = 1:size(M,2)
    m = M(index);
    %function myfitPolynomial is written to fit the data, plot, and get
    %errors. Defined in this file below.
    [erms_train, erms_val, erms_test] = myfitPolynomial(x_train, t_train, x_val, t_val, x_test, t_test, m, 0);
    etrain(index) = erms_train;
    etest(index) = erms_test;
    eval(index) = erms_val;   
end   
figure;
plot(M,etrain); hold on;
xlabel('M');
plot(M, etest); 
legend({'Train Error','Test Error'},'Location','northeast');
title('Train error plotted with values of M without Regularization');
hold off;

disp('ERMS for various values of M, without Regularization');
T = array2table([M; etrain; eval; etest],'RowNames',{'M','E_RMS_Train', 'E_RMS_Val','E_RMS_Test'})

%Adding regularization for M=20,68 with lambda = 0.5
[erms_train20, erms_val20, erms_test20] = myfitPolynomial(x_train, t_train, x_val, t_val, x_test, t_test, 20, 0.5);
[erms_train68, erms_val68, erms_test68] = myfitPolynomial(x_train, t_train, x_val, t_val, x_test, t_test, 68, 0.5);
disp('ERMS for various values of M, with Regularization, lambda = 0.5');

T = array2table([[20, 68]; [erms_train20, erms_train68]; [erms_test20, erms_test68]],'RowNames',{'M','E_RMS_Train','E_RMS_Test'})

%For different sizes, N1 = 30
N1 = 30;
rand_perm = randperm(N1);
train_ind = sort(rand_perm(1:N1));
x_train_n1 = x(train_ind);
t_train_n1 = t(train_ind);
%[erms_train_n1, erms_val_n1, erms_test_n1] = myfitPolynomial(x_train_n1, t_train_n1, x_val, t_val, x_test, t_test, 20, 0);


N2 = 10;
rand_perm = randperm(N2);
train_ind = sort(rand_perm(1:N2));
x_train_n2 = x(train_ind);
t_train_n2 = t(train_ind);
%[erms_train_n2, erms_val_n2, erms_test_n2] = myfitPolynomial(x_train_n2, t_train_n2, x_val, t_val, x_test, t_test, 9, 0);

%uncomment this code to get erms for lambda = exp(-18)
%Calc erms for lambda = exp(-18), M=20, N= 70
[erms_train_for_l, erms_val_for_l, erms_test_for_l] = myfitPolynomial(x_train, t_train, x_val, t_val, x_test, t_test, 20, exp(-18));
 %disp('Values for lambda = exp(-18), M=20');
 %erms_train_for_l
 %erms_val_for_l
 %erms_test_for_l
 %Calc erms for lambda = exp(-18), M=68, N= 70
 [erms_train_for_l1, erms_val_for_l1, erms_test_for_l1] = myfitPolynomial(x_train, t_train, x_val, t_val, x_test, t_test, 68, exp(-18));
 %disp('Values for lambda = exp(-18), M=68, N=70');
 %erms_train_for_l1
 %erms_val_for_l1
 %erms_test_for_l1
 
 
 
function e_rms = calcerror(x,y,t,n)
 e_rms = (sum((t-y).^2)/n).^0.5;
end

%This fn returns the error vector and plots the curve for training data.
function [erms_train,erms_val,erms_test] = myfitPolynomial(x_train, t_train, x_val, t_val, x_test, t_test, m, lambda)
 A = zeros(m+1,m+1);
 b = zeros(1,m+1);
 n = size(x_train,2);
 phi = zeros(m+1,n);
 for i = 1:m+1
     for j = 1:m+1
      A(i,j) = sum(x_train.^(i+j-2));
     end
 end
 %adding regularization term
 if lambda ~= 0
  for i = 1:m+1
      A(i,i) = A(i,i) + lambda;
  end
 end
 for j = 1:m+1
     b(j) = sum(t_train.*(x_train.^(j-1)));
 end
 %A
 b = b';
 %b
 w = linsolve(A,b);
 for i=1:m+1
     phi(i,:) = x_train.^(i-1);
 end 
 y_train = w'*phi;
 figure;
 plot(x_train,y_train,'color','b'); hold on;
 scatter(x_train,t_train);
 title(['Polynomial Fitting curve with {\lambda}=' num2str(lambda) ' for N=' num2str(n) '']);
 text(0.5,0.9,['M =' num2str(m) ''],'Units','normalized','FontWeight','bold');
 xlabel('x');
 ylabel('t');
 
 
 %For Validation data set
n_val = size(x_val,2);
phi_val = zeros(m+1, n_val);
for i=1:m+1
    phi_val(i,:) = x_val.^(i-1);
end  
y_val = w'*phi_val;

%For Test data set
n_test = size(x_test,2);
phi_test = zeros(m+1, n_test);
for i=1:m+1
    phi_test(i,:) = x_test.^(i-1);
end  
y_test = w'*phi_test;

%disp(['RMS Error for training set for m=' num2str(m) '']);
erms_train = calcerror(x_train, y_train, t_train,n);
erms_test = calcerror(x_test, y_test, t_test, n_test);
erms_val = calcerror(x_val, y_val, t_val, n_val);

%Plot of best model
if m == 6 && n == 70
    figure;
    plot([0 0.5 1 1.5 2 2.5 3 3.5],[0 0.5 1 1.5 2 2.5 3 3.5]); hold on;
    scatter(t_train, y_train);
    xlabel('t'); ylabel('y');
    title('Train Data Predicted v/s Expected for M=6');
    hold off;
    
    figure;
    plot([0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8],[0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7 7.5 8]); hold on;
    scatter(t_test, y_test);
    xlabel('t'); ylabel('y');
    title('Test data Predicted v/s Expected for M=6');
    hold off;
end
%erms_train
%erms_val
%erms_test
 
end
