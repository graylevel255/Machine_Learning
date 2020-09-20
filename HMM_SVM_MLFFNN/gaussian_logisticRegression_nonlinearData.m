function gaussian_logisticRegression_nonlinearData(data,d,nc,n)
% w = randn(nc,1);
w = [0.3018,;0.3999];

a = zeros(n,nc);

X = data(:,1:2);

phi = zeros(n,4);

n1 = n/nc;
n2 = n1;  

for i = 1:n1
    class1(i,1) = data(i,1);
    class1(i,2) = data(i,2);
end

for i = 1:n2
   class2(i,1) = data(i+n1,1);
   class2(i,2) = data(i+n1,2);
end


% Calculate prior probabilities

P_c1 = n1/n;
P_c2 = n2/n;

% Calculate mean and covariance matrices for different classes

mu_1 = mean(class1,1);
mu_2 = mean(class2,1);

% x_minus_mu vector for class 1

x_minus_mu1 = xmu(n1,d,class1,mu_1);
x_minus_mu2 = xmu(n2,d,class2,mu_2);
% Calculate covariance matrix

cov1 = mtimes(transpose(x_minus_mu1),x_minus_mu1)/n1;
cov2 = mtimes(transpose(x_minus_mu2),x_minus_mu2)/n2;

% Calculating global varinces of all classes

 gv1 = finding_global_variance(cov1,d);
 gv2 = finding_global_variance(cov2,d);
 
 phi_1 = calculate_phi(X,mu_1,gv1,n);
 phi_2 = calculate_phi(X,mu_2,gv2,n);
 
 phi(:,1) = phi_1(:);
 phi(:,2) = phi_2(:);
 
 global_phi = calculate_global_phi_matrix(X,n,d);
 
for i=1:nc
    a(:,i) = w(i,1)*phi(:,i);
end

y = zeros(n,nc);
t = zeros(n,nc);

for i = 1:n1
    t(i,1) = 1;
end

for i = 1:n2
    t(i+n1,2) = 1;
end


% Calculate predicted value

y = predict_outputs(a,n,nc);

% Write weight update code

eta = 0.001;

w_star = weight_update(w,eta,y,t,global_phi,n,d,nc,phi);

y_pred = classify_samples(w_star,X,n,nc,phi);
n3 = 0; n4 = 0;
Train_accuracy = calculate_accuracy(y_pred, n1,n2,n3,n4,n,nc)


%%% Evaluate on test data

[test_data,nc] = readCategoryFile('nonlinear', 'test');

X_test = test_data(:,2:3);
n_test = numel(test_data(:,1));
nt1 = n_test/nc; nt2 = nt1; nt3 = 0; nt4 = 0;

 t_phi_1 = calculate_phi(X_test,mu_1,gv1,n_test);
 t_phi_2 = calculate_phi(X_test,mu_2,gv2,n_test);
 

 t_phi(:,1) = t_phi_1(:);
 t_phi(:,2) = t_phi_2(:);
  
 t_global_phi = calculate_global_phi_matrix(X_test,n_test,d);

 y_pred_test = classify_samples(w_star,X_test,n_test,nc,t_phi);
 Test_accuracy = calculate_accuracy(y_pred_test, nt1,nt2,nt3,nt4,n_test,nc)
 
 
%%% Evaluate on validation data

[val_data,nc] = readCategoryFile('nonlinear', 'val');

X_val = val_data(:,2:3);
n_val = numel(val_data(:,1));
nv1 = n_val/nc; nv2 = nv1; nv3 = 0; nv4 = 0;

 v_phi_1 = calculate_phi(X_val,mu_1,gv1,n_val);
 v_phi_2 = calculate_phi(X_val,mu_2,gv2,n_val);

 v_phi(:,1) = v_phi_1(:);
 v_phi(:,2) = v_phi_2(:);
 
 v_global_phi = calculate_global_phi_matrix(X_val,n_val,d);

 y_pred_val = classify_samples(w_star,X_val,n_val,nc,v_phi);
 Val_accuracy = calculate_accuracy(y_pred_val, nv1,nv2,nv3,nv4,n_val,nc)
 
 % Plot for decision regions 
 
 [X1,X2] = meshgrid(-15:.1:15, -15:.1:15);
 X_1 = reshape(X1,301*301,1);
 X_2 = reshape(X2,301*301,1);
 
 N = numel(X_1);
 x = zeros(N,d); 
 for i = 1:N
         x(i,1) = X_1(i);
         x(i,2) = X_2(i);
 end
 
 x_phi_1 = calculate_phi(x,mu_1,gv1,N);
 x_phi_2 = calculate_phi(x,mu_2,gv2,N);
 
 x_phi(:,1) = x_phi_1(:);
 x_phi(:,2) = x_phi_2(:);

 x_global_phi = calculate_global_phi_matrix(x,N,d);

 x_pred = classify_samples(w_star,x,N,nc,x_phi);
 
 classes = zeros(N,1);
 
 for i=1:N
     for j=1:nc
         if x_pred(i,j) == 1
             classes(i) = j;
         end
     end
 end
 
 
 %%%% SCATTER PLOT %%%%
 
%  figure;
%  scatter(X_1(classes == 1),X_2(classes == 1),25,[0.9961, 0.9961, 0.5859],'filled') %yellow
%  hold on;
%  scatter(X_1(classes == 2),X_2(classes == 2),25,[0.6758, 0.8555, 0.9961],'filled') %dark blue
%  hold on; 
%  scatter(class1(:,1),class1(:,2),25,[0.8438,0.6602,0],'filled');
%  hold on;
%  scatter(class2(:,1),class2(:,2),25,[0.0078,0.3438,0.4648],'filled');
%  hold on;
%  xlabel('x1');
%  ylabel('x2');
%  title('Decision region prediction for logistic regression using Gaussian Basis Function');
%  legend({'class1','class2'},'Location','north');
 
end