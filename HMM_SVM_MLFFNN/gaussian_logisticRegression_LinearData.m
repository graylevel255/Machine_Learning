function gaussian_logisticRegression_LinearData(data,d,nc,n)


w = randn(nc,1);
a = zeros(n,nc);

X = data(:,2:3);

phi = zeros(n,4);

n1 = n/nc;
n2 = n1;  n3 = n1; n4 = n1;

for i = 1:n1
    class1(i,1) = data(i,2);
    class1(i,2) = data(i,3);
end

for i = 1:n2
   class2(i,1) = data(i+n1,2);
   class2(i,2) = data(i+n1,3);
end

t1 = n1+n2;
for i = 1:n3
   class3(i,1) = data(i+t1,2);
   class3(i,2) = data(i+t1,3);
end

t2 = t1+n3;
for i = 1:n4
   class4(i,1) = data(i+t2,2);
   class4(i,2) = data(i+t2,3);
end


% Calculate mean and covariance matrices for different classes

mu_1 = mean(class1,1);
mu_2 = mean(class2,1);
mu_3 = mean(class3,1);
mu_4 = mean(class4,1);

% x_minus_mu vector for class 1

x_minus_mu1 = xmu(n1,d,class1,mu_1);
x_minus_mu2 = xmu(n2,d,class2,mu_2);
x_minus_mu3 = xmu(n3,d,class3,mu_3);
x_minus_mu4 = xmu(n4,d,class4,mu_4);

% Calculate covariance matrix

cov1 = mtimes(transpose(x_minus_mu1),x_minus_mu1)/n1;
cov2 = mtimes(transpose(x_minus_mu2),x_minus_mu2)/n2;
cov3 = mtimes(transpose(x_minus_mu3),x_minus_mu3)/n3;
cov4 = mtimes(transpose(x_minus_mu4),x_minus_mu4)/n4;

% Calculating global varinces of all classes

 gv1 = finding_global_variance(cov1,d);
 gv2 = finding_global_variance(cov2,d);
 gv3 = finding_global_variance(cov3,d);
 gv4 = finding_global_variance(cov4,d);
 
 phi_1 = calculate_phi(X,mu_1,gv1,n);
 phi_2 = calculate_phi(X,mu_2,gv2,n);
 phi_3 = calculate_phi(X,mu_3,gv3,n);
 phi_4 = calculate_phi(X,mu_4,gv4,n);
 
 phi(:,1) = phi_1(:);
 phi(:,2) = phi_2(:);
 phi(:,3) = phi_3(:);
 phi(:,4) = phi_4(:);
 
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

for i = 1:n3
    t(i+t1,3) = 1;
end

for i = 1:n4
    t(i+t2,4) = 1;
end

% Calculate predicted value

y = predict_outputs(a,n,nc);

% Write weight update code

eta = 0.001;

w_star = weight_update(w,eta,y,t,global_phi,n,d,nc,phi);

y_pred = classify_samples(w_star,X,n,nc,phi);

Train_accuracy = calculate_accuracy(y_pred, n1,n2,n3,n4,n,nc)


%%% Evaluate on test data

[test_data,nc] = readCategoryFile('linear', 'test');

X_test = test_data(:,2:3);
n_test = numel(test_data(:,1));
nt1 = n_test/nc; nt2 = nt1; nt3 = nt1; nt4 = nt1;

 t_phi_1 = calculate_phi(X_test,mu_1,gv1,n_test);
 t_phi_2 = calculate_phi(X_test,mu_2,gv2,n_test);
 t_phi_3 = calculate_phi(X_test,mu_3,gv3,n_test);
 t_phi_4 = calculate_phi(X_test,mu_4,gv4,n_test);

 t_phi(:,1) = t_phi_1(:);
 t_phi(:,2) = t_phi_2(:);
 t_phi(:,3) = t_phi_3(:);
 t_phi(:,4) = t_phi_4(:);
 
 t_global_phi = calculate_global_phi_matrix(X_test,n_test,d);

 y_pred_test = classify_samples(w_star,X_test,n_test,nc,t_phi);
 Test_accuracy = calculate_accuracy(y_pred_test, nt1,nt2,nt3,nt4,n_test,nc)
 
 
%%% Evaluate on validation data

[val_data,nc] = readCategoryFile('linear', 'val');

X_val = val_data(:,2:3);
n_val = numel(val_data(:,1));
nv1 = n_val/nc; nv2 = nv1; nv3 = nv1; nv4 = nv1;

 v_phi_1 = calculate_phi(X_val,mu_1,gv1,n_val);
 v_phi_2 = calculate_phi(X_val,mu_2,gv2,n_val);
 v_phi_3 = calculate_phi(X_val,mu_3,gv3,n_val);
 v_phi_4 = calculate_phi(X_val,mu_4,gv4,n_val);

 v_phi(:,1) = v_phi_1(:);
 v_phi(:,2) = v_phi_2(:);
 v_phi(:,3) = v_phi_3(:);
 v_phi(:,4) = v_phi_4(:);
 
 v_global_phi = calculate_global_phi_matrix(X_val,n_val,d);

 y_pred_val = classify_samples(w_star,X_val,n_val,nc,v_phi);
 Val_accuracy = calculate_accuracy(y_pred_val, nv1,nv2,nv3,nv4,n_val,nc)
 
 % Plot for decision regions 
 
 [X1,X2] = meshgrid(-15:.1:20, -15:.1:20);
 X_1 = reshape(X1,351*351,1);
 X_2 = reshape(X2,351*351,1);
 
 N = numel(X_1);
 x = zeros(N,d); 
 for i = 1:N
         x(i,1) = X_1(i);
         x(i,2) = X_2(i);
 end
 
 x_phi_1 = calculate_phi(x,mu_1,gv1,N);
 x_phi_2 = calculate_phi(x,mu_2,gv2,N);
 x_phi_3 = calculate_phi(x,mu_3,gv3,N);
 x_phi_4 = calculate_phi(x,mu_4,gv4,N);

 x_phi(:,1) = x_phi_1(:);
 x_phi(:,2) = x_phi_2(:);
 x_phi(:,3) = x_phi_3(:);
 x_phi(:,4) = x_phi_4(:);
 
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
 
 figure;
 scatter(X_1(classes == 1),X_2(classes == 1),25,[0.9961, 0.9961, 0.5859],'filled') %yellow
 hold on;
 scatter(X_1(classes == 2),X_2(classes == 2),25,[0.9961, 0.6445, 0.9141],'filled') %pink
 hold on;
 scatter(X_1(classes == 3),X_2(classes == 3),25,[0.6758, 0.8555, 0.9961],'filled') %dark blue
 hold on;
 scatter(X_1(classes == 4),X_2(classes == 4),25,[0.2578, 0.9531, 0.7930],'filled') %green
 hold on;
 scatter(class1(:,1),class1(:,2),25,[0.8438,0.6602,0],'filled');
 hold on;
 scatter(class2(:,1),class2(:,2),25,[0.4648,0.0078,0.4570],'filled');
 hold on;
 scatter(class3(:,1),class3(:,2),25,[0.0078,0.3438,0.4648],'filled');
 hold on;
 scatter(class4(:,1),class4(:,2),25,[0.0117,0.4453,0.0703],'filled');
 xlabel('x1');
 ylabel('x2');
 title('Decision region prediction for logistic regression using Gaussian Basis Function');
 legend({'class1','class2','class3','class4'},'Location','north');
 
 
end
 
