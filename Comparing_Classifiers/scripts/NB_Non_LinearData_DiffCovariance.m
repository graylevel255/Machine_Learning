disp('Naive Bayes classifier on non-linearly separable data with different covariance matrix')

loc1 = './datasets/group23/nonlinearly_separable/class1_train.txt';
loc2 = './datasets/group23/nonlinearly_separable/class2_train.txt';

class1 = importdata(loc1, ' ');
class2 = importdata(loc2, ' ');
n_c = 2;

%Club the data into a single matrix of dim 1000*2

n1 = numel(class1(:,1));
n2 = numel(class2(:,1));

num = n1 + n2;
d = numel(class1(1,:));

data = zeros(num,d);

for i = 1:n1
    data(i,1) = class1(i,1);
    data(i,2) = class1(i,2);
end

for i = 1:n2
    data(i+n1,1) = class2(i,1);
    data(i+n2,2) = class2(i,2);
end

labels = zeros(num,1);

for i = 1:n1
    labels(i) = 1;
end

for i = 1:n2
    labels(i+n1) = 2;
end


% Plot the data of 4 classes
% figure;
% gscatter(data(:,1),data(:,2),labels,'by','od');
% xlabel('x1');
% ylabel('x2');

% Determining prior probabilities of classes 1 to 4
P_c1 = n1/num;
P_c2 = n2/num;

% mean and covariance matrices for different classes

mu_1 = mean(class1,1);
mu_2 = mean(class2,1);

% x_minus_mu vector for class 1

x_minus_mu1 = xmu(n1,d,class1,mu_1);
x_minus_mu2 = xmu(n2,d,class2,mu_2);

% finding covariance marices
cov1 = mtimes(transpose(x_minus_mu1),x_minus_mu1)/n1;
cov2 = mtimes(transpose(x_minus_mu2),x_minus_mu2)/n2;

c1 = get_diag_cov(cov1,d);
c2 = get_diag_cov(cov2,d);

% slightly different value by using covariance function 
% 
%  cov1 = cov(x_minus_mu1);
%  cov2 = cov(x_minus_mu2);

% Constructing the discriminants

 g1 = calculate_g_of_x(data,mu_1,c1,P_c1);
 g2 = calculate_g_of_x(data,mu_2,c2,P_c2);

% Assigning class labels to trained data

g_star = zeros(num,1);
g_labels = zeros(num,1); 

for i = 1:num
    g_star(i) = max(g1(i),g2(i));
    
    if g_star(i) == g1(i)
        g_labels(i) = 1;
    else
        g_labels(i) = 2;
    end
end

% Plotting decision regions

[X1,X2] = meshgrid(-15:.1:15, -15:.1:15);
 X_1 = reshape(X1,301*301,1);
 X_2 = reshape(X2,301*301,1);
 
 n = numel(X_1);
 X = zeros(n,d); 
 
 for i = 1:n
         X(i,1) = X_1(i);
         X(i,2) = X_2(i);
 end
 
 g_train = zeros(n,n_c+4);

 g_t1 = calculate_g_of_x(X,mu_1,c1,P_c1);
 g_t2 = calculate_g_of_x(X,mu_2,c2,P_c2);
 
g_train(:,1) = X_1(:);
g_train(:,2) = X_2(:);
g_train(:,3) = g_t1(:,1);
g_train(:,4) = g_t2(:,1);
 
 for i = 1:n                 

         g_train(i,5) = max(g_train(i,3),g_train(i,4));
         
         if g_train(i,5) == g_train(i,3)
             g_train(i,6) = 1;
         else
             g_train(i,6) = 2;
         end
 end
 
%  
%  % Separating out data for plotting
 
 x_1 = g_train(:,1);
 x_2 = g_train(:,2);
 classes = g_train(:,6);
 
 lyellow = [0.9961, 0.9961, 0.5859];
 dyellow = [0.8438,0.6602,0];
 lblue = [0.6758, 0.8555, 0.9961];
 dblue = [0.0078,0.3438,0.4648];

 figure;
 scatter(x_1(classes == 1),x_2(classes == 1),25,lyellow,'filled')
 hold on;
 scatter(x_1(classes == 2),x_2(classes == 2),25,lblue,'filled')
 hold on; 
 scatter(class1(:,1),class1(:,2),25,dyellow,'filled')
 hold on;
 scatter(class2(:,1),class2(:,2),25,dblue,'filled');
 hold on;
 xlabel('x1');
 ylabel('x2');
 title('Decision region prediction for 2-class Naive Bayes classifier on Training data (Non-linearly separable data)');
 legend({'class1','class2'},'Location','north');

%%% Creating confusion matrix for training data
 
  conf_mat_train = zeros(n_c,n_c);
  c_num = zeros(n_c,n_c);
  
          for k = 1:n1
             if g_labels(k) == 1
              c_num(1,1) = c_num(1,1) + 1;
             else
              c_num(1,2) = c_num(1,2) + 1;
             end
          end
  
 
          for k = n1+1:n1+n2
             if g_labels(k) == 1
              c_num(2,1) = c_num(2,1) + 1;
             else
              c_num(2,2) = c_num(2,2) + 1;
             end
          end
          
   
  %%% Percentage values of final confusion matrix of training data %%%
  
  for i = 1:n_c
      for j = 1:n_c
          if i == 1
           conf_mat_train(i,j) = c_num(i,j)/n1*100;
          else
           conf_mat_train(i,j) = c_num(i,j)/n2*100;
          end
      end
  end

 conf_mat_train
%%% Calculating accuracy for training and validation %%%
cc = 0; mc = 0;

for i = 1:n_c
    cc = cc + c_num(i,i);
end

for i = 1:n_c
    for j = 1: n_c
        if i ~= j
        mc = mc + c_num(i,j);
        end
    end
end
 
correctly_classified = cc;
mis_classified = mc;
training_accuray = cc/(cc+mc)*100 

%%%% Testing the model for test files %%%%

loct1 = './datasets/group23/nonlinearly_separable/class1_test.txt';
loct2 = './datasets/group23/nonlinearly_separable/class2_test.txt';

classt1 = importdata(loct1, ' ');
classt2 = importdata(loct2, ' ');
n_ct = 2;

%Club the data into a single test matrix

nt1 = numel(classt1(:,1));
nt2 = numel(classt2(:,1));

size_t = nt1 + nt2;
data_t = zeros(size_t,d);

for i = 1:nt1
    data_t(i,1) = classt1(i,1);
    data_t(i,2) = classt1(i,2);
end

for i = 1:nt2
    data_t(i+nt1,1) = classt2(i,1);
    data_t(i+nt1,2) = classt2(i,2);
end

labels_t = zeros(size_t,1);
for i = 1:nt1
    labels_t(i,:) = 1;
end

for i = nt1+1:nt1+nt2
    labels_t(i,:) = 2;
end


% % Plotting decision regions for test files
% % Calculate gi's for each value of xi's

 g_test1 = calculate_g_of_x(data_t,mu_1,c1,P_c1);
 g_test2 = calculate_g_of_x(data_t,mu_2,c2,P_c2);
 
% Assigning class labels to trained data
g_star_t = zeros(size_t,1);
g_labels_t = zeros(size_t,1);
 
for i = 1:size_t
    g_star_t(i) = max(g_test1(i),g_test2(i));
    
    if g_star_t(i) == g_test1(i)
        g_labels_t(i) = 1;
    else
        g_labels_t(i) = 2;
    end
end


%%% Confusion matrix for test data %%%

conf_mat_test = zeros(n_c,n_c);
ct_num = zeros(n_c,n_c);

          for k = 1:nt1
             if g_labels_t(k) == 1
              ct_num(1,1) = ct_num(1,1) + 1;
             else
              ct_num(1,2) = ct_num(1,2) + 1;
             end
          end
  
 
          for k = nt1+1:nt1+nt2
             if g_labels_t(k) == 1
              ct_num(2,1) = ct_num(2,1) + 1;
             else
              ct_num(2,2) = ct_num(2,2) + 1;           
             end
          end
          
          
   
  %%% Percentage values of final confusion matrix of test data %%%
  
  for i = 1:n_c
      for j = 1:n_c
          if i == 1
           conf_mat_test(i,j) = ct_num(i,j)/nt1*100;
          else
           conf_mat_test(i,j) = ct_num(i,j)/nt2*100;
          end
      end
  end
 conf_mat_test


%%%%%% Testing the model for validation files  %%%%%

locv1 = './datasets/group23/nonlinearly_separable/class1_val.txt';
locv2 = './datasets/group23/nonlinearly_separable/class2_val.txt';

classv1 = importdata(locv1, ' ');
classv2 = importdata(locv2, ' ');
n_cv = 2;

%Club the data into a single validation data matrix

nv1 = numel(classv1(:,1));
nv2 = numel(classv2(:,1));

size_v = nv1 + nv2;
d = numel(classv1(1,:));

data_v = zeros(size_v,d);

for i = 1:nv1
    data_v(i,1) = classv1(i,1);
    data_v(i,2) = classv1(i,2);
end

for i = 1:nv2
    data_v(i+nv1,1) = classv2(i,1);
    data_v(i+nv1,2) = classv2(i,2);
end

labels_v = zeros(size_v,1);

% Calculate gi's for each value of xi's
 
g_val1 = calculate_g_of_x(data_v,mu_1,c1,P_c1);
g_val2 = calculate_g_of_x(data_v,mu_2,c2,P_c2);

% Assigning class labels to validation data
g_star_v = zeros(size_v,1);
g_labels_v = zeros(size_v,1);
 
for i = 1:size_v
    g_star_v(i) = max(g_val1(i),g_val2(i));
    
    if g_star_v(i) == g_val1(i)
        g_labels_v(i) = 1;
    else
        g_labels_v(i) = 2;
    end
end


%%% Calculating accuracy for validation data %%%

for i = 1:nv1
    labels_v(i,:) = 1;
end

for i = nv1+1:nv1+nv2
    labels_v(i,:) = 2;
end


cv_num = zeros(n_c,n_c);

         for k = 1:nv1
             if g_labels_v(k) == 1
              cv_num(1,1) = cv_num(1,1) + 1;
             else
              cv_num(1,2) = cv_num(1,2) + 1;
             end
          end
  
 
          for k = nv1+1:nv1+nv2
             if g_labels_v(k) == 1
              cv_num(2,1) = cv_num(2,1) + 1;
             else
              cv_num(2,2) = cv_num(2,2) + 1;
             end
          end
          
          
ccv = 0; mcv = 0;
for i = 1:n_c
    ccv = ccv + cv_num(i,i);
end

for i = 1:n_c
    for j = 1: n_c
        if i ~= j
        mcv = mcv + cv_num(i,j);
        end
    end
end
 
correctly_classified_val = ccv;
mis_classified_val = mcv;
val_accuray = ccv/(ccv+mcv)*100 

%%% Classification accuracy for test data

test_accuracy = sum(diag(ct_num))/sum(sum(ct_num))*100
