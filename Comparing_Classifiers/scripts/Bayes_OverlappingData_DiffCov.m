disp('Bayes classifier on overlapping data with different covariance matrix')

loc1 = './datasets/group23/overlapping/class1_train.txt';
loc2 = './datasets/group23/overlapping/class2_train.txt';
loc3 = './datasets/group23/overlapping/class3_train.txt';
loc4 = './datasets/group23/overlapping/class4_train.txt';

class1 = importdata(loc1, ' ');
class2 = importdata(loc2, ' ');
class3 = importdata(loc3, ' ');
class4 = importdata(loc4, ' ');
n_c = 4;

%Club the data into a single matrix of dim 1000*2

n1 = numel(class1(:,1));
n2 = numel(class2(:,1));
n3 = numel(class3(:,1));
n4 = numel(class4(:,1));

num = n1 + n2 + n3 + n4;
d = numel(class1(1,:));

data = zeros(num,d);

for i = 1:n1
    data(i,1) = class1(i,1);
    data(i,2) = class1(i,2);
end

for i = 1:n2
    data(i+250,1) = class2(i,1);
    data(i+250,2) = class2(i,2);
end

for i = 1:n3
    data(i+500,1) = class3(i,1);
    data(i+500,2) = class3(i,2);
end

for i = 1:n4
    data(i+750,1) = class4(i,1);
    data(i+750,2) = class4(i,2);
end

labels = zeros(num,1);

for i = 1:n1
    labels(i) = 1;
end

for i = 1:n2
    labels(i+n1) = 2;
end

for i = 1:n3
    labels(i+n1+n2) = 3;
end

for i = 1:n4
    labels(i+n1+n2+n3) = 4;
end

% Plot the data of 4 classes
% figure;
% gscatter(data(:,1),data(:,2),labels,'gmrb','ox+d');
% xlabel('x1');
% ylabel('x2');

% Determining prior probabilities of classes 1 to 4
P_c1 = n1/num;
P_c2 = n2/num;
P_c3 = n3/num;
P_c4 = n4/num;

% mean and covariance matrices for different classes

mu_1 = mean(class1,1);
mu_2 = mean(class2,1);
mu_3 = mean(class3,1);
mu_4 = mean(class4,1);

% x_minus_mu vector for class 1

x_minus_mu1 = xmu(n1,d,class1,mu_1);
x_minus_mu2 = xmu(n2,d,class2,mu_2);
x_minus_mu3 = xmu(n3,d,class3,mu_3);
x_minus_mu4 = xmu(n4,d,class4,mu_4);

% finding covariance marices
cov1 = mtimes(transpose(x_minus_mu1),x_minus_mu1)/n1;
cov2 = mtimes(transpose(x_minus_mu2),x_minus_mu2)/n2;
cov3 = mtimes(transpose(x_minus_mu3),x_minus_mu3)/n3;
cov4 = mtimes(transpose(x_minus_mu4),x_minus_mu4)/n4;

% slightly different value by using covariance function 
% 
%  cov1 = cov(x_minus_mu1);
%  cov2 = cov(x_minus_mu2);
%  cov3 = cov(x_minus_mu3);
%  cov4 = cov(x_minus_mu4);

% Constructing the discriminants

 g1 = calculate_g_of_x(data,mu_1,cov1,P_c1);
 g2 = calculate_g_of_x(data,mu_2,cov2,P_c2);
 g3 = calculate_g_of_x(data,mu_3,cov3,P_c3);
 g4 = calculate_g_of_x(data,mu_4,cov4,P_c4);

% Assigning class labels to trained data

g_star = zeros(num,1);
g_labels = zeros(num,1); 

for i = 1:num
    g_star(i) = mymax(g1(i),g2(i),g3(i),g4(i));
    
    if g_star(i) == g1(i)
        g_labels(i) = 1;
    elseif g_star(i) == g2(i)
        g_labels(i) = 2;
    elseif g_star(i) == g3(i)
        g_labels(i) = 3;
    else
        g_labels(i) = 4;
    end
end

% Plotting decision regions

[X1,X2] = meshgrid(-15:.1:20, -15:.1:20);
 X_1 = reshape(X1,351*351,1);
 X_2 = reshape(X2,351*351,1);
 
 n = numel(X_1);
 X = zeros(n,d); 
 
 for i = 1:n
         X(i,1) = X_1(i);
         X(i,2) = X_2(i);
 end
 
 g_train = zeros(n,n_c+4);

 g_t1 = calculate_g_of_x(X,mu_1,cov1,P_c1);
 g_t2 = calculate_g_of_x(X,mu_2,cov2,P_c2);
 g_t3 = calculate_g_of_x(X,mu_3,cov3,P_c3);
 g_t4 = calculate_g_of_x(X,mu_4,cov4,P_c4);
 
 
g_train(:,1) = X_1(:);
g_train(:,2) = X_2(:);
g_train(:,3) = g_t1(:,1);
g_train(:,4) = g_t2(:,1);
g_train(:,5) = g_t3(:,1);
g_train(:,6) = g_t4(:,1); 
 
 for i = 1:n                 

         g_train(i,7) = mymax(g_train(i,3),g_train(i,4),g_train(i,5),g_train(i,6));
         
         if g_train(i,7) == g_train(i,3)
             g_train(i,8) = 1;
         elseif g_train(i,7) == g_train(i,4)
             g_train(i,8) = 2;
         elseif g_train(i,7) == g_train(i,5)
             g_train(i,8) = 3;
         else
             g_train(i,8) = 4;
         end
 end
 
%  
%  % Separating out data for plotting
 
 x_1 = g_train(:,1);
 x_2 = g_train(:,2);
 classes = g_train(:,8);
 
 figure;
 scatter(x_1(classes == 1),x_2(classes == 1),25,[0.9961, 0.9961, 0.5859],'filled') %yellow
 hold on;
 scatter(x_1(classes == 2),x_2(classes == 2),25,[0.9961, 0.6445, 0.9141],'filled') %pink
 hold on;
 scatter(x_1(classes == 3),x_2(classes == 3),25,[0.6758, 0.8555, 0.9961],'filled') %dark blue
 hold on;
 scatter(x_1(classes == 4),x_2(classes == 4),25,[0.2578, 0.9531, 0.7930],'filled') %green
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
title('Decision region prediction for 4-class Bayes classifier on Training data (Overlapping data)');
legend({'class1','class2','class3','class4'},'Location','north');

%  
%  % Creating confusion matrix for training data
 
  conf_mat_train = zeros(n_c,n_c);
  c_num = zeros(n_c,n_c);
  
          for k = 1:n1
             if g_labels(k) == 1
              c_num(1,1) = c_num(1,1) + 1;
             elseif g_labels(k) == 2
              c_num(1,2) = c_num(1,2) + 1;
             elseif g_labels(k) == 3
              c_num(1,3) = c_num(1,3) + 1;
             else
              c_num(1,4) = c_num(1,4) + 1;
             end
          end
  
 
          for k = n1+1:n1+n2
             if g_labels(k) == 1
              c_num(2,1) = c_num(2,1) + 1;
             elseif g_labels(k) == 2
              c_num(2,2) = c_num(2,2) + 1;
             elseif g_labels(k) == 3
              c_num(2,3) = c_num(2,3) + 1;
             else
              c_num(2,4) = c_num(2,4) + 1;
             end
          end
          
          for k = n1+n2+1:n1+n2+n3
             if g_labels(k) == 1
              c_num(3,1) = c_num(3,1) + 1;
             elseif g_labels(k) == 2
              c_num(3,2) = c_num(3,2) + 1;
             elseif g_labels(k) == 3
              c_num(3,3) = c_num(3,3) + 1;
             else
              c_num(3,4) = c_num(3,4) + 1;
             end
          end
          
          for k = n1+n2+n3+1:n1+n2+n3+n4
             if g_labels(k) == 1
              c_num(4,1) = c_num(4,1) + 1;
             elseif g_labels(k) == 2
              c_num(4,2) = c_num(4,2) + 1;
             elseif g_labels(k) == 3
              c_num(4,3) = c_num(4,3) + 1;
             else
              c_num(4,4) = c_num(4,4) + 1;
             end
          end
          
   
  %%% Percentage values of final confusion matrix of training data %%%
  
  for i = 1:n_c
      for j = 1:n_c
          if i == 1
           conf_mat_train(i,j) = c_num(i,j)/n1*100;
          elseif i == 2
           conf_mat_train(i,j) = c_num(i,j)/n2*100;
          elseif i == 3
           conf_mat_train(i,j) = c_num(i,j)/n3*100;
          else
           conf_mat_train(i,j) = c_num(i,j)/n4*100;
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

loct1 = './datasets/group23/overlapping/class1_test.txt';
loct2 = './datasets/group23/overlapping/class2_test.txt';
loct3 = './datasets/group23/overlapping/class3_test.txt';
loct4 = './datasets/group23/overlapping/class4_test.txt';

classt1 = importdata(loct1, ' ');
classt2 = importdata(loct2, ' ');
classt3 = importdata(loct3, ' ');
classt4 = importdata(loct4, ' ');
n_ct = 4;

%Club the data into a single test matrix

nt1 = numel(classt1(:,1));
nt2 = numel(classt2(:,1));
nt3 = numel(classt3(:,1));
nt4 = numel(classt4(:,1));

size_t = nt1 + nt2 + nt3 + nt4;
data_t = zeros(size_t,d);

for i = 1:nt1
    data_t(i,1) = classt1(i,1);
    data_t(i,2) = classt1(i,2);
end

for i = 1:nt2
    data_t(i+nt1,1) = classt2(i,1);
    data_t(i+nt1,2) = classt2(i,2);
end

for i = 1:nt3
    data_t(i+nt1+nt2,1) = classt3(i,1);
    data_t(i+nt1+nt2,2) = classt3(i,2);
end

for i = 1:nt4
    data_t(i+nt1+nt2+nt3,1) = classt4(i,1);
    data_t(i+nt1+nt2+nt3,2) = classt4(i,2);
end

labels_t = zeros(size_t,1);
for i = 1:nt1
    labels_t(i,:) = 1;
end

for i = nt1+1:nt1+nt2
    labels_t(i,:) = 2;
end

for i = nt1+nt2+1:nt1+nt2+nt3
    labels_t(i,:) = 3;
end

for i = nt1+nt2+nt3+1:nt1+nt2+nt3+nt4
    labels_t(i,:) = 4;
end

% % Plotting decision regions for test files
% % Calculate gi's for each value of xi's

 g_test1 = calculate_g_of_x(data_t,mu_1,cov1,P_c1);
 g_test2 = calculate_g_of_x(data_t,mu_2,cov2,P_c2);
 g_test3 = calculate_g_of_x(data_t,mu_3,cov3,P_c3);
 g_test4 = calculate_g_of_x(data_t,mu_4,cov4,P_c4);
 
% Assigning class labels to trained data
g_star_t = zeros(size_t,1);
g_labels_t = zeros(size_t,1);
 
for i = 1:size_t
    g_star_t(i) = mymax(g_test1(i),g_test2(i),g_test3(i),g_test4(i));
    
    if g_star_t(i) == g_test1(i)
        g_labels_t(i) = 1;
    elseif g_star_t(i) == g_test2(i)
        g_labels_t(i) = 2;
    elseif g_star_t(i) == g_test3(i)
        g_labels_t(i) = 3;
    else
        g_labels_t(i) = 4;
    end
end


%%% Confusion matrix for test data %%%

conf_mat_test = zeros(n_c,n_c);
ct_num = zeros(n_c,n_c);

          for k = 1:nt1
             if g_labels_t(k) == 1
              ct_num(1,1) = ct_num(1,1) + 1;
             elseif g_labels_t(k) == 2
              ct_num(1,2) = ct_num(1,2) + 1;
             elseif g_labels_t(k) == 3
              ct_num(1,3) = ct_num(1,3) + 1;
             else
              ct_num(1,4) = ct_num(1,4) + 1;
             end
          end
  
 
          for k = nt1+1:nt1+nt2
             if g_labels_t(k) == 1
              ct_num(2,1) = ct_num(2,1) + 1;
             elseif g_labels_t(k) == 2
              ct_num(2,2) = ct_num(2,2) + 1;
             elseif g_labels_t(k) == 3
              ct_num(2,3) = ct_num(2,3) + 1;
             else
              ct_num(2,4) = ct_num(2,4) + 1;
             end
          end
          
          for k = nt1+nt2+1:nt1+nt2+nt3
             if g_labels_t(k) == 1
              ct_num(3,1) = ct_num(3,1) + 1;
             elseif g_labels_t(k) == 2
              ct_num(3,2) = ct_num(3,2) + 1;
             elseif g_labels_t(k) == 3
              ct_num(3,3) = ct_num(3,3) + 1;
             else
              ct_num(3,4) = ct_num(3,4) + 1;
             end
          end
          
          for k = nt1+nt2+nt3+1:nt1+nt2+nt3+nt4
             if g_labels_t(k) == 1
              ct_num(4,1) = ct_num(4,1) + 1;
             elseif g_labels_t(k) == 2
              ct_num(4,2) = ct_num(4,2) + 1;
             elseif g_labels_t(k) == 3
              ct_num(4,3) = ct_num(4,3) + 1;
             else
              ct_num(4,4) = ct_num(4,4) + 1;
             end
          end
          
   
  %%% Percentage values of final confusion matrix of test data %%%
  
  for i = 1:n_c
      for j = 1:n_c
          if i == 1
           conf_mat_test(i,j) = ct_num(i,j)/nt1*100;
          elseif i == 2
           conf_mat_test(i,j) = ct_num(i,j)/nt2*100;
          elseif i == 3
           conf_mat_test(i,j) = ct_num(i,j)/nt3*100;
          else
           conf_mat_test(i,j) = ct_num(i,j)/nt4*100;
          end
      end
 end

conf_mat_test
%%%%%% Testing the model for validation files  %%%%%

locv1 = './datasets/group23/overlapping/class1_val.txt';
locv2 = './datasets/group23/overlapping/class2_val.txt';
locv3 = './datasets/group23/overlapping/class3_val.txt';
locv4 = './datasets/group23/overlapping/class4_val.txt';

classv1 = importdata(locv1, ' ');
classv2 = importdata(locv2, ' ');
classv3 = importdata(locv3, ' ');
classv4 = importdata(locv4, ' ');
n_cv = 4;

%Club the data into a single validation data matrix

nv1 = numel(classv1(:,1));
nv2 = numel(classv2(:,1));
nv3 = numel(classv3(:,1));
nv4 = numel(classv4(:,1));

size_v = nv1 + nv2 + nv3 + nv4;
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

for i = 1:nv3
    data_v(i+nv1+nv2,1) = classv3(i,1);
    data_v(i+nv1+nv2,2) = classv3(i,2);
end

for i = 1:nv4
    data_v(i+nv1+nv2+nv3,1) = classv4(i,1);
    data_v(i+nv1+nv2+nv3,2) = classv4(i,2);
end

labels_v = zeros(size_v,1);
l_v = char(labels_t);


% Calculate gi's for each value of xi's
 
g_val1 = calculate_g_of_x(data_v,mu_1,cov1,P_c1);
g_val2 = calculate_g_of_x(data_v,mu_2,cov2,P_c2);
g_val3 = calculate_g_of_x(data_v,mu_3,cov3,P_c3);
g_val4 = calculate_g_of_x(data_v,mu_4,cov4,P_c4);

% Assigning class labels to validation data
g_star_v = zeros(size_v,1);
g_labels_v = zeros(size_v,1);
 
for i = 1:size_v
    g_star_v(i) = mymax(g_val1(i),g_val2(i),g_val3(i),g_val4(i));
    
    if g_star_v(i) == g_val1(i)
        g_labels_v(i) = 1;
    elseif g_star_v(i) == g_val2(i)
        g_labels_v(i) = 2;
    elseif g_star_v(i) == g_val3(i)
        g_labels_v(i) = 3;
    else
        g_labels_v(i) = 4;
    end
end


%%% Calculating accuracy for validation data %%%

for i = 1:nv1
    labels_v(i,:) = 1;
end

for i = nv1+1:nv1+nv2
    labels_v(i,:) = 2;
end

for i = nv1+nv2+1:nv1+nv2+nv3
    labels_v(i,:) = 3;
end

for i = nv1+nv2+nv3+1:nv1+nv2+nv3+nv4
    labels_v(i,:) = 4;
end

cv_num = zeros(n_c,n_c);

         for k = 1:nv1
             if g_labels_v(k) == 1
              cv_num(1,1) = cv_num(1,1) + 1;
             elseif g_labels_v(k) == 2
              cv_num(1,2) = cv_num(1,2) + 1;
             elseif g_labels_v(k) == 3
              cv_num(1,3) = cv_num(1,3) + 1;
             else
              cv_num(1,4) = cv_num(1,4) + 1;
             end
          end
  
 
          for k = nv1+1:nv1+nv2
             if g_labels_v(k) == 1
              cv_num(2,1) = cv_num(2,1) + 1;
             elseif g_labels_v(k) == 2
              cv_num(2,2) = cv_num(2,2) + 1;
             elseif g_labels_v(k) == 3
              cv_num(2,3) = cv_num(2,3) + 1;
             else
              cv_num(2,4) = cv_num(2,4) + 1;
             end
          end
          
          for k = nv1+nv2+1:nv1+nv2+nv3
             if g_labels_v(k) == 1
              cv_num(3,1) = cv_num(3,1) + 1;
             elseif g_labels_v(k) == 2
              cv_num(3,2) = cv_num(3,2) + 1;
             elseif g_labels_v(k) == 3
              cv_num(3,3) = cv_num(3,3) + 1;
             else
              cv_num(3,4) = cv_num(3,4) + 1;
             end
          end
          
          for k = nv1+nv2+nv3+1:nv1+nv2+nv3+nv4
             if g_labels_v(k) == 1
              cv_num(4,1) = cv_num(4,1) + 1;
             elseif g_labels_v(k) == 2
              cv_num(4,2) = cv_num(4,2) + 1;
             elseif g_labels_v(k) == 3
              cv_num(4,3) = cv_num(4,3) + 1;
             else
              cv_num(4,4) = cv_num(4,4) + 1;
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