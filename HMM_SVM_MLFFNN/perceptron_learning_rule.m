% Read class train data

loc1 = '../data/linearly_separable/class1_train.txt';
loc2 = '../data/linearly_separable/class2_train.txt';
loc3 = '../data/linearly_separable/class3_train.txt';
loc4 = '../data/linearly_separable/class4_train.txt';

nc = 4; %no. of classes

class1 = importdata(loc1, ' ');
class2 = importdata(loc2, ' ');
class3 = importdata(loc3, ' ');
class4 = importdata(loc4, ' ');

n1 = numel(class1(:,1));
n2 = numel(class2(:,1));
n3 = numel(class3(:,1));
n4 = numel(class4(:,1));

% N is the number of samples

N = n1 + n2 + n3 + n4;
d = numel(class1(1,:));

% define input vector X

X = zeros(N,d+1);

for i = 1:n1
    X(i,2:3) = class1(i,:);
end

for i = 1:n2
    X(i+n1,2:3) = class2(i,:);
end

t1 = n1 + n2;
for i = 1:n3
    X(i+t1,2:3) = class3(i,:);
end

t2 = t1 + n3;
for i = 1:n4
    X(i+t2,2:3) = class4(i,:);
end

% Set x0 = 1 as coefficient of bias

x0 = ones(N);
for i=1:N
    X(i,1) = x0(i);
end


% Normalize data
% x = X(:,2:3);
% 
% range = max(x(:)) - min(x(:));
% m01 = (x - min(x(:))) / range;
% mOut = 2 * m01 - 1;
% 
% for i=1:N
%     for j=1:d
%         X(i,j+1) = m01(i,j);
%     end
% end


% Target output vector

t = zeros(N,nc);
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

% Randomly initialize weight vector where wt(1,i) are biases
% loc_wt = '../data/weights3.txt';
% wt = randn(4,3);
wt = [-2.2015, 0.5256, -0.3202;
      -0.7745, 1.5233, 0.8175;
      -1.3933, 1.7985, 0.4902;
      -0.3862, -0.1169,0.7653];
% wt = importdata(loc_wt,' ');
f = zeros(nc,1);
y = zeros(N,4);
eta = 0.1;

% check error

[w_new,y] = update_weights(wt,t,y,nc,N,X,eta);
sse = calc_error(t,y,N,nc);


% print accuracy
c1=0;
for i=1:n1
    c1 = c1 + y(i,1);
end

c2=0;
for i=1:n2
    c2 = c2 + y(i+n1,2);
end

c3=0;
for i=1:n3
    c3 = c3 + y(i+t1,3);
end

c4=0;
for i=1:n1
    c4 = c4 + y(i+t2,4);
end

Train_accuracy = (c1+c2+c3+c4)/N*100


% Write classifier

actual_op = percep_classifier(w_new, X, N, nc);

% Read test data and classify
loc_t1 = '../data/linearly_separable/class1_test.txt';
loc_t2 = '../data/linearly_separable/class2_test.txt';
loc_t3 = '../data/linearly_separable/class3_test.txt';
loc_t4 = '../data/linearly_separable/class4_test.txt';

class1_test = importdata(loc_t1, ' ');
class2_test = importdata(loc_t2, ' ');
class3_test = importdata(loc_t3, ' ');
class4_test = importdata(loc_t4, ' ');

nt_1 = numel(class1_test(:,1));
nt_2 = numel(class2_test(:,1));
nt_3 = numel(class3_test(:,1));
nt_4 = numel(class4_test(:,1));

% N is the number of samples

N_test = nt_1 + nt_2 + nt_3 + nt_4;

X_test = zeros(N_test,d+1);

for i = 1:nt_1
    X_test(i,2:3) = class1_test(i,:);
end

for i = 1:nt_2
    X_test(i+nt_1,2:3) = class2_test(i,:);
end

tp1 = nt_1 + nt_2;
for i = 1:nt_3
    X_test(i+tp1,2:3) = class3_test(i,:);
end

tp2 = tp1 + nt_3;
for i = 1:nt_4
    X_test(i+tp2,2:3) = class4_test(i,:);
end

for i=1:N_test
    X_test(i,1) = 1;    % initializing biases
end

y_test =  percep_classifier(w_new, X_test, N_test, nc);

% Print test accuracy

ct1=0;
for i=1:nt_1
    ct1 = ct1 + y_test(i,1);
end

ct2=0;
for i=1:nt_2
    ct2 = ct2 + y_test(i+nt_1,2);
end

ct3=0;
tp1 = nt_1 + nt_2;
for i=1:nt_3
    ct3 = ct3 + y_test(i+tp1,3);
end

ct4=0;
tp2 = tp1 + nt_3;
for i=1:nt_4
    ct4 = ct4 + y_test(i+tp2,4);
end

Test_accuracy = (ct1+ct2+ct3+ct4)/N_test*100


%%% SCATTER PLOT %%%

[X1,X2] = meshgrid(-15:.1:20, -15:.1:20);
 X_1 = reshape(X1,351*351,1);
 X_2 = reshape(X2,351*351,1);
 
 num = numel(X_1);
 x = zeros(num,d+1); 
 for i = 1:num
         x(i,2) = X_1(i);
         x(i,3) = X_2(i);
 end
 
 % set bias
 
 x0b = ones(num,1);
 x(:,1) = x0b(:);

 x_pred = percep_classifier(w_new, x, num, nc);
 
 classes = zeros(num,1);
 
 for i=1:num
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
 title('Decision region prediction for Perceptron Learning Rule');
 legend({'class1','class2','class3','class4'},'Location','north');

 



