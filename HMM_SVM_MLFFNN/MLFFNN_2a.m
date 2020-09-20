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

x0 = ones(N,1);
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

X = X';
t = t';


% Two layer feed forward neural network 

x1 =  X(2,:);
x2 =  X(3,:);
x = {x1;x2};
net = patternnet([3,3]);
net.numinputs = 2;
net.inputConnect = [1 1 ; 0 0; 0 0];
net = configure(net,x);
net = train(net,x,t);
net.trainParam.epochs = 1000;
net.trainParam.lr = 5;

% best model for eta = 0.01

% net.trainFcn = 'trains';
% [net,tr] = train(net,x,t); 
y = net(x); 
error = gsubtract(t,y);
performance = perform(net,t,y) 
tind = vec2ind(t); 
yind = vec2ind(y); 
percentErrors = sum(tind ~= yind{1,1})/numel(tind)
%Train_accuracy = 100 - percentErrors
net.IW{1,1}; net.b{1}; %net.LW {1};

view(net)   % Viewing the network

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

X_test = X_test';

xt1 = X_test(2,:);
xt2 = X_test(3,:);

test = {xt1; xt2};

% Evaluate performance on test data
  Y_pred = sim(net,test);
  Y_pp = Y_pred{1,1};
  Y_test = zeros(N_test,nc);

% YPred = predict(net,test)
for i=1:N_test
    f1 = Y_pp(1,i);
    f2 = Y_pp(2,i);
    f3 = Y_pp(3,i);
    f4 = Y_pp(4,i);
    id = mymax(f1,f2,f3,f4);
    Y_test(i,id) = 1;
end

% Calculate accuracy of test data

ct1=0;
for i=1:nt_1
    ct1 = ct1 + Y_test(i,1);
end

ct2=0;
for i=1:nt_2
    ct2 = ct2 + Y_test(i+nt_1,2);
end

ct3=0;
tp1 = nt_1 + nt_2;
for i=1:nt_3
    ct3 = ct3 + Y_test(i+tp1,3);
end

ct4=0;
tp2 = tp1 + nt_3;
for i=1:nt_4
    ct4 = ct4 + Y_test(i+tp2,4);
end

Test_accuracy = (ct1+ct2+ct3+ct4)/N_test*100


% Read validation data and classify
loc_v1 = '../data/linearly_separable/class1_val.txt';
loc_v2 = '../data/linearly_separable/class2_val.txt';
loc_v3 = '../data/linearly_separable/class3_val.txt';
loc_v4 = '../data/linearly_separable/class4_val.txt';

class1_val = importdata(loc_v1, ' ');
class2_val = importdata(loc_v2, ' ');
class3_val = importdata(loc_v3, ' ');
class4_val = importdata(loc_v4, ' ');

nv_1 = numel(class1_val(:,1));
nv_2 = numel(class2_val(:,1));
nv_3 = numel(class3_val(:,1));
nv_4 = numel(class4_val(:,1));

% N is the number of samples

N_val = nv_1 + nv_2 + nv_3 + nv_4;

X_val = zeros(N_val,d+1);

for i = 1:nv_1
    X_val(i,2:3) = class1_val(i,:);
end

for i = 1:nv_2
    X_val(i+nv_1,2:3) = class2_val(i,:);
end

tv1 = nv_1 + nv_2;
for i = 1:nv_3
    X_val(i+tv1,2:3) = class3_val(i,:);
end

tv2 = tv1 + nv_3;
for i = 1:nv_4
    X_val(i+tv2,2:3) = class4_val(i,:);
end

for i=1:N_val
    X_val(i,1) = 1;    % initializing biases
end

X_val = X_val';

xv1 = X_val(2,:);
xv2 = X_val(3,:);

val = {xv1; xv2};

% Evaluate performance on test data
  Y_pred_val = sim(net,val);
  Y_ppv = Y_pred_val{1,1};
  Y_val = zeros(N_val,nc);

% YPred = predict(net,test)
for i=1:N_val
    f1 = Y_ppv(1,i);
    f2 = Y_ppv(2,i);
    f3 = Y_ppv(3,i);
    f4 = Y_ppv(4,i);
    id = mymax(f1,f2,f3,f4);
    Y_val(i,id) = 1;
end

% Calculate accuracy of test data

cv1=0;
for i=1:nv_1
    cv1 = cv1 + Y_val(i,1);
end

cv2=0;
for i=1:nv_2
    cv2 = cv2 + Y_val(i+nv_1,2);
end

cv3=0;
tmp1 = nv_1 + nv_2;
for i=1:nv_3
    cv3 = cv3 + Y_val(i+tmp1,3);
end

cv4=0;
tmp2 = tmp1 + nv_3;
for i=1:nv_4
    cv4 = cv4 + Y_val(i+tmp2,4);
end

Val_accuracy = (cv1+cv2+cv3+cv4)/N_val*100