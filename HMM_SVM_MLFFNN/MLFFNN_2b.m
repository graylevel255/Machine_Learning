% Read class train data

loc1 = '../data/nonlinearly_separable/class1_train.txt';
loc2 = '../data/nonlinearly_separable/class2_train.txt';

nc = 2; %no. of classes

class1 = importdata(loc1, ' ');
class2 = importdata(loc2, ' ');

n1 = numel(class1(:,1));
n2 = numel(class2(:,1));

N = n1 + n2;
d = numel(class1(1,:));

X = zeros(N,d);

for i = 1:n1
    X(i,:) = class1(i,:);
end

for i = 1:n2
    X(i+n1,:) = class2(i,:);
end

% Set x0 = 1 as coefficient of bias

% x0 = ones(N,1);
% for i=1:N
%     X(i,1) = x0(i);
% end

% Target output vector

t = zeros(N,nc);
for i = 1:n1
    t(i,1) = 1;
end

for i = 1:n2
    t(i+n1,2) = 1;
end

X = X';
t = t';

% Two layer feed forward neural network 
h1 = 15;
h2 = 15;
x1 =  X(1,:);
x2 =  X(2,:);
x = {x1;x2};
net = patternnet([15,15]);
net.numinputs = 2;
net.inputConnect = [1 1 ; 0 0; 0 0];
net = configure(net,x);
net = train(net,x,t);
net.trainParam.epochs = 1000;
% net.trainParam.lr = 0.01

y = net(x); 
% y_sg = sign(y{1,1});
% error = gsubtract(t,y);
performance = perform(net,t,y) 
tind = vec2ind(t); 
yind = vec2ind(y); 
% percentErrors = sum(tind*log(yind{1,1}))
% accuracy = 100 - percentErrors
% wts_h1 = net.IW{1,1}
% wts_h2 = net.IW{1,2}
% 
% b_h1 = net.b{1}
% b_h2 = net.b{2}
view(net)   % Viewing the network

wb = getwb(net);
[bias,IW,LW] = separatewb(net,wb);   % Separates bias and ip wts and layer wts
% op_bias = net.b{3}

% Compute output of 1st hidden layer nodes
% op_h1 = zeros(N,5);    % op of all nodes from ip layer 
w1h1 = IW{1,1};
w2h1 = IW{1,2};
bh1 = bias{1,1};
wh1 = zeros(h1,d);
    
% ip weights in a matrix
    wh1(:,1) = w1h1(:);
    wh1(:,2) = w2h1(:);


op_node_h1 = zeros(N,h1);

% op calculated for all samples for all nodes of first hidden layer

for i=1:N
    for j = 1:h1
        op_node_h1(i,j) = tanh(wh1(j,:)*X(:,i) + bh1(j));
    end
end
  
% Compute output of 2nd hidden layer nodes
LW2 = zeros(h1,h2);
LW2 = LW{2,1}';
% w2h2 = LW{2,1}(2,:);
bh2 = bias{2,1};

op_node_h2 = zeros(N,h2);

% op calculated for all samples for all nodes of 2nd hidden layer hidden layer 

 op_node_h2 = op_node_h1 * LW2;
    for j=1:h2
        op_node_h2(:,j) = tanh(op_node_h2(:,j)  + bh2(j));
    end

% op calculated for all samples for all nodes of output layer 

op_node_o = zeros(N,nc);
LW3 = zeros(h2,nc);
LW3 = LW{3,2}';
bh3 = bias{3,1};

op_node_o = op_node_h2 * LW3;

for j=1:nc
     op_node_o(:,j) = (op_node_o(:,j)  + bh3(j));
end

% Read test data and classify
loc_t1 = '../data/nonlinearly_separable/class1_test.txt';
loc_t2 = '../data/nonlinearly_separable/class2_test.txt';

class1_test = importdata(loc_t1, ' ');
class2_test = importdata(loc_t2, ' ');

nt_1 = numel(class1_test(:,1));
nt_2 = numel(class2_test(:,1));


% N is the number of samples

N_test = nt_1 + nt_2;

X_test = zeros(N_test,d+1);

for i = 1:nt_1
    X_test(i,2:3) = class1_test(i,:);
end

for i = 1:nt_2
    X_test(i+nt_1,2:3) = class2_test(i,:);
end

for i=1:N_test
    X_test(i,1) = 1;    % initializing biases
end

X_test = X_test';

xt1 = X_test(2,:);
xt2 = X_test(3,:);

test = {xt1; xt2};

% Test on test data

Y_pred = sim(net,test);
Y_pp = Y_pred{1,1};
Y_test = zeros(N_test,nc);

for i=1:N_test
    f1 = Y_pp(1,i);
    f2 = Y_pp(2,i);
    id = max(f1,f2);
    if id == f1
        Y_test(i,1) = 1;
    else
        Y_test(i,2) = 1;
    end
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

Test_accuracy = (ct1+ct2)/N_test*100


%%% Read and classify val data

loc_v1 = '../data/nonlinearly_separable/class1_val.txt';
loc_v2 = '../data/nonlinearly_separable/class2_val.txt';

class1_val = importdata(loc_t1, ' ');
class2_val = importdata(loc_t2, ' ');

nv_1 = numel(class1_val(:,1));
nv_2 = numel(class2_val(:,1));


% N is the number of samples

N_val = nv_1 + nv_2;

X_val = zeros(N_val,d+1);

for i = 1:nv_1
    X_val(i,2:3) = class1_val(i,:);
end

for i = 1:nv_2
    X_val(i+nv_1,2:3) = class2_val(i,:);
end

for i=1:N_val
    X_val(i,1) = 1;    % initializing biases
end

X_val = X_val';

xv1 = X_val(2,:);
xv2 = X_val(3,:);

val = {xv1; xv2};

% Test on validation data

Y_pred_val = sim(net,val);
Y_ppv = Y_pred_val{1,1};
Y_val = zeros(N_val,nc);

for i=1:N_val
    f1 = Y_ppv(1,i);
    f2 = Y_ppv(2,i);
    id = max(f1,f2);
    if id == f1
        Y_val(i,1) = 1;
    else
        Y_val(i,2) = 1;
    end
end

% Calculate accuracy of val data
cv1=0;
for i=1:nv_1
    cv1 = cv1 + Y_val(i,1);
end

cv2=0;
for i=1:nv_2
    cv2 = cv2 + Y_val(i+nt_1,2);
end

Val_accuracy = (cv1+cv2)/N_val*100



%%% DECISION REGION PLOT FOR O/P OF FIRST HIDDEN LAYER %%%

[X1,X2] = meshgrid(-15:0.1:15,-15:0.1:15);
X_1 = reshape(X1,301*301,1);
X_2 = reshape(X2,301*301,1);
num = numel(X_1);
dfp = zeros(d,num);

for i=1:num
    dfp(1,i) = X_1(i);
    dfp(2,i) = X_2(i);
end

[oph1,oph2,opo] = nn_plot(dfp,wh1,bh1,num,h1,LW2,h2,bh2,LW3,bh3,nc);
 
yp = zeros(num,d);
yp = predict_outputs(opo,num,nc);

classes = zeros(num,1);
 
 for i=1:num
     for j=1:nc
         if yp(i,j) == 1
             classes(i) = j;
         end
     end
 end

  figure;
 scatter(X_1(classes == 1),X_2(classes == 1),25,[0.9961, 0.9961, 0.5859],'filled') %yellow
 hold on;
 scatter(X_1(classes == 2),X_2(classes == 2),25,[0.6758, 0.8555, 0.9961],'filled') %dark blue
 hold on; 
 scatter(class1(:,1),class1(:,2),25,[0.8438,0.6602,0],'filled');
 hold on;
 scatter(class2(:,1),class2(:,2),25,[0.0078,0.3438,0.4648],'filled');
 hold on;
 xlabel('x1');
 ylabel('x2');
 title('Decision region prediction for multi-layer feed forward neural network');
 legend({'class1','class2'},'Location','north');