%LINEAR
[traindata,nc] = readCategoryFile('linear','train');
d = 2;
M = 1;
D = factorial(M+d)/(factorial(M)*factorial(d));
ntrain = size(traindata,1);
classsize = ntrain/nc;
[trainaccuracy,w] = polylogisticregression(traindata,d,nc,M);
trainaccuracy

% Test data
[testdata,nc] = readCategoryFile('linear','test');
ntest = size(testdata,1);
x1_test = testdata(:,2);
x2_test = testdata(:,3);
phi_test = zeros(ntest,D);
d_index = 1;
for m = 0:M
    for i = 0:m
       j = m-i;
       phi_test(:,d_index) = (x1_test.^j).*(x2_test.^i);
       d_index = d_index+1;
    end
end
a_test = phi_test*w';
y_test = logistic_a_to_y(a_test);
testaccuracy = logistic_accuracy(y_test)

% Validation data
[valdata,nc] = readCategoryFile('linear','val');
nval = size(valdata,1);
x1_val = valdata(:,2);
x2_val = valdata(:,3);
phi_val = zeros(nval,D);
d_index = 1;
for m = 0:M
    for i = 0:m
       j = m-i;
       phi_val(:,d_index) = (x1_val.^j).*(x2_val.^i);
       d_index = d_index+1;
    end
end
a_val = phi_val*w';
y_val = logistic_a_to_y(a_val);
valaccuracy = logistic_accuracy(y_val)


% Train - mesh grid
%TRAIN MESH
% min x1 = -14.1950 ; max x1 = 15.9270;
% min x2 = -11.9080 ; max x2 = 19.4320;
[ax1,ax2] = meshgrid(-15:0.1:20);
ax1 = reshape(ax1,351*351,1);
ax2 = reshape(ax2,351*351,1);
linseptrainmesh = [ax1,ax2];
ntm = size(linseptrainmesh,1);
phi_trainmesh = zeros(ntm,D);
d_index = 1;
for m = 0:M
    for i = 0:m
       j = m-i;
       phi_trainmesh(:,d_index) = (ax1.^j).*(ax2.^i);
       d_index = d_index+1;
    end
end
a_trainmesh = phi_trainmesh*w';
y_trainmesh = logistic_a_to_y(a_trainmesh);
predictedClass = zeros(ntm,1);
modelData = [linseptrainmesh,predictedClass];
for i = 1:ntm
    if(y_trainmesh(i,1)==1)
        class = 1;
    end
    if(y_trainmesh(i,2)==1)
        class = 2;
    end
    if(y_trainmesh(i,3)==1)
        class = 3;
    end
    if(y_trainmesh(i,4)==1)
        class = 4;
    end
    modelData(i,3) = class;
end
traindata_correctformat = zeros(ntrain,d+1);
traindata_correctformat(:,1) = traindata(:,2);
traindata_correctformat(:,2) = traindata(:,3);
traindata_correctformat(:,3) = traindata(:,1);
plotClassData('linear',modelData,traindata_correctformat,2);