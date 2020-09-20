% NON-LINEAR
[traindata_nl,nc_nl] = readCategoryFile('nonlinear','train');
d = 2;
M = 2;
D = factorial(M+d)/(factorial(M)*factorial(d));
ntrain_nl = size(traindata_nl,1);
classsize_nl = ntrain_nl/nc_nl;
[trainaccuracy_nl,w_nl] = polylogisticregression(traindata_nl,d,nc_nl,M);
trainaccuracy_nl

% Test data
[testdata_nl,nc_nl] = readCategoryFile('nonlinear','test');
ntest = size(testdata_nl,1);
x1_test_nl = testdata_nl(:,2);
x2_test_nl = testdata_nl(:,3);
phi_test_nl = zeros(ntest,D);
d_index = 1;
for m = 0:M
    for i = 0:m
       j = m-i;
       phi_test_nl(:,d_index) = (x1_test_nl.^j).*(x2_test_nl.^i);
       d_index = d_index+1;
    end
end
a_test_nl = phi_test_nl*w_nl';
y_test_nl = logistic_a_to_y(a_test_nl);
testaccuracy = logistic_accuracy(y_test_nl)

% Validation data
[valdata_nl,nc_nl] = readCategoryFile('nonlinear','val');
nval = size(valdata_nl,1);
x1_val_nl = valdata_nl(:,2);
x2_val_nl = valdata_nl(:,3);
phi_val_nl = zeros(nval,D);
d_index = 1;
for m = 0:M
    for i = 0:m
       j = m-i;
       phi_val_nl(:,d_index) = (x1_val_nl.^j).*(x2_val_nl.^i);
       d_index = d_index+1;
    end
end
a_val_nl = phi_val_nl*w_nl';
y_val_nl = logistic_a_to_y(a_val_nl);
valaccuracy = logistic_accuracy(y_val_nl)


% Train - mesh grid
% min x1 = -13.2170 ; max x1 = 14.5240;
% min x2 = -11.9230 ; max x2 = 12.9640;
[ax1_nl,ax2_nl] = meshgrid(-15:0.1:15);
ax1_nl = reshape(ax1_nl,301*301,1);
ax2_nl = reshape(ax2_nl,301*301,1);
nonlinseptrainmesh = [ax1_nl,ax2_nl];
ntm = size(nonlinseptrainmesh,1);
phi_trainmesh_nl = zeros(ntm,D);
d_index = 1;
for m = 0:M
    for i = 0:m
       j = m-i;
       phi_trainmesh_nl(:,d_index) = (ax1_nl.^j).*(ax2_nl.^i);
       d_index = d_index+1;
    end
end
a_trainmesh_nl = phi_trainmesh_nl*w_nl';
y_trainmesh_nl = logistic_a_to_y(a_trainmesh_nl);
predictedClass_nl = zeros(ntm,1);
modelData_nl = [nonlinseptrainmesh,predictedClass_nl];
for i = 1:ntm
    if(y_trainmesh_nl(i,1)==1)
        class = 1;
    end
    if(y_trainmesh_nl(i,2)==1)
        class = 2;
    end
    modelData_nl(i,3) = class;
end
traindata_correctformat_nl = zeros(ntrain_nl,d+1);
traindata_correctformat_nl(:,1) = traindata_nl(:,2);
traindata_correctformat_nl(:,2) = traindata_nl(:,3);
traindata_correctformat_nl(:,3) = traindata_nl(:,1);
plotClassData('nonlinear',modelData_nl,traindata_correctformat_nl,2);



