%%%%%%%%%%%%%%  LINEAR KERNEL  %%%%%%%%%%%%%%%%%%
function linearkernel()
baseLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\';
% Train
command_train = 'svm-train -s 0 -t 0 -c 10 linseptrain';
model = system(command_train);

% Predict train
command_predict_train = 'svm-predict linseptrain linseptrain.model linseptrainoutput';
train_result = system(command_predict_train);

% Predict train mesh
command_predict_trainmesh = 'svm-predict linseptrainmesh linseptrain.model linseptrainmeshoutput';
trainmesh_result = system(command_predict_trainmesh);
labels = importdata(strcat(baseLoc,'linseptrainmeshoutput'));
coords = importdata(strcat(baseLoc,'linsepmesh2'));
ntl = size(coords,1);
modelData = zeros(ntl,3);
for i=1:ntl
    modelData(i,1) = coords(i,1);
    modelData(i,2) = coords(i,2);
    modelData(i,3) = labels(i);
end

linLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\datasets\linearly_separable\';
loc1 = 'class1_train.txt';
loc2 = 'class2_train.txt';
loc3 = 'class3_train.txt';
loc4 = 'class4_train.txt';
c1 = importdata(strcat(linLoc,loc1),' ');
c2 = importdata(strcat(linLoc,loc2),' ');
c3 = importdata(strcat(linLoc,loc3),' ');
c4 = importdata(strcat(linLoc,loc4),' ');
ntrain = size(c1,1);
origclasslabel = zeros(ntrain,1);
c1 = [c1,origclasslabel+1];
c2 = [c2,origclasslabel+2];
c3 = [c3,origclasslabel+3];
c4 = [c4,origclasslabel+4];
data = [c1;c2;c3;c4];
plotSVMData('linear',modelData,data,2,'linear');

% Predict test
command_predict_test = 'svm-predict linseptest linseptrain.model linseptestoutput';
test_result = system(command_predict_test);

% Predict val
command_predict_val = 'svm-predict linsepval linseptrain.model linsepvaloutput';
val_result = system(command_predict_val);
end