%%%% NON-LINEAR (GAUSSIAN - RADIAL BASIS FUNCTION )KERNEL %%%%
function gaussiankernel(category)
baseLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\';
linLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\datasets\linearly_separable\';
nonlinLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\datasets\nonlinearly_separable\';
if(strcmp(category,'linear'))
% Train
command_gauss_train = 'svm-train -s 0 -t 2 -c 12 -g 0.005 linseptrain';
system(command_gauss_train);

% Predict train
command_gausstrain_predict = 'svm-predict linseptrain linseptrain.model linsep_gauss_trainoutput';
system(command_gausstrain_predict);

% Predict train mesh
command_gausstrainmesh_predict = 'svm-predict linseptrainmesh linseptrain.model linsep_gauss_trainmeshoutput';
system(command_gausstrainmesh_predict);
gausslabels = importdata(strcat(baseLoc,'linsep_gauss_trainmeshoutput'));
gausscoords = importdata(strcat(baseLoc,'linsepmesh2'));
np = size(gausscoords,1);
gaussmodelData = zeros(np,3);
for i=1:np
    gaussmodelData(i,1) = gausscoords(i,1);
    gaussmodelData(i,2) = gausscoords(i,2);
    gaussmodelData(i,3) = gausslabels(i);
end

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

plotSVMData('linear',gaussmodelData,data,2,'gaussian');

% Predict test
command_gausstest_predict = 'svm-predict linseptest linseptrain.model linsep_gauss_testoutput';
system(command_gausstest_predict);

% Predict val
command_gaussval_predict = 'svm-predict linsepval linseptrain.model linsep_gauss_valoutput';
system(command_gaussval_predict);

elseif(strcmp(category,'nonlinear'))
% Train
command_gauss_train = 'svm-train -s 0 -t 2 -c 10 -g 0.1 nonlinseptrain';
system(command_gauss_train);

% Predict train
command_gausstrain_predict = 'svm-predict nonlinseptrain nonlinseptrain.model nonlinsep_gauss_trainoutput';
system(command_gausstrain_predict);

% Predict train mesh
command_gausstrainmesh_predict = 'svm-predict nonlinseptrainmesh nonlinseptrain.model nonlinsep_gauss_trainmeshoutput';
system(command_gausstrainmesh_predict);
gausslabels = importdata(strcat(baseLoc,'nonlinsep_gauss_trainmeshoutput'));
gausscoords = importdata(strcat(baseLoc,'nonlinsepmesh2'));
np = size(gausscoords,1);
gaussmodelData = zeros(np,3);
for i=1:np
    gaussmodelData(i,1) = gausscoords(i,1);
    gaussmodelData(i,2) = gausscoords(i,2);
    gaussmodelData(i,3) = gausslabels(i);
end

loc1 = 'class1_train.txt';
loc2 = 'class2_train.txt';
c1 = importdata(strcat(nonlinLoc,loc1),' ');
c2 = importdata(strcat(nonlinLoc,loc2),' ');
ntrain = size(c1,1);
origclasslabel = zeros(ntrain,1);
c1 = [c1,origclasslabel+1];
c2 = [c2,origclasslabel+2];
data = [c1;c2];

plotSVMData('nonlinear',gaussmodelData,data,2,'gaussian');

% Predict test
command_gausstest_predict = 'svm-predict nonlinseptest nonlinseptrain.model nonlinsep_gauss_testoutput';
system(command_gausstest_predict);

% Predict val
command_gaussval_predict = 'svm-predict nonlinsepval nonlinseptrain.model nonlinsep_gauss_valoutput';
system(command_gaussval_predict);   

elseif(strcmp(category,'image'))
% Train
command_gauss_train = 'svm-train -s 0 -t 2 -c 5 -g 0.005 imagetrain';
system(command_gauss_train);

% Predict train
command_gausstrain_predict = 'svm-predict imagetrain imagetrain.model image_gauss_trainoutput';
system(command_gausstrain_predict);

% Predict test
command_gausstest_predict = 'svm-predict imagetest imagetrain.model image_gauss_testoutput';
system(command_gausstest_predict);

% Predict val
command_gaussval_predict = 'svm-predict imageval imagetrain.model image_gauss_valoutput';
system(command_gaussval_predict);   
end