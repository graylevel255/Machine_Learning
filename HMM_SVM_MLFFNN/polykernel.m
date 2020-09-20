%%%%%%%%%%%  NON-LINEAR(POLYNOMIAL)KERNEL  %%%%%%%%%%%
function polykernel(category)
baseLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\';
linLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\datasets\linearly_separable\';
nonlinLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\datasets\nonlinearly_separable\';
if(strcmp(category,'linear'))
% Train
command_poly_train = 'svm-train -s 0 -t 1 -c 1 -g 1 -r 1 -d 3 linseptrain';
system(command_poly_train);

% Predict train
command_polytrain_predict = 'svm-predict linseptrain linseptrain.model linsep_poly_trainoutput';
system(command_polytrain_predict);

% Predict train mesh
command_polytrainmesh_predict = 'svm-predict linseptrainmesh linseptrain.model linsep_poly_trainmeshoutput';
system(command_polytrainmesh_predict);

polylabels = importdata(strcat(baseLoc,'linsep_poly_trainmeshoutput'));
polycoords = importdata(strcat(baseLoc,'linsepmesh2'));
np = size(polycoords,1);
polymodelData = zeros(np,3);
for i=1:np
    polymodelData(i,1) = polycoords(i,1);
    polymodelData(i,2) = polycoords(i,2);
    polymodelData(i,3) = polylabels(i);
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

plotSVMData('linear',polymodelData,data,2,'poly');

% Predict test
command_polytest_predict = 'svm-predict linseptest linseptrain.model linsep_poly_testoutput';
system(command_polytest_predict);

% Predict val
command_polyval_predict = 'svm-predict linsepval linseptrain.model linsep_poly_valoutput';
system(command_polyval_predict);


elseif(strcmp(category,'nonlinear'))
% Train
command_poly_train = 'svm-train -s 0 -t 1 -c 1 -g 1 -r 1 -d 2 -h 0 nonlinseptrain';
system(command_poly_train);

% Predict train
command_polytrain_predict = 'svm-predict nonlinseptrain nonlinseptrain.model nonlinsep_poly_trainoutput';
system(command_polytrain_predict);

% Predict train mesh
command_polytrainmesh_predict = 'svm-predict nonlinseptrainmesh nonlinseptrain.model nonlinsep_poly_trainmeshoutput';
system(command_polytrainmesh_predict);

polylabels = importdata(strcat(baseLoc,'nonlinsep_poly_trainmeshoutput'));
polycoords = importdata(strcat(baseLoc,'nonlinsepmesh2'));
np = size(polycoords,1);
polymodelData = zeros(np,3);
for i=1:np
    polymodelData(i,1) = polycoords(i,1);
    polymodelData(i,2) = polycoords(i,2);
    polymodelData(i,3) = polylabels(i);
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

plotSVMData('nonlinear',polymodelData,data,2,'poly');

% Predict test
command_polytest_predict = 'svm-predict nonlinseptest nonlinseptrain.model nonlinsep_poly_testoutput';
system(command_polytest_predict);

% Predict val
command_polyval_predict = 'svm-predict nonlinsepval nonlinseptrain.model nonlinsep_poly_valoutput';
system(command_polyval_predict);

elseif(strcmp(category,'image'))
% Train
command_poly_train = 'svm-train -s 0 -t 1 -c 1 -g 10 -r 2 -d 3 imagetrain';
system(command_poly_train);

% Predict train
command_polytrain_predict = 'svm-predict imagetrain imagetrain.model image_poly_trainoutput';
system(command_polytrain_predict);

% Predict test
command_polytest_predict = 'svm-predict imagetest imagetrain.model image_poly_testoutput';
system(command_polytest_predict);

% Predict val
command_polyval_predict = 'svm-predict imageval imagetrain.model image_poly_valoutput';
system(command_polyval_predict);
end