% This file gives outputs for the best models for each of the 3 categories 
% of data.
var = 2;

disp('LINEARLY SEPARABLE DATA');
k=1;
%Read training data
[data,nc] = readCategoryFile('linear','train');

%Plot model
% min x1 = -14.1950 ; max x1 = 15.9270;
% min x2 = -11.9080 ; max x2 = 19.4320;
[ax1,ax2] = meshgrid(-15:0.1:20);
ax1 = reshape(ax1,351*351,1);
ax2 = reshape(ax2,351*351,1);
modelData = [ax1,ax2,zeros(123201,1)];
modelData = NP_findDistance(modelData,data,4,var,k);
modelData = NP_find_min_Rmax(modelData,var,4);
plotClassData('linear',modelData,data,var,k,4,'nonparamknn');

disp('TRAIN');
[trainData,CM,accuracy] = performNonParamKNN('linear','train',data,var,k);
accuracy
CM

disp('TEST');
[trainData,CM,accuracy] = performNonParamKNN('linear','test',data,var,k);
accuracy
CM

disp('VAL')
[trainData,CM,accuracy] = performNonParamKNN('linear','val',data,var,k);
accuracy
CM

disp('NON - LINEARLY SEPARABLE DATA');
k=1;
%Read training data
[data,nc] = readCategoryFile('nonlinear','train');

%Plot model
% min x1 = -13.2170 ; max x1 = 14.5240;
% min x2 = -11.9230 ; max x2 = 12.9640;

[ax1,ax2] = meshgrid(-15:0.1:15);
ax1 = reshape(ax1,301*301,1);
ax2 = reshape(ax2,301*301,1);
modelData = [ax1,ax2,zeros(90601,1)];
modelData = NP_findDistance(modelData,data,2,var,k);
modelData = NP_find_min_Rmax(modelData,var,2);
plotClassData('nonlinear',modelData,data,var,k,2,'nonparamknn');

disp('TRAIN');
[trainData,CM,accuracy] = performNonParamKNN('nonlinear','train',data,var,k);
accuracy
CM

disp('TEST');
[testData,CM,accuracy] = performNonParamKNN('nonlinear','test',data,var,k);
accuracy
CM

disp('VAL');
[valData,CM,accuracy] = performNonParamKNN('nonlinear','val',data,var,k);
accuracy
CM

disp('OVERLAPPING DATA');
k=3;
%Read training data
[data,nc] = readCategoryFile('overlap','train');

%Plot model
% min x1 = -8.5294 ; max x1 = 10.8380;
% min x2 = -10.0460 ; max x2 = 13.1940;

[ax1,ax2] = meshgrid(-15:0.1:20);
ax1 = reshape(ax1,351*351,1);
ax2 = reshape(ax2,351*351,1);
modelData = [ax1,ax2,zeros(123201,1)];
modelData = NP_findDistance(modelData,data,4,var,k);
modelData = NP_find_min_Rmax(modelData,var,4);
plotClassData('overlap',modelData,data,var,k,4,'nonparamknn');

disp('TRAIN');
[trainData,CM,accuracy] = performNonParamKNN('overlap','train',data,var,k);
accuracy
CM

disp('TEST');
[testData,CM,accuracy] = performNonParamKNN('overlap','test',data,var,k);
accuracy
CM

disp('VAL');
[valData,CM,accuracy] = performNonParamKNN('overlap','val',data,var,k);
accuracy
CM

