%This function generated 4 files in the correct format
%train,trainmesh,test,val
function dataformat(category)
baseLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\';
if(strcmp(category,'linear'))
    linLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\datasets\linearly_separable\'; 
    nc = 4;
elseif(strcmp(category,'nonlinear'))
    nonlinLoc = 'C:\Users\Aakriti\Desktop\3assgnPRML\datasets\nonlinearly_separable\'; 
    nc = 2;
end

if(strcmp(category,'linear'))
%TRAIN
trainfileid = fopen(strcat(baseLoc,'linseptrain'),'w');
loc1 = 'class1_train.txt';
loc2 = 'class2_train.txt';
loc3 = 'class3_train.txt';
loc4 = 'class4_train.txt';
c1 = importdata(strcat(linLoc,loc1),' ');
c2 = importdata(strcat(linLoc,loc2),' ');
c3 = importdata(strcat(linLoc,loc3),' ');
c4 = importdata(strcat(linLoc,loc4),' ');
c = [c1;c2;c3;c4];
ntrain = size(c1,1);
for i=1:nc
for j=1:ntrain
   k = (i-1)*ntrain+j;
   fprintf(trainfileid,'%d ',i);
   fprintf(trainfileid,'%d:%f',1,c(k,1));
   fprintf(trainfileid,' %d:%f',2,c(k,2));
   fprintf(trainfileid,'\n');
end
end
 
%TRAIN MESH
% min x1 = -14.1950 ; max x1 = 15.9270;
% min x2 = -11.9080 ; max x2 = 19.4320;
[ax1,ax2] = meshgrid(-15:0.1:20);
ax1 = reshape(ax1,351*351,1);
ax2 = reshape(ax2,351*351,1);
linseptrainmesh = [ax1,ax2];
ntm = size(linseptrainmesh,1);
trainmeshfileid = fopen(strcat(baseLoc,'linseptrainmesh'),'w');
tmfileid = fopen(strcat(baseLoc,'linsepmesh2'),'w');
for i=1:ntm
   fprintf(trainmeshfileid,'%d ',0);
   fprintf(trainmeshfileid,'%d:%f',1,linseptrainmesh(i,1));
   fprintf(trainmeshfileid,' %d:%f',2,linseptrainmesh(i,2));
   fprintf(trainmeshfileid,'\n');
   
   fprintf(tmfileid,'%f ',linseptrainmesh(i,1));
   fprintf(tmfileid,'%f',linseptrainmesh(i,2));
   fprintf(tmfileid,'\n');
end

%TEST
testfileid = fopen(strcat(baseLoc,'linseptest'),'w');
testloc1 = 'class1_test.txt';
testloc2 = 'class2_test.txt';
testloc3 = 'class3_test.txt';
testloc4 = 'class4_test.txt';
c1test = importdata(strcat(linLoc,testloc1),' ');
c2test = importdata(strcat(linLoc,testloc2),' ');
c3test = importdata(strcat(linLoc,testloc3),' ');
c4test = importdata(strcat(linLoc,testloc4),' ');
ctest = [c1test;c2test;c3test;c4test];
ntest = size(c1test,1);
for i=1:nc
for j=1:ntest
    k = (i-1)*ntest+j;
   fprintf(testfileid,'%d ',i);
   fprintf(testfileid,'%d:%f',1,ctest(k,1));
   fprintf(testfileid,' %d:%f',2,ctest(k,2));
   fprintf(testfileid,'\n');
end
end

%VAL
valfileid = fopen(strcat(baseLoc,'linsepval'),'w');
loc1 = 'class1_val.txt';
loc2 = 'class2_val.txt';
loc3 = 'class3_val.txt';
loc4 = 'class4_val.txt';
c1 = importdata(strcat(linLoc,loc1),' ');
c2 = importdata(strcat(linLoc,loc2),' ');
c3 = importdata(strcat(linLoc,loc3),' ');
c4 = importdata(strcat(linLoc,loc4),' ');
c = [c1;c2;c3;c4];
nval = size(c1,1);
for i=1:nc
for j=1:nval
    k = (i-1)*nval+j;
   fprintf(valfileid,'%d ',i);
   fprintf(valfileid,'%d:%f',1,c(k,1));
   fprintf(valfileid,' %d:%f',2,c(k,2));
   fprintf(valfileid,'\n');
end
end

% CHECK ALL 4 FILE FORMATS : train,trainmodel,test,val
%train
commandStr = 'C:\Users\Aakriti\Desktop\3assgnPRML\libsvm-3.23\libsvm-3.23\tools\checkdata.py C:\Users\Aakriti\Desktop\3assgnPRML\linseptrain';
 [status, commandOut] = system(commandStr);
 if status==0
     fprintf('Train : Format is correct\n');
 else
     fprintf('ERRORS : ');
     fprintf(commandOut)
 end
%trainmesh
commandStr = 'C:\Users\Aakriti\Desktop\3assgnPRML\libsvm-3.23\libsvm-3.23\tools\checkdata.py C:\Users\Aakriti\Desktop\3assgnPRML\linseptrainmesh';
 [status, commandOut] = system(commandStr);
 if status==0
     fprintf('Train mesh : Format is correct\n');
 else
     fprintf('ERRORS : ');
     fprintf(commandOut)
 end
%test
commandStr = 'C:\Users\Aakriti\Desktop\3assgnPRML\libsvm-3.23\libsvm-3.23\tools\checkdata.py C:\Users\Aakriti\Desktop\3assgnPRML\linseptest';
 [status, commandOut] = system(commandStr);
 if status==0
     fprintf('Test : Format is correct\n');
 else
     fprintf('ERRORS : ');
     fprintf(commandOut)
 end
%val
 commandStr = 'C:\Users\Aakriti\Desktop\3assgnPRML\libsvm-3.23\libsvm-3.23\tools\checkdata.py C:\Users\Aakriti\Desktop\3assgnPRML\linsepval';
 [status, commandOut] = system(commandStr);
 if status==0
     fprintf('Val : Format is correct\n');
 else
     fprintf('ERRORS : ');
     fprintf(commandOut)
 end
 
 
elseif(strcmp(category,'nonlinear'))
%TRAIN
trainfileid = fopen(strcat(baseLoc,'nonlinseptrain'),'w');
loc1 = 'class1_train.txt';
loc2 = 'class2_train.txt';
c1 = importdata(strcat(nonlinLoc,loc1),' ');
c2 = importdata(strcat(nonlinLoc,loc2),' ');
c = [c1;c2];
ntrain = size(c1,1);
for i=1:nc
for j=1:ntrain
   k = (i-1)*ntrain+j;
   fprintf(trainfileid,'%d ',i);
   fprintf(trainfileid,'%d:%f',1,c(k,1));
   fprintf(trainfileid,' %d:%f',2,c(k,2));
   fprintf(trainfileid,'\n');
end
end
 
%TRAIN MESH
% min x1 = -13.2170 ; max x1 = 14.5240;
% min x2 = -11.9230 ; max x2 = 12.9640;
[ax1,ax2] = meshgrid(-15:0.1:15);
ax1 = reshape(ax1,301*301,1);
ax2 = reshape(ax2,301*301,1);
nonlinseptrainmesh = [ax1,ax2];
ntm = size(nonlinseptrainmesh,1);
trainmeshfileid = fopen(strcat(baseLoc,'nonlinseptrainmesh'),'w');
tmfileid = fopen(strcat(baseLoc,'nonlinsepmesh2'),'w');
for i=1:ntm
   fprintf(trainmeshfileid,'%d ',0);
   fprintf(trainmeshfileid,'%d:%f',1,nonlinseptrainmesh(i,1));
   fprintf(trainmeshfileid,' %d:%f',2,nonlinseptrainmesh(i,2));
   fprintf(trainmeshfileid,'\n');
   
   fprintf(tmfileid,'%f ',nonlinseptrainmesh(i,1));
   fprintf(tmfileid,'%f',nonlinseptrainmesh(i,2));
   fprintf(tmfileid,'\n');
end

%TEST
testfileid = fopen(strcat(baseLoc,'nonlinseptest'),'w');
testloc1 = 'class1_test.txt';
testloc2 = 'class2_test.txt';
c1test = importdata(strcat(nonlinLoc,testloc1),' ');
c2test = importdata(strcat(nonlinLoc,testloc2),' ');
ctest = [c1test;c2test];
ntest = size(c1test,1);
for i=1:nc
for j=1:ntest
    k = (i-1)*ntest+j;
   fprintf(testfileid,'%d ',i);
   fprintf(testfileid,'%d:%f',1,ctest(k,1));
   fprintf(testfileid,' %d:%f',2,ctest(k,2));
   fprintf(testfileid,'\n');
end
end

%VAL
valfileid = fopen(strcat(baseLoc,'nonlinsepval'),'w');
loc1 = 'class1_val.txt';
loc2 = 'class2_val.txt';
c1 = importdata(strcat(nonlinLoc,loc1),' ');
c2 = importdata(strcat(nonlinLoc,loc2),' ');
c = [c1;c2];
nval = size(c1,1);
for i=1:nc
for j=1:nval
    k = (i-1)*nval+j;
   fprintf(valfileid,'%d ',i);
   fprintf(valfileid,'%d:%f',1,c(k,1));
   fprintf(valfileid,' %d:%f',2,c(k,2));
   fprintf(valfileid,'\n');
end
end

% CHECK ALL 4 FILE FORMATS : train,trainmodel,test,val
%train
commandStr = 'C:\Users\Aakriti\Desktop\3assgnPRML\libsvm-3.23\libsvm-3.23\tools\checkdata.py C:\Users\Aakriti\Desktop\3assgnPRML\nonlinseptrain';
 [status, commandOut] = system(commandStr);
 if status==0
     fprintf('Train : Format is correct\n');
 else
     fprintf('ERRORS : ');
     fprintf(commandOut)
 end
%trainmesh
commandStr = 'C:\Users\Aakriti\Desktop\3assgnPRML\libsvm-3.23\libsvm-3.23\tools\checkdata.py C:\Users\Aakriti\Desktop\3assgnPRML\nonlinseptrainmesh';
 [status, commandOut] = system(commandStr);
 if status==0
     fprintf('Train mesh : Format is correct\n');
 else
     fprintf('ERRORS : ');
     fprintf(commandOut)
 end
%test
commandStr = 'C:\Users\Aakriti\Desktop\3assgnPRML\libsvm-3.23\libsvm-3.23\tools\checkdata.py C:\Users\Aakriti\Desktop\3assgnPRML\nonlinseptest';
 [status, commandOut] = system(commandStr);
 if status==0
     fprintf('Test : Format is correct\n');
 else
     fprintf('ERRORS : ');
     fprintf(commandOut)
 end
%val
 commandStr = 'C:\Users\Aakriti\Desktop\3assgnPRML\libsvm-3.23\libsvm-3.23\tools\checkdata.py C:\Users\Aakriti\Desktop\3assgnPRML\nonlinsepval';
 [status, commandOut] = system(commandStr);
 if status==0
     fprintf('Val : Format is correct\n');
 else
     fprintf('ERRORS : ');
     fprintf(commandOut)
 end
end