function [output,numberOfClasses] = readCategoryFile(category,tvt)

if(strcmp(category,'linear'))
    baseLoc = '../data/linearly_separable/';
elseif(strcmp(category,'nonlinear'))
    baseLoc = '../data/nonlinearly_separable/';
end
if(strcmp(category,'linear'))
    numberOfClasses=4;
if(strcmp(tvt,'train'))

    loc1 = 'class1_train.txt';
    loc2 = 'class2_train.txt';
    loc3 = 'class3_train.txt';
    loc4 = 'class4_train.txt';
elseif(strcmp(tvt,'test'))
    loc1 = 'class1_test.txt';
    loc2 = 'class2_test.txt';
    loc3 = 'class3_test.txt';
    loc4 = 'class4_test.txt';
elseif(strcmp(tvt,'val'))
    loc1 = 'class1_val.txt';
    loc2 = 'class2_val.txt';
    loc3 = 'class3_val.txt';
    loc4 = 'class4_val.txt';
end
    c1 = importdata(strcat(baseLoc,loc1),' ');
    c2 = importdata(strcat(baseLoc,loc2),' ');
    c3 = importdata(strcat(baseLoc,loc3),' ');
    c4 = importdata(strcat(baseLoc,loc4),' ');
    sizeOfClasses = size(c1,1);
    classnum = zeros(sizeOfClasses,1);
    c1 = [classnum+1,c1];
    c2 = [classnum+2,c2];
    c3 = [classnum+3,c3];
    c4 = [classnum+4,c4];
    output = [c1;c2;c3;c4];
elseif(strcmp(category,'nonlinear'))
    numberOfClasses=2;
    if(strcmp(tvt,'train'))
        loc1 = 'class1_train.txt';
        loc2 = 'class2_train.txt';
    elseif(strcmp(tvt,'test'))
        loc1 = 'class1_test.txt';
        loc2 = 'class2_test.txt';
    elseif(strcmp(tvt,'val'))
        loc1 = 'class1_val.txt';
        loc2 = 'class2_val.txt';
    end
    c1 = importdata(strcat(baseLoc,loc1),' ');
    c2 = importdata(strcat(baseLoc,loc2),' ');
    sizeOfClasses = size(c1,1);
    classnum = zeros(sizeOfClasses,1);
    c1 = [c1,classnum+1];
    c2 = [c2,classnum+2];
    output = [c1;c2];
end
end