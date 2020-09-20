function readimagedata()
% Read data of 3 classes

% Class 1 - highway (N = 260)
% Class 2 - mountain (N = 374)
% Class 3 - tall building (N = 356)
nc = 3;
var = 23;

file1_loc = './datasets/image_dataset/Features/highway';
file2_loc = './datasets/image_dataset/Features/mountain';
file3_loc = './datasets/image_dataset/Features/tallbuilding';

D = dir(fullfile(file1_loc, '*.jpg_color_edh_entropy'));
A = cell(size(D));
class1_data = zeros(260,828);
for i=1:length(D)
      A{i} = importdata(fullfile(file1_loc,D(i).name));
      R = reshape(A{i},1,[]);
      for j=1:828
          class1_data(i,j) = R(j);
      end
end

D = dir(fullfile(file2_loc, '*.jpg_color_edh_entropy'));
A = cell(size(D));
class2_data = zeros(374,828);
for i=1:length(D)
      A{i} = importdata(fullfile(file2_loc,D(i).name)); 
      R = reshape(A{i},1,[]);
      for j=1:828
          class2_data(i,j) = R(j);
      end
end

D = dir(fullfile(file3_loc, '*.jpg_color_edh_entropy'));
A = cell(size(D));
class3_data = zeros(356,828);
for i=1:length(D)
      A{i} = importdata(fullfile(file3_loc,D(i).name));
      R = reshape(A{i},1,[]);
      for j=1:828
          class3_data(i,j) = R(j);
      end
end
n1 = 260;
n2 = 374;
n3 = 356;
%class1 = 260 ; train = 182 ; test = 26 ; val = 52
%class2 = 374 ; train = 262 ; test = 37 ; val = 75
%class3 = 356 ; train = 250 ; test = 35 ; val = 71
class1_data = [zeros(n1,1)+1,class1_data];
class2_data = [zeros(n2,1)+2,class2_data];
class3_data = [zeros(n3,1)+3,class3_data];
train1 = class1_data(1:182,:);
test1 = class1_data(183:208,:);
val1 = class1_data(209:260,:);
train2 = class2_data(1:262,:);
test2 = class2_data(263:299,:);
val2 = class2_data(300:374,:);
train3 = class3_data(1:250,:);
test3 = class3_data(251:285,:);
val3 = class3_data(286:356,:);

train = [train1;train2;train3];
test = [test1;test2;test3];
val = [val1;val2;val3];

dim = 828;
ntrain = size(train,1);
trainfileid = fopen('./imagetrain','w');
for i=1:ntrain
   fprintf(trainfileid,'%d ',train(i,1));
   for j=1:dim
   fprintf(trainfileid,'%d:%f ',j,train(i,j+1));
   end
   fprintf(trainfileid,'\n');
end

ntest = size(test,1);
testfileid = fopen('./imagetest','w');
for i=1:ntest
   fprintf(testfileid,'%d ',test(i,1));
   for j=1:dim
   fprintf(testfileid,'%d:%f ',j,test(i,j+1));
   end
   fprintf(testfileid,'\n');
end

nval = size(val,1);
valfileid = fopen('./imageval','w');
for i=1:nval
   fprintf(valfileid,'%d ',val(i,1));
   for j=1:dim
   fprintf(valfileid,'%d:%f ',j,val(i,j+1));
   end
   fprintf(valfileid,'\n');
end

end