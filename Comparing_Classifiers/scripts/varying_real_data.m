% Class 1  - 008.bathtub   (N = 32226)
% Class 2 - 109.hot-tub (55595)
% Class 3 - 138.mattress (32853)
% Class 4 - 232.t-shirt (76075)
% Class 5 - 129.leopards-101 (10831)

nc = 5;
var = 64;
k = 1;
train_classes_sizes = zeros(5,1);
test_classes_sizes = zeros(5,1);
val_classes_sizes = zeros(5,1);

parent_path = './datasets/group23/feats_surf';
file1_loc = './datasets/group23/feats_surf/008.bathtub';
file2_loc = './datasets/group23/feats_surf/109.hot-tub';
file3_loc = './datasets/group23/feats_surf/138.mattress';
file4_loc = './datasets/group23/feats_surf/232.t-shirt';
file5_loc = './datasets/group23/feats_surf/129.leopards-101';

D = dir(fullfile(file1_loc, '*.txt'));
A = cell(size(D));
class1_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file1_loc,D(i).name)); 
      class1_data = [class1_data ; A{i}];
end

D = dir(fullfile(file2_loc, '*.txt'));
A = cell(size(D));
class2_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file2_loc,D(i).name)); 
      class2_data = [class2_data ; A{i}];
end

D = dir(fullfile(file3_loc, '*.txt'));
A = cell(size(D));
class3_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file3_loc,D(i).name)); 
      class3_data = [class3_data ; A{i}];
end

D = dir(fullfile(file4_loc, '*.txt'));
A = cell(size(D));
class4_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file4_loc,D(i).name)); 
      class4_data = [class4_data ; A{i}];
end

D = dir(fullfile(file5_loc, '*.txt'));
A = cell(size(D));
class5_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file5_loc,D(i).name)); 
      class5_data = [class5_data ; A{i}];
end


[class1_train, class1_test, class1_val] = divide_data(class1_data);
[class2_train, class2_test, class2_val] = divide_data(class2_data);
[class3_train, class3_test, class3_val] = divide_data(class3_data);
[class4_train, class4_test, class4_val] = divide_data(class4_data);
[class5_train, class5_test, class5_val] = divide_data(class5_data);

train_classes_sizes(1) = size(class1_train,1);
train_classes_sizes(2) = size(class2_train,1);
train_classes_sizes(3) = size(class3_train,1);
train_classes_sizes(4) = size(class4_train,1);
train_classes_sizes(5) = size(class5_train,1);

test_classes_sizes(1) = size(class1_test,1);
test_classes_sizes(2) = size(class2_test,1);
test_classes_sizes(3) = size(class3_test,1);
test_classes_sizes(4) = size(class4_test,1);
test_classes_sizes(5) = size(class5_test,1);

val_classes_sizes(1) = size(class1_val,1);
val_classes_sizes(2) = size(class2_val,1);
val_classes_sizes(3) = size(class3_val,1);
val_classes_sizes(4) = size(class4_val,1);
val_classes_sizes(5) = size(class5_val,1);


train1vector = zeros(train_classes_sizes(1),1)+1;test1vector = zeros(test_classes_sizes(1),1)+1;val1vector = zeros(val_classes_sizes(1),1)+1;
train2vector = zeros(train_classes_sizes(2),1)+2;test2vector = zeros(test_classes_sizes(2),1)+2;val2vector = zeros(val_classes_sizes(2),1)+2;
train3vector = zeros(train_classes_sizes(3),1)+3;test3vector = zeros(test_classes_sizes(3),1)+3;val3vector = zeros(val_classes_sizes(3),1)+3;
train4vector = zeros(train_classes_sizes(4),1)+4;test4vector = zeros(test_classes_sizes(4),1)+4;val4vector = zeros(val_classes_sizes(4),1)+4;
train5vector = zeros(train_classes_sizes(5),1)+5;test5vector = zeros(test_classes_sizes(5),1)+5;val5vector = zeros(val_classes_sizes(5),1)+5;

class1_train=[class1_train,train1vector]; class1_test=[class1_test,test1vector]; class1_val=[class1_val,val1vector];
class2_train=[class2_train,train2vector]; class2_test=[class2_test,test2vector]; class2_val=[class2_val,val2vector];
class3_train=[class3_train,train3vector]; class3_test=[class3_test,test3vector]; class3_val=[class3_val,val3vector];
class4_train=[class4_train,train4vector]; class4_test=[class4_test,test4vector]; class4_val=[class4_val,val4vector];
class5_train=[class5_train,train5vector]; class5_test=[class5_test,test5vector]; class5_val=[class5_val,val5vector];

trainData = [class1_train;class2_train;class3_train;class4_train;class5_train];

% for k=2:5
class1_train = r_performNonParamKNN(class1_train,trainData,train_classes_sizes,nc,var,k);
class2_train = r_performNonParamKNN(class2_train,trainData,train_classes_sizes,nc,var,k);
class3_train= r_performNonParamKNN(class3_train,trainData,train_classes_sizes,nc,var,k);
class4_train = r_performNonParamKNN(class4_train,trainData,train_classes_sizes,nc,var,k);
class5_train = r_performNonParamKNN(class5_train,trainData,train_classes_sizes,nc,var,k);

trainData = [class1_test;class2_test;class3_test;class4_test;class5_test];
[CM,accuracy] = confusionMatrix(testData,nc,var,k,'nonparamknn');
k
accuracy
CM
% end













