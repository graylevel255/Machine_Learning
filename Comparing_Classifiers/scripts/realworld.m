% This code gives output for the best model.

%Divide the data by classes
A = importdata('./datasets/group23/real_world_static.txt');
class_label_col_no = 9;
n = size(A,1);
var = size(A,2)-1;
k = 8;
nc = 3;
train_classes_sizes = zeros(3,1);
class1_data = A(A(:,class_label_col_no)==1, :);
class2_data = A(A(:,class_label_col_no)==2, :);
class3_data = A(A(:,class_label_col_no)==3, :);


%Divide into train, test, val
[class1_train, class1_test, class1_val] = divide_data(class1_data);
[class2_train, class2_test, class2_val] = divide_data(class2_data);
[class3_train, class3_test, class3_val] = divide_data(class3_data);


train_classes_sizes(1) = size(class1_train,1);
train_classes_sizes(2) = size(class2_train,1);
train_classes_sizes(3) = size(class3_train,1);


trainData = [class1_train;class2_train;class3_train];

k
disp('TRAIN');
class1_train = r_performNonParamKNN(class1_train,trainData,train_classes_sizes,nc,var,k);
class2_train = r_performNonParamKNN(class2_train,trainData,train_classes_sizes,nc,var,k);
class3_train = r_performNonParamKNN(class3_train,trainData,train_classes_sizes,nc,var,k);
trainDataa = [class1_train;class2_train;class3_train;];
[CM,accuracy] = confusionMatrix(trainDataa,nc,var,k,'nonparamknn');
accuracy
CM

disp('TEST');
class1_test = r_performNonParamKNN(class1_test,trainData,train_classes_sizes,nc,var,k);
class2_test = r_performNonParamKNN(class2_test,trainData,train_classes_sizes,nc,var,k);
class3_test = r_performNonParamKNN(class3_test,trainData,train_classes_sizes,nc,var,k);

testData = [class1_test;class2_test;class3_test];
[CM,accuracy] = confusionMatrix(testData,nc,var,k,'nonparamknn');
accuracy
CM

disp('VALIDATION');
class1_val = r_performNonParamKNN(class1_val,trainData,train_classes_sizes,nc,var,k);
class2_val = r_performNonParamKNN(class2_val,trainData,train_classes_sizes,nc,var,k);
class3_val = r_performNonParamKNN(class3_val,trainData,train_classes_sizes,nc,var,k);

valData = [class1_val;class2_val;class3_val];
[CM,accuracy] = confusionMatrix(valData,nc,var,k,'nonparamknn');
accuracy











