function [CM,accuracy] = confusionMatrix(predictData,numOfClasses,var,k,type)
% var+1 gives the original class column index
% var+k+2 gives the predicted class column index for simple KNN
% var+numOfClasses+3 gives the predicted class column index for Non
% Parametric KNN
CM = zeros(numOfClasses,numOfClasses);
n = size(predictData,1);

origClassColumn = var+1;
predictedClassColumn = var+k+2;
if(strcmp(type,'nonparamknn'))
    predictedClassColumn = var+numOfClasses+3;
end

for i=1:n
    origClass = predictData(i,origClassColumn);
    predictedClass = predictData(i,predictedClassColumn);
    CM(origClass,predictedClass) = CM(origClass,predictedClass)+1;
end

sum = 0;
for i=1:numOfClasses
    for predictedClass=1:numOfClasses
        sum = sum + CM(i,predictedClass);
    end
    for predictedClass=1:numOfClasses
        CM(i,predictedClass) = 100*CM(i,predictedClass)/sum;
    end
    sum = 0;
end

count=0;
for i = 1:n
   if predictData(i,origClassColumn)==predictData(i,predictedClassColumn)
       count = count+1;
   end
end
accuracy = 100*count/n;
end