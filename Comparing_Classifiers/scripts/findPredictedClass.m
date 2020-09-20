function predictData = findPredictedClass(numOfClasses,predictData,var,k)
class1 = var+2;
classk = var+k+1;
count = zeros(numOfClasses,1);
n = size(predictData,1);
initializePredictedClass = zeros(n,1);
predictData = [predictData,initializePredictedClass];
predictedClassColumn = classk+1;
for i=1:n
    for j=class1:classk
        for p=1:numOfClasses
            if(predictData(i,j) == p)
                count(p) = count(p)+1;
                break;
            end
        end
    end
    predictedClass = findMaxClassCount(count,numOfClasses);
    predictData(i,predictedClassColumn) = predictedClass;
    for p=1:numOfClasses
        count(p) = 0;
    end
end 
end