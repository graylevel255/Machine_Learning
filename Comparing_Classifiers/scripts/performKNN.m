function [predictData,CM,accuracy] = performKNN(category,tvt,data,var,k)
[predictData,numOfClasses] = readCategoryFile(category,tvt);
predictData = findDistance(predictData,data,k);
predictData = findPredictedClass(numOfClasses,predictData,var,k);
[CM,accuracy] = confusionMatrix(predictData,numOfClasses,var,k,'KNN');
end