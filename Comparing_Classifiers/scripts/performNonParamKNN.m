function [predictData,CM,accuracy] = performNonParamKNN(category,tvt,data,var,k)
[predictData,numOfClasses] = readCategoryFile(category,tvt);
predictData = NP_findDistance(predictData,data,numOfClasses,var,k);
predictData = NP_find_min_Rmax(predictData,var,numOfClasses);
[CM,accuracy] = confusionMatrix(predictData,numOfClasses,var,k,'nonparamknn');
end