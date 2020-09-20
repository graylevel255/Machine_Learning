function predictData = r_performNonParamKNN(predictData,trainData,train_classes_sizes,nc,var,k)
predictData = r_NP_findDistance(predictData,trainData,nc,var,k,train_classes_sizes);
predictData = NP_find_min_Rmax(predictData,var,nc);
%[CM,accuracy] = confusionMatrix(predictData,numOfClasses,var,k,'nonparamknn');
end