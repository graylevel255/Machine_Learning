function predictData = r_NP_findDistance(predictData,trainData,nc,var,k,train_classes_sizes)
% Adds nc columns to predictData
% Each column contains the Kth NN distance for each of the classes
pd = size(predictData,1);
x = zeros(var,1);
m = var + 1;
initialize_nc_columns = zeros(pd,nc);
predictData = [predictData,initialize_nc_columns];
for i = 1:pd
    for v = 1:var
        x(v) = predictData(i,v);
    end
elems = 0;
for j = 1:nc
    if j==1
        startIndex = 1;
    else
        startIndex = elems+1;
    end
        elems = elems + train_classes_sizes(j);
    endIndex = elems;
    classIndex = m+j; % Column Index of class j in predictData
    [kthNN_distance,kthNN_class] = r_NP_findKthNNEachClass(x,trainData(startIndex:endIndex,:),k);
     predictData(i,classIndex) = kthNN_distance;
end
end

end