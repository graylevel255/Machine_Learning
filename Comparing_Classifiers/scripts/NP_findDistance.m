function predictData = NP_findDistance(predictData,data,nc,var,k)
% Adds nc columns to predictData
% Each column contains the Kth NN distance for each of the classes
pd = size(predictData,1);
n = size(data,1);
x = zeros(var,1);
m = var+1;
initialize_nc_columns = zeros(pd,nc);
predictData = [predictData,initialize_nc_columns];
trainDataClassSize = n/nc;
for i = 1:pd
    for v = 1:var
        x(v) = predictData(i,v);
    end
for j = 1:nc
    startIndex = (j-1)*trainDataClassSize + 1;
    endIndex = j*trainDataClassSize;
    classIndex = m+j; % Column Index of class j in predictData
    [kthNN_distance,kthNN_class] = NP_findKthNNEachClass(x,data(startIndex:endIndex,:),k);
    predictData(i,classIndex) = kthNN_distance;
end
end

end