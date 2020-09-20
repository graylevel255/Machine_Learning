function predictData = findDistance(predictData,data,k)
pd = size(predictData,1);
var = size(predictData,2)-1;
n = size(data,1);
cols = size(data,2);
distanceCol = cols+1;
for s=1:pd
for i=1:n
    elem(1) = data(i,1);
    elem(2) = data(i,2);
    dis1 = (predictData(s,1)-elem(1))^2;
    dis2 = (predictData(s,2)-elem(2))^2;
    dis = (dis1+dis2)^0.5;
    data(i,distanceCol) = dis;
end

%Choosing k minimum distances
max = 9999999999;
for m=1:k
    min = 9999999999;
    minClass = 0;
    idxOfMin = 0;
    for j=1:n
        if(data(j,distanceCol)<min)
            min = data(j,distanceCol);
            minClass = data(j,3);
            idxOfMin = j;
        end
    end
    data(idxOfMin,distanceCol) = max; %Removing that element from the league of finding the next minimum
   predictData(s,var+m+1) = minClass;
end
end
end




