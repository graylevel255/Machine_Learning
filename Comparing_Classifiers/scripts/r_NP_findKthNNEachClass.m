function [kthNN_distance,kthNN_class] = r_NP_findKthNNEachClass(x,trainClassData,k)
% Here 'trainClassData' refers to the training data of a particular class.
var = size(x,1);
n = size(trainClassData,1); % Here, n is the class size.
elem = zeros(var,1);
cols = size(trainClassData,2);
distanceCol = cols+1;
for i=1:n
    for v=1:var
        elem(v) = trainClassData(i,v);
    end
    diff = elem-x;
    dist = norm(diff);
    trainClassData(i,distanceCol) = dist;
end

%Choosing kth minimum distance of test sample from the training data
%samples of a particular class.
max = 9999999999;
for m=1:k
    min = 9999999999;
    minClass = 0;
    idxOfMin = 0;
    for j=1:n
        if(trainClassData(j,distanceCol)<min)
            min = trainClassData(j,distanceCol);
            minClass = trainClassData(j,var+1);
            idxOfMin = j;
        end
    end
    trainClassData(idxOfMin,distanceCol) = max; %Removing that element from the league of finding the next minimum
end
kthNN_distance = min;
kthNN_class = minClass;
end