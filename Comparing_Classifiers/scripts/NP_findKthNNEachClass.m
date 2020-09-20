function [kthNN_distance,kthNN_class] = NP_findKthNNEachClass(x,data,k)
% Here 'data' refers to the training data of a particular class.
var = size(x,1);
n = size(data,1); % Here, n is the class size.
elem = zeros(var,1);
cols = size(data,2);
distanceCol = cols+1;
for i=1:n
    for v=1:var
        elem(v) = data(i,v);
    end
    diff = elem-x;
    dist = norm(diff);
    data(i,distanceCol) = dist;
end

%Choosing kth minimum distance of test sample from the training data
%samples of a particular class.
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
end
kthNN_distance = min;
kthNN_class = minClass;
end