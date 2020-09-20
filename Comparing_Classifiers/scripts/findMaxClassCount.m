function maxClass = findMaxClassCount(count,numOfClasses)
max = count(1);
maxClass = 1;
for i=2:numOfClasses
    if(count(i)>max)
        max = count(i);
        maxClass = i;
    end
end
end