function accuracy = logistic_accuracy(y)
n = size(y,1);
nc = size(y,2);
c = zeros(nc,1);
class_size = n/nc;
for i = 1:nc
    start_index = (i-1)*nc+1;
    end_index = i*class_size;
    for j = start_index:end_index
        c(i) = c(i) + y(j,i);
    end
end
sum = 0;
for i=1:nc
    sum = sum+c(i);
end
accuracy = 100*sum/n;
end