function [oph1,oph2,opo] = nn_plot(data,wh,bias,N,h1,LW,h2,bh2,LW3,bh3,nc)

oph1 = zeros(N,h1);

% op calculated for all samples for all nodes of first hidden layer

for i=1:N
    for j = 1:h1
        oph1(i,j) = tanh(wh(j,:)*data(:,i) + bias(j));
    end
end


oph2 = zeros(N,h2);

% op calculated for all samples for all nodes of 2nd hidden layer hidden layer 

 oph2 = oph1 * LW;
    for j=1:h2
        oph2(:,j) = tanh(oph2(:,j)  + bh2(j));
    end
    
    nc = 2;
opo = zeros(N,nc);

opo = oph2 * LW3;
for j=1:nc
     opo(:,j) = (opo(:,j)  + bh3(j));
end


end