function accuracy = calculate_accuracy(y,n1,n2,n3,n4,N,nc)

if nc == 4
% print accuracy
c1=0;
for i=1:n1
    c1 = c1 + y(i,1);
end

c2=0;
for i=1:n2
    c2 = c2 + y(i+n1,2);
end

t1 = n1 + n2;
c3=0;
for i=1:n3
    c3 = c3 + y(i+t1,3);
end

t2 = t1 + n3;
c4=0;
for i=1:n1
    c4 = c4 + y(i+t2,4);
end

accuracy = (c1+c2+c3+c4)/N*100;

else
    
c1=0;
for i=1:n1
    c1 = c1 + y(i,1);
end

c2=0;
for i=1:n2
    c2 = c2 + y(i+n1,2);
end

accuracy = (c1+c2)/N*100;

end