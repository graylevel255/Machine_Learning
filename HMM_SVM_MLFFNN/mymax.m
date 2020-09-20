function id = mymax(n1, n2, n3, n4)

%This function returns the index of the maximum of the
% four numbers given as input
id = 1;
max =  n1;
if(n2 > max)
   max = n2;
   id = 2;
end
if(n3 > max)
   max = n3;
   id = 3;
end
if(n4 > max)
   max = n4;
   id = 4;
end