function y = logistic_a_to_y(a)
nr = size(a,1);
nc = size(a,2);
y = zeros(nr,nc);
for i = 1:nr
    max = a(i,1);
    id = 1;
    for j = 1:nc
        if(a(i,j)>max)
            max = a(i,j);
            id = j;
        end
    end
%     for j = 1:nc
%         y(i,j) = 0;
%     end
    y(i,id) = 1;
end
end