%function t = construct_t_matrix(category,y)
function t = construct_t_matrix(y)
n = size(y,1);
nc = size(y,2);
t = zeros(n,nc);
class_size = n/nc;
for i = 1:nc
    start_index = (i-1)*class_size+1;
    end_index = i*class_size;
    for j = start_index:end_index
     t(j,i) = 1;   
    end
end
% ntrain = size(y,1);
% nc = size(y,2);
% t = zeros(ntrain,nc);
% if(strcmp(category,'linear'))
% for i = 1:250
%     t(i,1) = 1;
% end
% for i = 251:500
%     t(i,2) = 1;
% end
% for i = 501:750
%     t(i,3) = 1;
% end
% for i = 751:1000
%     t(i,4) = 1;
% end
% end
end