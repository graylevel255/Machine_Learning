function sv = fetch_SVs(category,kernel)
dim = 2;
if(strcmp(category,'linear'))
fileID = fopen('linseptrain.model');
elseif(strcmp(category,'nonlinear'))
fileID = fopen('nonlinseptrain.model');
end

if(strcmp(category,'linear') && strcmp(kernel,'linear'))
L1 = textscan(fileID,'%s %s\n',1);
L2 = textscan(fileID,'%s %s\n',1);
L3 = textscan(fileID,'%s %d\n',1);
S = textscan(fileID, '%s %d\n',1);
nsv = S{2};
L4 = textscan(fileID, '%s %f %f %f %f %f %f\n',1);
L5 = textscan(fileID, '%s %d %d %d %d\n',1);
L6 = textscan(fileID, '%s %d %d %d %d\n',1);
L7 = textscan(fileID, '%s\n',1);  
sv = zeros(nsv,dim);
for i = 1:nsv
    line = textscan(fileID, '%f %f %f %d:%f %d:%f\n',1);
    sv(i,1) = line{5};
    sv(i,2) = line{7};
end
elseif(strcmp(category,'linear') && strcmp(kernel,'poly'))
    L1 = textscan(fileID,'%s %s\n',1);
    L2 = textscan(fileID,'%s %s\n',1);
    L3 = textscan(fileID,'%s %d\n',1);
    L4 = textscan(fileID,'%s %d\n',1);
    L5 = textscan(fileID,'%s %d\n',1);
    L6 = textscan(fileID,'%s %d\n',1);
    S = textscan(fileID,'%s %d\n',1);
    nsv = S{2};
    L8 = textscan(fileID, '%s %f %f %f %f %f %f\n',1);
    L9 = textscan(fileID, '%s %d %d %d %d\n',1);
    L10 = textscan(fileID, '%s %d %d %d %d\n',1);
    L11 = textscan(fileID, '%s\n',1);  
    sv = zeros(nsv,dim);
    for i = 1:nsv
        line = textscan(fileID, '%f %f %f %d:%f %d:%f\n',1);
        sv(i,1) = line{5};
        sv(i,2) = line{7};
    end

elseif(strcmp(category,'linear') && strcmp(kernel,'gaussian'))
    L1 = textscan(fileID,'%s %s\n',1);
    L2 = textscan(fileID,'%s %s\n',1);
    L3 = textscan(fileID,'%s %d\n',1);
    L4 = textscan(fileID,'%s %d\n',1);
    S = textscan(fileID,'%s %d\n',1);
    nsv = S{2};
    L6 = textscan(fileID, '%s %f %f %f %f %f %f\n',1);
    L7 = textscan(fileID, '%s %d %d %d %d\n',1);
    L8 = textscan(fileID, '%s %d %d %d %d\n',1);
    L9 = textscan(fileID, '%s\n',1);  
    sv = zeros(nsv,dim);
    for i = 1:nsv
        line = textscan(fileID, '%f %f %f %d:%f %d:%f\n',1);
        sv(i,1) = line{5};
        sv(i,2) = line{7};
    end
   
    elseif(strcmp(category,'nonlinear') && strcmp(kernel,'poly'))
    L1 = textscan(fileID,'%s %s\n',1);
    L2 = textscan(fileID,'%s %s\n',1);
    L3 = textscan(fileID,'%s %d\n',1);
    L4 = textscan(fileID,'%s %d\n',1);
    L5 = textscan(fileID,'%s %d\n',1);
    L6 = textscan(fileID,'%s %d\n',1);
    S = textscan(fileID,'%s %d\n',1);
    nsv = S{2};
    L8 = textscan(fileID, '%s %f\n',1);
    L9 = textscan(fileID, '%s %d %d\n',1);
    L10 = textscan(fileID, '%s %d %d\n',1);
    L11 = textscan(fileID, '%s\n',1);  
    sv = zeros(nsv,dim);
    for i = 1:nsv
        line = textscan(fileID, '%f %d:%f %d:%f\n',1);
        sv(i,1) = line{3};
        sv(i,2) = line{5};
    end
    
    elseif(strcmp(category,'nonlinear') && strcmp(kernel,'gaussian'))
    L1 = textscan(fileID,'%s %s\n',1);
    L2 = textscan(fileID,'%s %s\n',1);
    L3 = textscan(fileID,'%s %d\n',1);
    L4 = textscan(fileID,'%s %d\n',1);
    S = textscan(fileID,'%s %d\n',1);
    nsv = S{2};
    L6 = textscan(fileID, '%s %f\n',1);
    L7 = textscan(fileID, '%s %d %d\n',1);
    L8 = textscan(fileID, '%s %d %d\n',1);
    L9 = textscan(fileID, '%s\n',1);  
    sv = zeros(nsv,dim);
    for i = 1:nsv
        line = textscan(fileID, '%f %d:%f %d:%f\n',1);
        sv(i,1) = line{3};
        sv(i,2) = line{5};
    end
end
end









