function g_cov = finding_same_cov_mat(cov1,cov2,cov3,cov4,d,n_c)
    g_cov = zeros(d,d);
    
    if n_c == 4
        for i = 1:d
            g_cov(i,i) = (cov1(i,i) + cov2(i,i) + cov3(i,i) + cov4(i,i))/n_c;
        end      
    else 
        for i = 1:d
            g_cov(i,i) = (cov1(i,i) + cov2(i,i))/n_c;
        end
    end 

