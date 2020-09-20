function g_cov = finding_same_cov_mat_for_Bayes(cov1,cov2,cov3,cov4,d,n_c)

if n_c == 4
        for i = 1:d
            for j=1:d
              g_cov(i,j) = (cov1(i,j) + cov2(i,j) + cov3(i,j) + cov4(i,j))/n_c;
            end
        end      
    else 
        for i = 1:d
            for j=1:d
                g_cov(i,j) = (cov1(i,j) + cov2(i,j))/n_c;
            end
        end
    end 