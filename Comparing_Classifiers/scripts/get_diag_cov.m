function cov_diag = get_diag_cov(cov,d)

cov_diag = zeros(d,d);
for i=1:d
    cov_diag(i,i) = cov(i,i);
end