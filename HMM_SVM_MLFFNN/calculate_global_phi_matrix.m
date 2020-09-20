function g_phi = calculate_global_phi_matrix(X,n,d)
    mu = mean(X,1);
    x_minus_mu = xmu(n,d,X,mu);
    cov = mtimes(transpose(x_minus_mu),x_minus_mu)/n;
    gv = finding_global_variance(cov,d);
    g_phi= calculate_phi(X,mu,gv,n);
end