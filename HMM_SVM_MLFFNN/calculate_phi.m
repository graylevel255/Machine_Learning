function phi = calculate_phi(X,mu_1,gv1,n)

phi = zeros(n,1);

for i=1:n
    p = X(i,:) - mu_1;
    norm_val = norm(p);
    value = -0.5*norm_val^2/gv1;
    phi(i,1) = exp(value);
end