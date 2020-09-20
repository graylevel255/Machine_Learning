function x_minus_mu = xmu(n,dimension,class,mu_1)
x_minus_mu = zeros(n,dimension);

for i = 1:n
    for j = 1:dimension
      x_minus_mu(i,j) = class(i,j) - mu_1(j);
    end
end