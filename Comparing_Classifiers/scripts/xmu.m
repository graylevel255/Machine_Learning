function x_minus_mu = xmu(sz,dimension,class,mu_1)
x_minus_mu = zeros(sz,dimension);

for i = 1:sz
    for j = 1:dimension
      x_minus_mu(i,j) = class(i,j) - mu_1(j);
    end
end