function discriminant  = calculate_g_of_x(data, mu, cov_mat, p)

%This function calculates the discriminant function of a class given its
%parameters

n = numel(data(:,1));
x_m_u = xmu(n,2,data,mu);
discriminant = zeros(n,2);

for i = 1:n
    t1 = mtimes(x_m_u(i,:), mtimes(inv(cov_mat),transpose(x_m_u(i,:))));
    discriminant(i,:) = -0.5*t1 + log(p);
    
end