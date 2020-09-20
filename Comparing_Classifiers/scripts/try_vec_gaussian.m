
%gaussian([1,1,1]', [0,0,0]', [1 0 0; 0 1 0; 0 0 1])
%gaussian([0.1,0.1,0.1]', [0,0,0]', [1 0 0; 0 1 0; 0 0 1])

%try_mat = [1 1 1; 0.1 0.1 0.1];
%vec_gaussian(try_mat, [0,0,0], [1 0 0; 0 1 0; 0 0 1])

try_mat = [18.099  0.89722 -0.066222 -2.9121 2.1101 -0.17424 -1.8464 0.54211 1.4563 1.693; 
18.677 5.1372 1.5868 0.6806 2.8568 -5.6688 2.1388 0.13572 2.9878 1.432 ;
17.905 -3.0575 -0.094315 2.4274 0.90232 2.9756 -6.9872 1.2112 1.2562 -0.99048];

mu = mean(try_mat);
c = cov(try_mat);

vec_gaussian(try_mat, mu, c)

function p = vec_gaussian(X, mu, c)
n = size(X,1);
d = size(X,2);
one_vector(1:n) = 1;
mu_vec = mu(one_vector,:);
dev = X - mu_vec;
p = (1/(sqrt((2*pi)^d * det(c)))) * exp(-0.5 .* sum((dev*inv(c)) .* dev,2));
end


%x is row vector
function probab = gaussian(x, mu, c)
    d = size(x, 1);
    temp = inv(c)*(x-mu);
    pow = (x-mu)'*temp;
    pow = -0.5*pow;
    constant_term = 1/(((2*3.14)^(d/2))*det(c));
    probab = constant_term*exp(pow);
 end