function y = percep_classifier(w,x,N,nc)

f = zeros(N,nc);
y = zeros(N,nc);

for i=1:N   % for each trainig example
    for j=1:nc
            f(j) = w(j,:) * x(i,:)';
    end
    
    id = mymax(f(1), f(2), f(3), f(4));
    y(i,id) = 1;
end