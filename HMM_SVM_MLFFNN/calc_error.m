% Calculate SSE 
function sse = calc_error(t,y,N,nc)
sse = 0;
for i=1:N
    for j=1:nc
        sse = sse + (t(i,j) - y(i,j)).^2;
    end
end

sse = 0.5*sse;