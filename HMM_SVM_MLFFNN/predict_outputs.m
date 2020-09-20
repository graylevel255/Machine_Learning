function y = predict_outputs(a,n,nc)
if nc == 4
    for i=1:n
        for k = 1:nc
           y(i,k) = 0;
        end
    id = mymax(a(i,1),a(i,2),a(i,3),a(i,4));
    y(i,id) = 1;
    end
else
    for i=1:n
        for k = 1:nc
           y(i,k) = 0;
        end
        m = max(a(i,1),a(i,2));
        if m == a(i,1)
            y(i,1) = 1;
        else
            y(i,2) =1;
        end
    end
end
