function w_new = weight_update(w,eta,y,t,global_phi,n,d,nc,phi)
    iter = 0;
    for i=1:nc
        a(:,i) = w(i,1)*phi(:,i);
    end
    
    y = predict_outputs(a,n,nc);
    w_new = w - eta*(transpose(y-t))*global_phi;

while 1
    iter = iter + 1
    w = w_new;
    for i=1:nc
        a(:,i) = w(i,1)*phi(:,i);
    end
    
    y = predict_outputs(a,n,nc);
    w_new = w - eta*(transpose(y-t))*global_phi;

    if norm(w_new - w) < 0.01  || iter == 2000
        break;
    end
end