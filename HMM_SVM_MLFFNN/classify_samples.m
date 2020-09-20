function y_pred = classify_samples(w_star,X,n,nc,phi)
    for i=1:nc
        a(:,i) = w_star(i,1)*phi(:,i);
    end
    
    y_pred = predict_outputs(a,n,nc);

end