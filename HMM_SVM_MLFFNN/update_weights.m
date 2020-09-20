function [w_new,y] = update_weights(w,t,y,nc,N,X,eta)
w_new = zeros(4,3);
epoch = 0;
% Here is initial iteration

for i=1:N % one epoch
  
        for j=1:nc
            f(j) = w(j,:)*X(i,:)';
        end
        
        for k = 1:nc
            y(i,k) = 0;
        end
            id = mymax(f(1),f(2),f(3),f(4));
            y(i,id) = 1;
            
            
        %Update weights if target output not equal to actual output
        for k=1:nc
            if(t(i,k) ~= y(i,k))
                w_new(k,:) = w_new(k,:) - eta*(y(k) - t(k))*X(i,:);
            end            
        end
            
end

epochs = 50;

% while 1     % one epoch
for i=1:epochs
       epoch = epoch + 1
       w = w_new;

    for i=1:N 
         for j=1:nc
            f(j) = w(j,:)*X(i,:)';
         end
         
         for k = 1:nc
            y(i,k) = 0;
         end
         
         id = mymax(f(1),f(2),f(3),f(4));
            y(i,id) = 1;
            
            
            %Update weights if target output not equal to actual output
         for k=1:nc
            if(t(i,k) ~= y(i,k))
                w_new(k,:) = w(k,:) - eta*(y(k) - t(k))*X(i,:);
            end            
         end   
     end
        
        if norm(w_new - w) < 0.01
            break;
        end        
end

end