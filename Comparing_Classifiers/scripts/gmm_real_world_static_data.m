%Divide the data by classes
B = importdata('./datasets/group23/real_world_static.txt');
class_label_col_no = 9;
n = size(B,1);
dim = size(B,2)-1;
no_of_classes = 3;
%Adding random noise to col5 and col6
% r5 = normrnd(0, 0.5, n, 1);
% r6 = normrnd(0, 0.5, n, 1);
% A(:, 5) = A(:, 5) + r5;
% A(:, 6) = A(:, 6) + r6;
%Remove col 5 and 6
A = B(:,1:4);
A = [A B(:,7:9)];
class_label_col_no = 7;
%min-max normalize
dim = size(A,2)-1;
for i=1:dim
    min_val = min(A(:,i));
    max_val = max(A(:,i));
    den_term = max_val-min_val;
    A(:,i) = A(:,i)-min_val*ones(n,1);
    A(:,i) = A(:,i)/den_term;
end
class1_data = A(A(:,class_label_col_no)==1, :);
class2_data = A(A(:,class_label_col_no)==2, :);
class3_data = A(A(:,class_label_col_no)==3, :);


%Divide into train, test, val
[class1_train, class1_test, class1_val] = divide_data(class1_data);
[class2_train, class2_test, class2_val] = divide_data(class2_data);
[class3_train, class3_test, class3_val] = divide_data(class3_data);


disp('class1');
params_class1 = train_model_single_class(class1_train, class1_val);
disp('class2');
params_class2 = train_model_single_class(class2_train, class2_val);
disp('class3');
params_class3 = train_model_single_class(class3_train, class3_val);


params{1} = params_class1;
params{2} = params_class2;
params{3} = params_class3;


input_data = [class1_train; class2_train; class3_train];
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
disp('Train Accuracy');
accuracy
confusion_matrix

input_data = [class1_val; class2_val; class3_val];
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
disp('Val Accuracy');
accuracy

input_data = [class1_test; class2_test; class3_test];
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
disp('Test Accuracy');
accuracy
confusion_matrix

%divides data 70% into train, 10%val, 20%test
function [train_data, test_data, val_data] = divide_data(class_data)
    n = size(class_data, 1);
    rand_perm = randperm(n);
    end_index_for_train = round(0.7*n);
    train_index = rand_perm(1:end_index_for_train);
    end_index_for_val = round(0.8*n);
    val_index = rand_perm(end_index_for_train+1 : end_index_for_val);
    test_index = rand_perm(end_index_for_val+1 : n);
    
    train_data = class_data(train_index, :);
    test_data = class_data(test_index, :);
    val_data = class_data(val_index, : );
end

function params_single_class = train_model_single_class(class_data, val_set_class_data)
    
    n_data = size(class_data, 1);
    dim = size(class_data, 2)-1;
    class_data = class_data(:,1:dim);
    %val_set_class_data = val_set_class_data(:,1:dim);
    best_w = [];
    best_mu = [];
    best_sigma = [];
    l_max = -Inf;
    best_q = 0;
    Q_min = 2;
    Q_max = 2;  %TODO: make this n_data/10, now changing for test purpose.
   
    
    Q = [Q_min:1:Q_max];
    l_per_Q = [];
    l_val_per_Q = [];
    for i=1 : size(Q,2)
        q = Q(i);
        spl_flag = false;
        disp(['For Q=' num2str(q) ''])
        [w,mu, sigma] = get_initial_estimates(class_data, q);
        iter = 1;
          while true
            %iter
            if (spl_flag == true)
                [w,mu, sigma] = get_initial_estimates(class_data, q);
                spl_flag = false;
            end
            l_old = calc_likelihood(class_data, w, mu, sigma,q);
            resp_term = calc_responsibility_terms(class_data, w, sigma, mu, q);
            if(isnan(resp_term(isnan(resp_term))) == true)
                spl_flag = true;
                resp_term;
            end
            [w_new,mu_new,sigma_new] = calc_new_params(resp_term,class_data, q);
             if(isnan(w_new(isnan(w_new))) == true)
                spl_flag = true;
                w_new;
             end
             if(isnan(mu_new(isnan(mu_new))) == true)
                spl_flag = true; 
                mu_new;
             end
             if(isnan(sigma_new(isnan(sigma_new))) == true)
                spl_flag = true; 
                sigma_new;
             end
     
            l_new = calc_likelihood(class_data, w_new, mu_new, sigma_new,q);
            %l_val = calc_likelihood(val_set_class_data, w_new,mu_new, sigma_new,q);
            
            if(abs(l_old - l_new) < 0.01)
            
                l_per_Q = [l_per_Q l_new];
                %l_val_per_Q = [l_val_per_Q l_val];
                if (l_new > l_max)
                    l_max = l_new;
                    best_w = w_new;
                    best_mu = mu_new;
                    best_sigma = sigma_new;
                    best_q = q;
                end
                w = w_new;
                mu = mu_new;
                sigma = sigma_new;
                break
            end
            w = w_new;
            sigma = sigma_new;
            mu = mu_new;
            iter = iter+1;
          end

    end
    l_per_Q
    %l_val_per_Q
    params_single_class = {best_w, best_sigma,best_mu, best_q};
end

 function [w,mu, sigma] = get_initial_estimates(class_data, q)

    n = size(class_data, 1);
    dim = size(class_data, 2);
    cov_all_clusters = [];
    w = [];
    [idx, means] = kmeans(class_data, q);
    %'MaxIter',50
    for c_index = 1:q
        p = idx == c_index;
        single_cluster_data = class_data(p,:);
        cov_single_cluster = calc_covariance(single_cluster_data, means(c_index,:));
        cov_all_clusters = [cov_all_clusters cov_single_cluster];
        w = [w sum(p)/n];
        if(w(w == inf) == inf)
            w;
        end
        if(w(w == -inf) == -inf)
            w;
        end
    end
    mu = means;
    sigma = cov_all_clusters;
 
 end
 
 function c = calc_covariance(A, means)
    n = size(A,1);
    d = size(A,2);
    one_vector(1:n) = 1;
    %duplicate mean vec
    mean_mat = means(one_vector, :);
    A_sub_mean = A - mean_mat;
    c = (A_sub_mean'*A_sub_mean)/n;
    if(rcond(c) < 1e-6)
        %reset mean and add noise of covar matrix
        for i=1:d
            c(i,i) = c(i,i) + normrnd(0, 0.5);
        end       
    end
 end

 function c = calc_cov_with_resp(A, mean, resp_term) %verify this with alt impl.
    n = size(A,1);
    d = size(A,2);
    sum1 = zeros(d,d);
    for i = 1:n
        example = A(i,:);
        sum1 = sum1 + resp_term(i)*(example-mean)'*(example-mean);
    end
    sum1 = sum1/sum(resp_term);
    c = sum1;
    if(rcond(c) < 1e-6)
        %reset mean and add noise of covar matrix
        mean = mean + normrnd(0, 0.5, 1, size(mean,2));
        for i=1:d
            c(i,i) = c(i,i) + normrnd(0, 0.5);
        end       
    end
    if(isnan(rcond(c)) == true)
        for i=1:d
            for j=1:d
                c(i,j) = normrnd(0, 0.5);
            end
        end  
    end
 end
 
  %x is a col vector
 %mu is a col vec
 function probab = gaussian(x, mu, c)
    d = size(x, 1);
    temp = inv(c)*(x-mu);
    pow = (x-mu)'*temp;
    pow = -0.5*pow;
    constant_term = 1/(((2*3.14)^(d/2))*det(c));
    probab = constant_term*exp(pow);
 end
 
 %x is row vector form, ie. 1st row is 1st example in n-dim
 %mu is also a row vec
 function resp_term = calc_responsibility_terms(x, w, c, mu, Q)
    n = size(x,1);
    dim = size(x,2);
    resp_term = [];
    if(isnan(w(isnan(w))) == true)
        w
    end
    if(isnan(c(isnan(c))) == true)
        c
    end
    if(isnan(mu(isnan(mu))) == true)
        mu
    end
    for i = 1:n
        resp_term_single_sample = [];
        one_sample = x(i,:);
        for q=1:Q
            wq = w(q);
            mu_q = mu(q,:);
            cq = c(1:dim, dim*(q-1)+1:dim*(q-1)+dim);
            if(rcond(cq) < 1e-6 ||  isnan(rcond(cq)) == true)
                cq
            end
            r = wq*gaussian(one_sample',mu_q', cq);
            if(r < 1e-5)
                r = 0.0;
            end
            if(isnan(r) == true)
                r;
            end
            resp_term_single_sample = [resp_term_single_sample r];
         
           if(isnan(resp_term_single_sample(isnan(resp_term_single_sample))) == true)
                resp_term_single_sample;
            end
        end
        tot_prob = sum(resp_term_single_sample,2);
        %if(tot_prob > 0)
            resp_term_single_sample = resp_term_single_sample./tot_prob;
        %end
            
%             %to avoid nan
%              val = max(log(resp_term_single_sample));
%              resp_term_single_sample = exp(-val)*resp_term_single_sample;
            
        %end
        if(isnan(resp_term_single_sample(isnan(resp_term_single_sample))) == true)
                resp_term_single_sample;
                
                i;
            end
        resp_term = [resp_term; resp_term_single_sample]; 
    end
 end
 
 function [w,mu,sigma] = calc_new_params(resp_term,class_data,q)
    n = size(class_data, 1);
    gamma = resp_term';
    mu = [];
    sigma = [];
    w = [];
            for j = 1:q
                rnq = gamma(j,:); % gamma term for q-th componet, for all examples
                mu_q = (rnq*class_data)/sum(rnq);
                c_q = calc_cov_with_resp(class_data, mu_q, rnq);
            
                mu = [mu; mu_q];
                sigma = [sigma c_q];
                w = [w sum(rnq)/n];
                if(isnan(w(isnan(w))) == true)
                    w;
                end
            end
 end
 
 %w is row vector, mu is cluster1 mu1, mu2; cluster2 mu1, mu2 like that
 function likelihood = calc_likelihood(X,w, mu, c, q)
    g_value_per_component = [];
    dim = size(X, 2);
    n = size(X,1);
    sum1 = 0;
    for j=1:n
        x = X(j,:)';
        s = 0;
        for i = 1:q
            wq = w(i);
            mu_q = mu(i,:);
            cq = c(1:dim, dim*(i-1)+1:dim*(i-1)+dim);
            s = s + wq*gaussian(x, mu_q', cq);
        end
        sum1 = sum1 + log(s);
    end
    likelihood = sum1;
 end
 
  
 %D has format (x1,x2, x_dim, orig_class, pred_class)
 function [confusion_matrix, accuracy] =calc_confusion_matrix(D, no_of_classes)
    n = size(D,1);
    dim = size(D,2)-2;
    confusion_matrix = zeros(no_of_classes,no_of_classes);
    for i=1:n
        orig_label = D(i,dim+1);
        assigned_label = D(i,dim+2);
        confusion_matrix(orig_label, assigned_label) = confusion_matrix(orig_label, assigned_label)+1;
    end
    accuracy = 0;
    for i =1:no_of_classes
        accuracy = accuracy + confusion_matrix(i,i);
        confusion_matrix(i,:) = (confusion_matrix(i,:)/sum(confusion_matrix(i,:)))*100; 
    end
    accuracy = (accuracy/n)*100;
 end

  %last param is a cell array, format {w1,c1,mu1,q1, w2, c2, mu2, q2.. params for each class}
 %D is a labelled dataset of examples.
 function output = test_gmm_classifier(D, no_of_classes, params, is_decision_region_plot_reqd)
    n = size(D,1);
    dim = size(D,2)-1;
    probab_vec_all_classes = zeros(n, no_of_classes);
    
    for class_ctr = 1:no_of_classes
        disp(['Class' num2str(class_ctr) ''])
        [w,c,mu,q] = params{class_ctr}{1 : 4};
        for i=1:n
            single_sample = D(i,1:dim);
            probab = gmm_classifier_single_class(single_sample, w, c, mu, q);
            probab_vec_all_classes(i,class_ctr) = probab;
        end
    end
    predicted_labels = [];
    for i=1:n
        [val, idx] = max(probab_vec_all_classes(i,:));
        class_label = idx;
        predicted_labels = [predicted_labels; class_label];
    end
    output = [D predicted_labels]; 
    if (is_decision_region_plot_reqd == true)
        
        plot_decision_region(output, no_of_classes)
    end
 end
 
  %x, w, mu, q is a row vectors
 function probab = gmm_classifier_single_class(x,w,c,mu,q)
    dim = size(x,2);
    g_value = [];
    for i = 1:q
        wq = w(i);
        mu_q = mu(i,:);
        cq = c(1:dim, dim*(i-1)+1:dim*(i-1)+dim);
        res = gaussian(x', mu_q', cq);
        g_value = [g_value res];
    end
    probab = w * g_value';
    
 end


