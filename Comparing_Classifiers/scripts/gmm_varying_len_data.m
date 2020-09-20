parent_path = './datasets/group23/feats_surf';
file1_loc = './datasets/group23/feats_surf/008.bathtub';
file2_loc = './datasets/group23/feats_surf/109.hot-tub';
file3_loc = './datasets/group23/feats_surf/138.mattress';
file4_loc = './datasets/group23/feats_surf/232.t-shirt';
file5_loc = './datasets/group23/feats_surf/129.leopards-101';

D = dir(fullfile(file1_loc, '*.txt'));
A = cell(size(D));
class1_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file1_loc,D(i).name)); 
      class1_data = [class1_data ; A{i}];
end

D = dir(fullfile(file2_loc, '*.txt'));
A = cell(size(D));
class2_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file2_loc,D(i).name)); 
      class2_data = [class2_data ; A{i}];
end

D = dir(fullfile(file3_loc, '*.txt'));
A = cell(size(D));
class3_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file3_loc,D(i).name)); 
      class3_data = [class3_data ; A{i}];
end

D = dir(fullfile(file4_loc, '*.txt'));
A = cell(size(D));
class4_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file4_loc,D(i).name)); 
      class4_data = [class4_data ; A{i}];
end

D = dir(fullfile(file5_loc, '*.txt'));
A = cell(size(D));
class5_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file5_loc,D(i).name)); 
      class5_data = [class5_data ; A{i}];
end
dim = size(class1_data,2);
n = size(class1_data,1);
for i=1:dim
    min_val = min(class1_data(:,i));
    max_val = max(class1_data(:,i));
    den_term = max_val-min_val;
    class1_data(:,i) = class1_data(:,i)-min_val*ones(n,1);
    class1_data(:,i) = class1_data(:,i)/den_term;
end
%to check code, remove later
class1_data = class1_data(1:300,:);
class2_data = class2_data(1:300,:);
class3_data = class3_data(1:300,:);
class4_data = class4_data(1:300,:);
class5_data = class5_data(1:300,:);
%Divide into train, test, val
[class1_train, class1_test, class1_val] = divide_data(class1_data);
[class2_train, class2_test, class2_val] = divide_data(class2_data);
[class3_train, class3_test, class3_val] = divide_data(class3_data);
[class4_train, class4_test, class4_val] = divide_data(class4_data);
[class5_train, class5_test, class5_val] = divide_data(class5_data);


disp('class1');
params_class1 = train_model_single_class(class1_train, class1_val);
disp('class2');
params_class2 = train_model_single_class(class2_train, class2_val);
disp('class3');
params_class3 = train_model_single_class(class3_train, class3_val);
disp('class4');
params_class4 = train_model_single_class(class4_train, class4_val);
disp('class5');
params_class5 = train_model_single_class(class5_train, class5_val);


params{1} = params_class1;
params{2} = params_class2;
params{3} = params_class3;
params{4} = params_class4;
params{5} = params_class5;


input_data = [class1_train ones(n,1); class2_train 2*ones(n,1); class3_train 3*ones(n,1); class4_train 4*ones(n,1); class5_train 5*ones(n,1)];
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
disp('Train Accuracy');
accuracy

input_data = [class1_val ones(n,1); class2_val 2*ones(n,1); class3_val 3*ones(n,1); class4_val 4*ones(n,1); class5_val 5*ones(n,1)];
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
disp('Val Accuracy');
accuracy

input_data = [class1_test ones(n,1); class2_test 2*ones(n,1); class3_test 3*ones(n,1); class4_test 4*ones(n,1); class5_test 5*ones(n,1)];
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
disp('Val Accuracy');
accuracy



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

function params_single_class = train_model_single_class(class_data, val_set_class_data, no_of_classes)
    
    n_data = size(class_data, 1);
    dim = size(class_data, 2);
    best_w = [];
    best_mu = [];
    best_sigma = [];
    l_max = -Inf;
    best_q = 0;
    Q_min = 2;
    Q_max = 4;
    Q = [Q_min:1:Q_max];
    l_per_Q = [];
    l_val_per_Q = [];
    for i=1 : size(Q,2)
        q = Q(i);
        disp(['For Q=' num2str(q) ''])
        [w,mu, sigma] = get_initial_estimates(class_data, q);
        
          while true

            l_old = calc_likelihood(class_data, w, mu, sigma,q);
            resp_term = calc_responsibility_terms(class_data, w, sigma, mu, q);
            [w_new,mu_new,sigma_new] = calc_new_params(resp_term,class_data, q);
            l_new = calc_likelihood(class_data, w_new, mu_new, sigma_new,q);
            l_val = calc_likelihood(val_set_class_data, w_new,mu_new, sigma_new,q);
            
            if(abs(l_old - l_new) < 0.01)
            
                l_per_Q = [l_per_Q l_new];
                l_val_per_Q = [l_val_per_Q l_val];
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
          end

    end
    l_per_Q
    l_val_per_Q
    %returning q=6 paramters for now, later we ll get the best on val set
    %ones
    params_single_class = {best_w, best_sigma,best_mu, best_q};
end
    
 function [w,mu, sigma] = get_initial_estimates(class_data, q)
    
    n = size(class_data, 1);
    dim = size(class_data, 2);
    cov_all_clusters = [];
    w = [];
    [idx, means] = kmeans(class_data, q);
    for c_index = 1:q
        p = idx == c_index;
        single_cluster_data = class_data(p,:);
        cov_single_cluster = calc_covariance(single_cluster_data, means(c_index,:));
        cov_all_clusters = [cov_all_clusters cov_single_cluster];
        w = [w sum(p)/n];
    end
    mu = means;
    sigma = cov_all_clusters;

%  n = size(class_data, 1);
%     dim = size(class_data, 2);
%     cov_all_clusters = [];
%     w = [];
%     eps = 1e-5;
%     flag = true;
%     while(flag == true)
%         %[idx, means] = kmeans(class_data, q, 'MaxIter',10);
%         [idx, means] = kmeans(class_data, q);
%         for c_index = 1:q
%             flag = false;
%             p = idx == c_index;
%             single_cluster_data = class_data(p,:);
%             cov_single_cluster = calc_covariance(single_cluster_data, means(c_index,:));
%             if (~(all(eig(cov_single_cluster) > eps)))
%                 flag = true;
%                 break;
%             end
%             cov_all_clusters = [cov_all_clusters cov_single_cluster];
%             w = [w sum(p)/n];     
%         end
%     end
%     
%    
%     mu = means;
%     sigma = cov_all_clusters;
 
 end
 
 %check def of covariance with surity for N, N-1
 function c = calc_covariance(A, means)
    n = size(A,1);
    d = size(A,2);
    one_vector(1:n) = 1;
    %duplicate mean vec
    mean_mat = means(one_vector, :);
    A_sub_mean = A - mean_mat;
    c = (A_sub_mean'*A_sub_mean)/n;
    if(rcond(c) < 1e-3)
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
    if(rcond(c) < 1e-3)
        for i=1:d
            c(i,i) = c(i,i) + normrnd(0, 0.5);
        end     
    end
 end
 
 
 %x is row vector form, ie. 1st row is 1st example in n-dim
 %mu is also a row vec
 function resp_term = calc_responsibility_terms(x, w, c, mu, Q)
    n = size(x,1);
    dim = size(x,2);
    resp_term = [];
    resp_term_single_cluster = [];
    for q=1:Q
        wq = w(q);
        mu_q = mu(q,:);
        cq = c(1:dim, dim*(q-1)+1:dim*(q-1)+dim);
        resp_term_single_cluster = wq*vec_gaussian(x,mu_q,cq);
        resp_term = [resp_term resp_term_single_cluster];
    end
%     for i = 1:n
%         resp_term_single_sample = [];
%         one_sample = x(i,:);
%         for q=1:Q
%             wq = w(q);
%             mu_q = mu(q,:);
%             cq = c(1:dim, dim*(q-1)+1:dim*(q-1)+dim);
%             resp_term_single_sample = [resp_term_single_sample wq*gaussian(one_sample',mu_q', cq)];
%         end
%         resp_term_single_sample = resp_term_single_sample./sum(resp_term_single_sample,2);
%         resp_term = [resp_term; resp_term_single_sample]; 
%     end
    for i = 1:n
        resp_term(i,:) = resp_term(i,:)/sum(resp_term(i,:), 2);
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
            end
 end
 
 %w is row vector, mu is cluster1 mu1, mu2; cluster2 mu1, mu2 like that
  %w is row vector, mu is cluster1 mu1, mu2; cluster2 mu1, mu2 like that
 function likelihood = calc_likelihood(X,w, mu, c, q)
%     g_value_per_component = [];
%     dim = size(X, 2);
%     n = size(X,1);
%     sum1 = 0;
%     for j=1:n
%         x = X(j,:)';
%         s = 0;
%         for i = 1:q
%             wq = w(i);
%             mu_q = mu(i,:);
%             cq = c(1:dim, dim*(i-1)+1:dim*(i-1)+dim);
%             s = s + wq*gaussian(x, mu_q', cq);
%         end
%         sum1 = sum1 + log(s);
%     end
%     likelihood = sum1;
dim = size(X, 2);
n = size(X,1);
s = zeros(n,1);
for i = 1:q
    wq = w(i);
    mu_q = mu(i,:);
    cq = c(1:dim, dim*(i-1)+1:dim*(i-1)+dim);
    wgaussian_for_q = wq*vec_gaussian(X, mu_q, cq);
    s = s + wgaussian_for_q;
end
likelihood = sum(log(s));
 end
 
 %last param is a cell array, format {w1,c1,mu1,q1, w2, c2, mu2, q2.. params for each class}
 %D is a labelled dataset of examples.
 function output = test_gmm_classifier(D, no_of_classes, params, is_decision_region_plot_reqd)
    n = size(D,1);
    dim = size(D,2)-1;
    probab_vec_all_classes = [];
    
    for class_ctr = 1:no_of_classes
        disp(['Class' num2str(class_ctr) ''])
        [w,c,mu,q] = params{class_ctr}{1 : 4};
        probab = gmm_classifier_single_class(D,w,c,mu,q);
%         for i=1:n
%             single_sample = D(i,1:dim);
%             probab = gmm_classifier_single_class(single_sample, w, c, mu, q);
%             probab_vec_all_classes(i,class_ctr) = probab;
%         end
        probab_vec_all_classes = [probab_vec_all_classes probab'];
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
        res = vec_gaussian(x, mu_q', cq);
        g_value = [g_value; res'];
    end
    probab = w * g_value;
    
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

function p = vec_gaussian(X, mu, c)
n = size(X,1);
d = size(X,2);
one_vector(1:n) = 1;
mu_vec = mu(one_vector,:);
dev = X - mu_vec;
p = exp(-0.5 * sum((dev*inv(c).*dev),2))/sqrt((2*pi)^d * det(c));
for i=1:n
    if(p(i,1) < 1e-5)
        p(i,1) = 0;
    if(p(i,1) > 1e+5)
        p(i,1) = 1;  
    end
    end
end
end