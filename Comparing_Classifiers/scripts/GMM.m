
%Analysis for Linearly Seperable data
disp('Analysis for Linearly Seperable data');
[D_lin_sep, no_of_classes] = pick_dataset("lin-sep", "train", false);
figure();
plot_data(D_lin_sep, no_of_classes);
n = size(D_lin_sep,1);
val_set = pick_dataset("lin-sep", "val", false);
params = train_model(D_lin_sep, no_of_classes, 2, val_set);
disp('Best model')
params{1}{4}
params{2}{4}
params{3}{4}
params{4}{4}
input_data = pick_dataset("lin-sep", "train", true);
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
accuracy
confusion_matrix
[X,Y] = meshgrid(-15:0.1:20);
D = [];
for i = 1 : size(Y,1)
    for j = 1 : size(Y,2)
        D = [D; X(i,j) Y(i,j) 0];
    end
end
D = test_gmm_classifier(D, no_of_classes, params, true);
plot_data(D_lin_sep, no_of_classes);
%Test on Val set
disp('For val set') 
input_data = pick_dataset("lin-sep", "val", true);
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
accuracy
confusion_matrix

%Analysis for Non-Linearly Seperable data
disp('Analysis for Non-Linearly Seperable data');
[D_lin_sep, no_of_classes] = pick_dataset("non-lin-sep", "train", false);
figure();
plot_data(D_lin_sep, no_of_classes);
n = size(D_lin_sep,1);
val_set = pick_dataset("non-lin-sep", "val", false);
params = train_model(D_lin_sep, no_of_classes, 2, val_set);
disp('Best model');
params{1}{4}
params{2}{4}
input_data = pick_dataset("non-lin-sep", "train", true);
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
accuracy
confusion_matrix
[X,Y] = meshgrid(-15:0.1:20);
D = [];
for i = 1 : size(Y,1)
    for j = 1 : size(Y,2)
        D = [D; X(i,j) Y(i,j) 0];
    end
end
test_gmm_classifier(D, no_of_classes, params, true);
plot_data(D_lin_sep, no_of_classes);

%test on val set
disp('For val set')
input_data = pick_dataset("non-lin-sep", "val", true);
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
accuracy
confusion_matrix


%Analysis for Overlapping data
disp('Analysis for Overlapping data');
[D_lin_sep,no_of_classes] = pick_dataset("overlapping", "train", false);
figure();
plot_data(D_lin_sep, no_of_classes);
n = size(D_lin_sep,1);
val_set = pick_dataset("overlapping", "val", false);
params = train_model(D_lin_sep, no_of_classes, 2, val_set);
disp('Best model');
params{1}{4}
params{2}{4}
params{3}{4}
params{4}{4}
input_data = pick_dataset("overlapping", "train", true);
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
accuracy
confusion_matrix
[X,Y] = meshgrid(-15:0.1:20);
D = [];
for i = 1 : size(Y,1)
    for j = 1 : size(Y,2)
        D = [D; X(i,j) Y(i,j) 0];
    end
end
test_gmm_classifier(D, no_of_classes, params, true);
plot_data(D_lin_sep, no_of_classes);

%val set
disp('For val set')
input_data = pick_dataset("overlapping", "val", true);
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
accuracy
confusion_matrix

disp('For test set')
input_data = pick_dataset("overlapping", "test", true);
predicted_output = test_gmm_classifier(input_data, no_of_classes,params, false);
[confusion_matrix, accuracy] = calc_confusion_matrix(predicted_output,no_of_classes);
accuracy
confusion_matrix

% 
% 
% %n = size(D_lin_sep,1);
% %params = train_model(D_lin_sep, no_of_classes, 2); %todo: make no of classes to 4 later
% %params = train_model(class3_data,1,2); %only take class 3 data to test
% %calc_covariance([4 2 0.5; 4.2 2.1 0.59; 3.9 2.0 0.58; 4.3 2.1 0.62; 4.1 2.2 0.63],[4.1000 2.0800 0.5840] )
% 
% 
% 
% %gaussian([1,1,1]', [0,0,0]', [1 0 0; 0 1 0; 0 0 1])
% %gaussian([0.1,0.1,0.1]', [0,0,0]', [1 0 0; 0 1 0; 0 0 1])
% %reshaping input data for predition.
% 
% %input_data = pick_dataset("overlapping", "train", true);
% %input_data = pick_dataset("non-lin-sep", "train", true);
% %predicted_output = test_gmm_classifier(input_data, 4, {w1,c1,mu1,q1, w2,c2,mu2,q2, w3,c3,mu3,q3, w4,c4,mu4,q4});
% 
% confusion_matrix
% %For decision region


%param1: 'lin-sep/non-lin-sep/overlapping', 
%param2: train/test/val,
%param3: set flag = true, when data is needed to be formatted for
%classifier
function [D, no_of_classes] = pick_dataset(type_of_data, purpose, flag)

    if(type_of_data == "lin-sep")
        disp('Linearly seperable data')
        if(purpose =="train")
            class1_data = importdata('./datasets/group23/linearly_separable/class1_train.txt');
            class2_data = importdata('./datasets/group23/linearly_separable/class2_train.txt');
            class3_data = importdata('./datasets/group23/linearly_separable/class3_train.txt');
            class4_data = importdata('./datasets/group23/linearly_separable/class4_train.txt');
        end
        if(purpose == "val")
            class1_data = importdata('./datasets/group23/linearly_separable/class1_val.txt');
            class2_data = importdata('./datasets/group23/linearly_separable/class2_val.txt');
            class3_data = importdata('./datasets/group23/linearly_separable/class3_val.txt');
            class4_data = importdata('./datasets/group23/linearly_separable/class4_val.txt');
        end
        if(purpose == "test")
            class1_data = importdata('./datasets/group23/linearly_separable/class1_test.txt');
            class2_data = importdata('./datasets/group23/linearly_separable/class2_test.txt');
            class3_data = importdata('./datasets/group23/linearly_separable/class3_test.txt');
            class4_data = importdata('./datasets/group23/linearly_separable/class4_test.txt');
        end
        if(flag == false)
            D = [class1_data class2_data class3_data class4_data];
            no_of_classes = 4;
        end  
        if (flag == true)
            n = size(class1_data,1);
            D = [class1_data ones(n,1); class2_data 2*ones(n,1); class3_data 3*ones(n,1); class4_data 4*ones(n,1)];
            no_of_classes = 4;
        end
    end
    
    if(type_of_data == "non-lin-sep")
        disp('Non-Linearly seperable data')
        if(purpose == "train")
            class1_data = importdata('./datasets/group23/nonlinearly_separable/class1_train.txt');
            class2_data = importdata('./datasets/group23/nonlinearly_separable/class2_train.txt');
        end
        if(purpose == "val")
            class1_data = importdata('./datasets/group23/nonlinearly_separable/class1_val.txt');
            class2_data = importdata('./datasets/group23/nonlinearly_separable/class2_val.txt');
        end
        if(purpose == "test")
            class1_data = importdata('./datasets/group23/nonlinearly_separable/class1_test.txt');
            class2_data = importdata('./datasets/group23/nonlinearly_separable/class2_test.txt');
        end
        if(flag == false)
            D = [class1_data class2_data];
            no_of_classes = 2;
        end  
        if (flag == true)
            n = size(class1_data,1);
            D = [class1_data ones(n,1); class2_data 2*ones(n,1)];
            no_of_classes = 2;
        end
    end
    
    if(type_of_data == "overlapping")
        disp('Overlapping data')
        if(purpose == "train")
            class1_data = importdata('./datasets/group23/overlapping/class1_train.txt');
            class2_data = importdata('./datasets/group23/overlapping/class2_train.txt');
            class3_data = importdata('./datasets/group23/overlapping/class3_train.txt');
            class4_data = importdata('./datasets/group23/overlapping/class4_train.txt');
        end
        if(purpose == "val")
            class1_data = importdata('./datasets/group23/overlapping/class1_val.txt');
            class2_data = importdata('./datasets/group23/overlapping/class2_val.txt');
            class3_data = importdata('./datasets/group23/overlapping/class3_val.txt');
            class4_data = importdata('./datasets/group23/overlapping/class4_val.txt');
        end
        if(purpose == "test")
            class1_data = importdata('./datasets/group23/overlapping/class1_test.txt');
            class2_data = importdata('./datasets/group23/overlapping/class2_test.txt');
            class3_data = importdata('./datasets/group23/overlapping/class3_test.txt');
            class4_data = importdata('./datasets/group23/overlapping/class4_test.txt');
        end
        if(flag == false)
            D = [class1_data class2_data class3_data class4_data];
            no_of_classes = 4;
        end  
        if (flag == true)
            n = size(class1_data,1);
            D = [class1_data ones(n,1); class2_data 2*ones(n,1); class3_data 3*ones(n,1); class4_data 4*ones(n,1)];
            no_of_classes = 4;
        end
    end
        
            
end

function plot_data(D, no_of_classes)
    
    dyellow = [0.8438,0.6602,0];
    dpink = [0.4648,0.0078,0.4570];
    dblue = [0.0078,0.3438,0.4648];
    dgreen = [0.0117,0.4453,0.0703];
   
    dim = size(D,2)/no_of_classes;
    %figure;
    k = 1;
    for index = 1:dim:(dim*no_of_classes-1)
        class_data = D(:, index:index+1);
        ci_x1 = class_data(:,1);
        ci_x2 = class_data(:,2);
        if (no_of_classes == 4)
            if( k == 1)
                scatter(ci_x1,ci_x2,25, dyellow, 'filled');
            end
            if( k == 2)
                scatter(ci_x1,ci_x2,25, dpink, 'filled');
            end
            if( k == 3)
                scatter(ci_x1,ci_x2,25, dblue, 'filled');
            end
            if( k == 4)
                scatter(ci_x1,ci_x2,25, dgreen, 'filled');
            end
            k = k + 1;
            hold on;
        end
         if (no_of_classes == 2)
            if( k == 1)
                scatter(ci_x1,ci_x2,25, dyellow, 'filled');
            end
            if( k == 2)
                scatter(ci_x1,ci_x2,25, dblue, 'filled');
            end
            k = k + 1;
            hold on;
        end
    end
    xlabel('x1')
    ylabel('x2')
    title(['Plot of dataset, points beloning to' num2str(no_of_classes) 'classes']);
    if (no_of_classes == 4)
        legend({'class1', 'class2' , 'class3', 'class4' },'Location','north');
    end
     if (no_of_classes == 2)
          legend({'class1', 'class2'},'Location','north');
     end
    hold off;
end

%D has all data mixed for all classes in x1,x2, orgi_label, pred_label
%format
function plot_decision_region(D, no_of_classes)
    lyellow = [0.9961, 0.9961, 0.5859];
    lpink = [0.9961, 0.6445, 0.9141];
    lblue = [0.6758, 0.8555, 0.9961];
    lgreen = [0.2578, 0.9531, 0.7930];
    n = size(D,1);
    dim = size(D,2)-2;
    
    figure();
    x1 = D(:,1);
    x2 = D(:,2);
    if (no_of_classes == 4)
        scatter(D(D(:,dim+2) == 1, 1), D(D(:,dim+2) == 1, 2), 25, lyellow, 'filled');
        hold on;
        scatter(D(D(:,dim+2) == 2, 1), D(D(:,dim+2) == 2, 2), 25, lpink,'filled');
        hold on;
        scatter(D(D(:,dim+2) == 3, 1), D(D(:,dim+2) == 3, 2), 25, lblue,'filled');
        hold on;
        scatter(D(D(:,dim+2) == 4, 1), D(D(:,dim+2) == 4, 2), 25, lgreen,'filled');
        hold on;
    end
    if (no_of_classes == 2)
        scatter(D(D(:,dim+2) == 1, 1), D(D(:,dim+2) == 1, 2), 25, lyellow, 'filled');
        hold on;
        scatter(D(D(:,dim+2) == 2, 1), D(D(:,dim+2) == 2, 2), 25, lblue,'filled');
        hold on;
    end
    
%     color_map = ['r', 'y','g', 'b'];
%     figure();
%     scatter(D(:,1), D(:,2), 10, D(:,dim+2));
% %     for i = 1:n
% %         predicted_class = D(i,dim+2);
% %         scatter(D(i,1), D(i,2), color_map(predicted_class), 'filled');
% %         hold on;
% %         
% %     end
%     
%     %hold off;
end

function params = train_model(D, no_of_classes, dim, val_set)
   
    class_ctr = 1;
    params = {};
    k = 1;
    for index = 1:dim:(dim*no_of_classes-1)
        disp(['Class' num2str(class_ctr) ''])
        class_data = D(:, index:(index+dim-1));
        val_set_class_data = val_set(:, index:(index+dim-1));
        params_single_class = train_model_single_class(class_data, val_set_class_data, no_of_classes);
        %params_single_class{:}
        params{k} = params_single_class;   
        class_ctr = class_ctr+1;
        k = k+1;
    end
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
    if (no_of_classes == 4)
        Q_max = 3;  %TODO: make this n_data/10, now changing for test purpose.
    end
    if (no_of_classes == 2)
        Q_max = 10;  %TODO: make this n_data/10, now changing for test purpose.
    end
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
 
 end
 
 %check def of covariance with surity for N, N-1
 function c = calc_covariance(A, means)
    n = size(A,1);
    one_vector(1:n) = 1;
    %duplicate mean vec
    mean_mat = means(one_vector, :);
    A_sub_mean = A - mean_mat;
    c = (A_sub_mean'*A_sub_mean)/n;
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
    for i = 1:n
        resp_term_single_sample = [];
        one_sample = x(i,:);
        for q=1:Q
            wq = w(q);
            mu_q = mu(q,:);
            cq = c(1:dim, dim*(q-1)+1:dim*(q-1)+dim);
            resp_term_single_sample = [resp_term_single_sample wq*gaussian(one_sample',mu_q', cq)];
        end
        resp_term_single_sample = resp_term_single_sample./sum(resp_term_single_sample,2);
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
    
