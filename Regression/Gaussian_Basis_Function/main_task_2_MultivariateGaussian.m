%Author: Aakriti Budhraja
N = 1030; %Sample size
Ntrain = 721; %0.7*1030 ; 1-721
Nval = 103; %0.1*1030   ; 722-824
Ntest = 206 ; %0.2*1030 ;   825-1030
k = 30; %Number of clusters to be formed
variate = 8; %Number of variables , say p-variate data(eg 2 means Bivariate data)

loc = '../data_assign1_group28/concrete-data.txt';
A = importdata(loc,' ');
A = A(randperm(size(A,1)),:);

X = A(:,1:variate);
t = A(:,variate+1);

Xtrain = X(1:Ntrain,:);
Xval = X(Ntrain+1:Ntrain+Nval,:);
Xtest = X(Ntrain+Nval+1:1030,:);

ttrain = t(1:Ntrain);
tval = t(Ntrain+1:Ntrain+Nval);
ttest = t(Ntrain+Nval+1:1030);

x1train = Xtrain(:,1);  x1test = Xtest(:,1);    x1val = Xval(:,1);
x2train = Xtrain(:,2);  x2test = Xtest(:,2);    x2val = Xval(:,2);
x3train = Xtrain(:,3);  x3test = Xtest(:,3);    x3val = Xval(:,3);
x4train = Xtrain(:,4);  x4test = Xtest(:,4);    x4val = Xval(:,4);
x5train = Xtrain(:,5);  x5test = Xtest(:,5);    x5val = Xval(:,5);
x6train = Xtrain(:,6);  x6test = Xtest(:,6);    x6val = Xval(:,6);
x7train = Xtrain(:,7);  x7test = Xtest(:,7);    x7val = Xval(:,7);
x8train = Xtrain(:,8);  x8test = Xtest(:,8);    x8val = Xval(:,8);

%Apply kMeans to form 'k' clusters
[groups,means] = kmeans(Xtrain,k);

%Global Variance (sigma_square)
C = cov(X);
sigma_square = trace(C)/variate;

% Training dataset
gtrain = zeros(Ntrain,k+variate);
x_minus_mutrain = zeros(variate,1);
mu = zeros(variate,1);  % mu is common to all datasets
for i = 1:Ntrain
    gtrain(i,1) = x1train(i);
    gtrain(i,2) = x2train(i);
    gtrain(i,3) = x3train(i);
    gtrain(i,4) = x4train(i);
    gtrain(i,5) = x5train(i);
    gtrain(i,6) = x6train(i);
    gtrain(i,7) = x7train(i);
    gtrain(i,8) = x8train(i);
    for j = 1:k
        for z = 1:variate
            mu(z) = means(j,z);
            x_minus_mutrain(z) = gtrain(i,z)-mu(z);
        end
    norm_value = norm(x_minus_mutrain);
    norm_square = norm_value^2;
    value = -0.5*norm_square/sigma_square;
    gaussianvalue = exp(value);
    gtrain(i,j+variate) = gaussianvalue;
    end
end

%Creating phitrain matrix
phitrain = zeros(Ntrain,k);
for i = 1:Ntrain
    for j = 1:k
    phitrain(i,j) = gtrain(i,j+variate);
    end
end

%Calculation of 'wstar' and output 'y'
phi_t = phitrain';
temp1 = mtimes(phi_t,phitrain);
temp2 = inv(temp1);    
temp3 = mtimes(temp2,phi_t);
wstar = mtimes(temp3,ttrain);
ytrain = mtimes(phitrain,wstar);

ermstrain = ((sum((ttrain-ytrain).^2))/Ntrain).^0.5

%Adding Quadratic Regularization for training data
lambda = exp(-18);
id = eye(k);
lambdaI = lambda*id;
phi_t = phitrain';
temp1 = mtimes(phi_t,phitrain);
t2 = inv(temp1+lambdaI);
t3 = mtimes(t2,phi_t);
wReg1 = mtimes(t3,ttrain);
ytrainReg1 = mtimes(phitrain,wReg1);
trainErmsWithQuadraticRegularization = ((sum((ttrain-ytrainReg1).^2))/Ntrain).^0.5

%Adding Tikhonov Regularization for training data
mui = zeros(variate,1);
muj = zeros(variate,1);
lambda = 0.5;
phi_tilda = zeros(k,k);
for i = 1:k
    for j = 1:k
        for z = 1:variate
            mui(z) = means(i,z);
            muj(z) = means(j,z);
        end
        n = (norm(mui-muj))^2;
        n = -0.5*n/sigma_square;
        phi_tilda(i,j) = exp(n);
    end
end
lambda_phi_tilda = lambda*phi_tilda;
phi_t = phitrain';
temp1 = mtimes(phi_t,phitrain);
l2 = inv(temp1+lambda_phi_tilda);
l3 = mtimes(l2,phi_t);
wReg2 = mtimes(l3,ttrain);
ytrainReg2 = mtimes(phitrain,wReg2);
trainErmsWithTikhonovRegularization = ((sum((ttrain-ytrainReg2).^2))/Ntrain).^0.5
r = mtimes(wReg2',phi_tilda);
r = mtimes(r,wReg2);
TrainingSetRoughness = r*0.5

%Running model on test data
gtest = zeros(Ntest,k+variate);
x_minus_mutest = zeros(variate,1);
for i = 1:Ntest
    gtest(i,1) = x1test(i);
    gtest(i,2) = x2test(i);
    gtest(i,3) = x3test(i);
    gtest(i,4) = x4test(i);
    gtest(i,5) = x5test(i);
    gtest(i,6) = x6test(i);
    gtest(i,7) = x7test(i);
    gtest(i,8) = x8test(i);
    for j = 1:k
        for z = 1:variate
            mu(z) = means(j,z);
            x_minus_mutest(z) = gtest(i,z)-mu(z);
        end
    norm_value = norm(x_minus_mutest);
    norm_square = norm_value^2;
    value = -0.5*norm_square/sigma_square;
    gaussianvalue = exp(value);
    gtest(i,j+variate) = gaussianvalue;
    end
end

%Creating phitest matrix
phitest = zeros(Ntest,k);
for i = 1:Ntest
    for j = 1:k
    phitest(i,j) = gtest(i,j+variate);
    end
end

ytest = mtimes(phitest,wstar);
testErms = ((sum((ttest-ytest).^2))/Ntest).^0.5

%Calculating ytest using Quadratic Regularization for Test data
ytestReg1 = mtimes(phitest,wReg1);
testErmsWithQuadraticRegularization = ((sum((ttest-ytestReg1).^2))/Ntest).^0.5

%Calculating ytest using Tikhonov Regularization for Test data
ytestReg2 = mtimes(phitest,wReg2);
testErmsWithTikhonovRegularization = ((sum((ttest-ytestReg2).^2))/Ntest).^0.5

%Running model on Validation data
gval = zeros(Nval,k+variate);
x_minus_muval = zeros(variate,1);
for i = 1:Nval
    gval(i,1) = x1val(i);
    gval(i,2) = x2val(i);
    gval(i,3) = x3val(i);
    gval(i,4) = x4val(i);
    gval(i,5) = x5val(i);
    gval(i,6) = x6val(i);
    gval(i,7) = x7val(i);
    gval(i,8) = x8val(i);
    for j = 1:k
        for z = 1:variate
            mu(z) = means(j,z);
            x_minus_muval(z) = gval(i,z)-mu(z);
        end
    norm_value = norm(x_minus_muval);
    norm_square = norm_value^2;
    value = -0.5*norm_square/sigma_square;
    gaussianvalue = exp(value);
    gval(i,j+variate) = gaussianvalue;
    end
end

%Creating phival matrix
phival = zeros(Nval,k);
for i = 1:Nval
    for j = 1:k
    phival(i,j) = gval(i,j+variate);
    end
end

yval = mtimes(phival,wstar);
valErms = ((sum((tval-yval).^2))/Nval).^0.5

%Calculating yval using Quadratic Regularization for Validation data
yvalReg1 = mtimes(phival,wReg1);
valErmsWithQuadraticRegularization = ((sum((tval-yvalReg1).^2))/Nval).^0.5

%Calculating yval using Tikhonov Regularization for Validation data
yvalReg2 = mtimes(phival,wReg2);
valErmsWithTikhonovRegularization = ((sum((tval-yvalReg2).^2))/Nval).^0.5

figure;
quad1 = subplot(3,2,1);
scatter(ttrain,ytrain);
title(quad1 , 'Training data : y vs t(without regularization)');
xlabel(quad1,'t');
ylabel(quad1,'y');

quad2 = subplot(3,2,2);
scatter(ttest,ytest);
title(quad2 , 'Test data : y vs t(without regularization)');
xlabel(quad2,'t');
ylabel(quad2,'y');

quad3 = subplot(3,2,3);
scatter(ttrain,ytrainReg1);
title(quad3 , 'Training data : y vs t(with Quadratic Regularization)');
xlabel(quad3,'t');
ylabel(quad3,'y');

quad4 = subplot(3,2,4);
scatter(ttest,ytestReg1);
title(quad4 , 'Test data : y vs t(with Quadratic Regularization)');
xlabel(quad4,'t');
ylabel(quad4,'y');

quad5 = subplot(3,2,5);
scatter(ttrain,ytrainReg2);
title(quad5 , 'Training data : y vs t(with Tikhonov Regularization)');
xlabel(quad5,'t');
ylabel(quad5,'y');

quad6 = subplot(3,2,6);
scatter(ttest,ytestReg2);
title(quad6 , 'Test data : y vs t(with Tikhonov Regularization)');
xlabel(quad6,'t');
ylabel(quad6,'y');