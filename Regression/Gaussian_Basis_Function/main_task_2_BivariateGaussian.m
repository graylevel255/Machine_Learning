%Author: Aakriti Budhraja
N = 2000; %Sample size
k = 35; %Number of clusters to be formed
variate = 2; %Number of variables , say p-variate data(eg 2 means Bivariate data)

Loc = '../data_assign1_group28/train.txt';
A = importdata(Loc,' ');
X = A(:,1:variate);
t = A(:,variate+1);

%Plot and see datapoints for t vs (x1,x2).
x1 = X(:,1);
x2 = X(:,2);
figure;
quad1 = subplot(2,2,1);
tri = delaunay(x1,x2);
trisurf(tri,x1,x2,t);
title(quad1 , 'System output t vs (x1,x2)');
text(0.5,0.9,0.9,['k =' num2str(k) ''],'Units','normalized','FontWeight','bold');
xlabel(quad1,'x1');
ylabel(quad1,'x2');
zlabel(quad1,'t');
                
%Apply kMeans to form 'k' clusters
[groups,means] = kmeans(X,k);

%Global Variance (sigma_square)
C = cov(X);
sigma_square = trace(C)/variate;

%Construct the Gaussian matrix 'g' for training data
g = zeros(N,k+variate);
x_minus_mu = zeros(variate,1);
mu = zeros(variate,1);
for i = 1:N
    g(i,1) = x1(i);
    g(i,2) = x2(i);
     for j = 1:k
        for z = 1:variate
            mu(z) = means(j,z);
            x_minus_mu(z) = g(i,z)-mu(z);
        end
    norm_value = norm(x_minus_mu);
    norm_square = norm_value^2;
    value = -0.5*norm_square/sigma_square;
    gaussianvalue = exp(value);
    g(i,j+variate) = gaussianvalue;
    end
end

%Creating phi matrix for training data
phi = zeros(N,k);
for i = 1:N
    for j = 1:k
    phi(i,j) = g(i,j+variate);
    end
end

%Calculation of 'wstar' and output 'y'
phi_t = phi';
temp1 = mtimes(phi_t,phi);
temp2 = inv(temp1);    
temp3 = mtimes(temp2,phi_t);
wstar = mtimes(temp3,t);
y = mtimes(phi,wstar);
trainErms = ((sum((t-y).^2))/N).^0.5

quad2 = subplot(2,2,2);
tri = delaunay(x1,x2);
trisurf(tri,x1,x2,y);
title(quad2,'Model output y vs (x1,x2) without regularization');
text(0.5,0.9,0.9,['k =' num2str(k) ''],'Units','normalized','FontWeight','bold');
xlabel(quad2,'x1');
ylabel(quad2,'x2');
zlabel(quad2,'y');

%Adding Quadratic Regularization for training data
lambda = exp(-17);
id = eye(k);
lambdaI = lambda*id;
t2 = inv(temp1+lambdaI);
t3 = mtimes(t2,phi_t);
wReg1 = mtimes(t3,t);
yReg1 = mtimes(phi,wReg1);
trainErmsWithQuadraticRegularization = ((sum((t-yReg1).^2))/N).^0.5

quad3 = subplot(2,2,3);
tri = delaunay(x1,x2);
trisurf(tri,x1,x2,yReg1);
title(quad3,'Model output y vs (x1,x2) with Quadratic Regularization({\lambda} = \ite^{-17})');
text(0.5,0.9,0.9,['k =' num2str(k) ''],'Units','normalized','FontWeight','bold');
xlabel(quad3,'x1');
ylabel(quad3,'x2');
zlabel(quad3,'y');

%Adding Tikhonov Regularization for training data
mui = zeros(2,1);
muj = zeros(2,1);
lambda = exp(-17);
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
l2 = inv(temp1+lambda_phi_tilda);
l3 = mtimes(l2,phi_t);
wReg2 = mtimes(l3,t);
yReg2 = mtimes(phi,wReg2);
trainErmsWithTikhonovRegularization = ((sum((t-yReg2).^2))/N).^0.5
r = mtimes(wReg2',phi_tilda);
r = mtimes(r,wReg2);
TrainingSetRoughness = r*0.5

quad4 = subplot(2,2,4);
tri = delaunay(x1,x2);
trisurf(tri,x1,x2,yReg2);
title(quad4,'Model output y vs (x1,x2) with Tikhonov Regularization({\lambda} = \ite^{-17})');
text(0.5,0.9,0.9,['k =' num2str(k) ''],'Units','normalized','FontWeight','bold');
xlabel(quad4,'x1');
ylabel(quad4,'x2');
zlabel(quad4,'y');

%Running model on test data
Nt = 200;
testLoc = '../data_assign1_group28/test.txt';
At = importdata(testLoc,' ');
Xt = At(:,1:variate);
tt = At(:,variate+1);
x1t = Xt(:,1);
x2t = Xt(:,2);

%Construct the Gaussian matrix 'gt' for test data
gt = zeros(Nt,k+variate);
testx_minus_mu = zeros(variate,1);
for i = 1:Nt
    gt(i,1) = x1t(i);
    gt(i,2) = x2t(i);
     for j = 1:k
        for z = 1:variate
            mu(z) = means(j,z);
            testx_minus_mu(z) = gt(i,z)-mu(z);
        end
    norm_value = norm(testx_minus_mu);
    norm_square = norm_value^2;
    value = -0.5*norm_square/sigma_square;
    gaussianvalue = exp(value);
    gt(i,j+variate) = gaussianvalue;
    end
end

%Creating phi matrix for test data
phitest = zeros(Nt,k);
for i = 1:Nt
    for j = 1:k
    phitest(i,j) = gt(i,j+variate);
    end
end

ytest = mtimes(phitest,wstar);
testErms = ((sum((tt-ytest).^2))/Nt).^0.5

%Calculating ytest using Quadratic Regularization for Test data
ytestReg1 = mtimes(phitest,wReg1);
testErmsWithQuadraticRegularization = ((sum((tt-ytestReg1).^2))/Nt).^0.5

%Calculating ytest using Tikhonov Regularization for Test data
ytestReg2 = mtimes(phitest,wReg2);
testErmsWithTikhonovRegularization = ((sum((tt-ytestReg2).^2))/Nt).^0.5

%Running code on validation data
Nv = 300;
valLoc = '../data_assign1_group28/val.txt';
Av = importdata(valLoc,' ');
Xv = Av(:,1:variate);
tv = Av(:,variate+1);
x1v = Xv(:,1);
x2v = Xv(:,2);

%Construct the Gaussian matrix 'gv' for validation data
gv = zeros(Nv,k+variate);
valx_minus_mu = zeros(variate,1);
for i = 1:Nv
    gv(i,1) = x1v(i);
    gv(i,2) = x2v(i);
     for j = 1:k
        for z = 1:variate
            mu(z) = means(j,z);
            valx_minus_mu(z) = gv(i,z)-mu(z);
        end
    norm_value = norm(valx_minus_mu);
    norm_square = norm_value^2;
    value = -0.5*norm_square/sigma_square;
    gaussianvalue = exp(value);
    gv(i,j+variate) = gaussianvalue;
    end
end

%Creating phi matrix for validation data
phival = zeros(Nv,k);
for i = 1:Nv
    for j = 1:k
    phival(i,j) = gv(i,j+variate);
    end
end

yval = mtimes(phival,wstar);
valErms = ((sum((tv-yval).^2))/Nv).^0.5

%Calculating yval using Quadratic Regularization for Validation data
yvalReg1 = mtimes(phival,wReg1);
ValErmsWithQuadraticRegularization = ((sum((tv-yvalReg1).^2))/Nv).^0.5

%Calculating yval using Tikhonov Regularization for Validation data
yvalReg2 = mtimes(phival,wReg2);
valErmsWithTikhonovRegularization = ((sum((tv-yvalReg2).^2))/Nv).^0.5

figure;
quad1 = subplot(3,2,1);
scatter(t,y);
title(quad1 , 'Training data : y vs t(without regularization)');
xlabel(quad1,'t');
ylabel(quad1,'y');

quad2 = subplot(3,2,2);
scatter(tt,ytest);
title(quad2 , 'Test data : y vs t(without regularization)');
xlabel(quad2,'t');
ylabel(quad2,'y');

quad3 = subplot(3,2,3);
scatter(t,yReg1);
title(quad3 , 'Training data : y vs t(with Quadratic Regularization)');
xlabel(quad3,'t');
ylabel(quad3,'y');

quad4 = subplot(3,2,4);
scatter(tt,ytestReg1);
title(quad4 , 'Test data : y vs t(with Quadratic Regularization)');
xlabel(quad4,'t');
ylabel(quad4,'y');

quad5 = subplot(3,2,5);
scatter(t,yReg2);
title(quad5 , 'Training data : y vs t(with Tikhonov Regularization)');
xlabel(quad5,'t');
ylabel(quad5,'y');

quad6 = subplot(3,2,6);
scatter(tt,ytestReg2);
title(quad6 , 'Test data : y vs t(with Tikhonov Regularization)');
xlabel(quad6,'t');
ylabel(quad6,'y');


