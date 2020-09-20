% Read data of 3 classes

% Class 1 - highway (N = 260)
% Class 2 - mountain (N = 374)
% Class 3 - tall building (N = 356)

nc = 3;
var = 23;   % dimension of features

parent_path = '../data/image_dataset/Features';
file1_loc = '../data/image_dataset/Features/highway';
file2_loc = '../data/image_dataset/Features/mountain';
file3_loc = '../data/image_dataset/Features/tallbuilding';

D = dir(fullfile(file1_loc, '*.jpg_color_edh_entropy'));
A = cell(size(D));
class1_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file1_loc,D(i).name));
      R = reshape(A{i},1,[]);
      class1_data = [class1_data ; R];
end

D = dir(fullfile(file2_loc, '*.jpg_color_edh_entropy'));
A = cell(size(D));
class2_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file2_loc,D(i).name)); 
      R = reshape(A{i},1,[]);
      class2_data = [class2_data ; R];
end

D = dir(fullfile(file3_loc, '*.jpg_color_edh_entropy'));
A = cell(size(D));
class3_data = [];
for i=1:length(D)
      A{i} = importdata(fullfile(file3_loc,D(i).name));
      R = reshape(A{i},1,[]);
      class3_data = [class3_data ; R];
end

Data = [class1_data ; class2_data ; class3_data];

% Create target output vector

t = zeros(numel(Data(:,1)),nc);
n1 = numel(class1_data(:,1))
n2 = numel(class2_data(:,1))
n3 = numel(class3_data(:,1))

for i=1:n1
    t(i,1) = 1;
end

for i=1:n2
    t(i+n1,2) = 1;
end

tmp1 = n1+n2;
for i=1:n3
    t(i+tmp1,3) = 1;
end

X = Data';
T = t';

% Bring data in that format
d_new = 828;
IP = cell(d_new,1);

for i=1:numel(X(:,1))
      IP{i,1} = X(i,:);
end 
 
net = patternnet([20,22]);
net.numinputs = d_new;
ip1 = ones(1,d_new);
ip2 = zeros(1,d_new);
ip3 = zeros(1,d_new);
% ip = [ip1;ip2;ip3]
net.inputConnect = [ip1;ip2;ip3];
net = configure(net,IP,T);
net = train(net,IP,T);
net.trainParam.epochs = 1000;
% net.trainParam.lr = 0.01;


Y = net(IP); 
error = gsubtract(T,Y);
performance = perform(net,T,Y) 
tind = vec2ind(T); 
yind = vec2ind(Y); 
net.IW{1,1}; 
net.b{1}; 

% net.LW {1};
% view(net)

