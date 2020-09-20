%Author: Sadbhavana Babar
%Last version modified on 19.09.18

loc = '../data_assign1_group28/';
% Create the data matrix
D = dir(fullfile(loc,'*.pgm'));
A = cell(size(D));
for i = 1:numel(D)
      I = imread(fullfile(loc,D(i).name));
      A{i} = I(:); % converts image into a column matrix 4096X1
end


%A{1} is the first image in vector form, A{2} second image....so on

n = numel(A);
d = 4096;
set = zeros(d,n); %2d matrix of 4096 rows and 10 cols

for i = 1:numel(A) %10
set(:,i) = A{i};   %storing image vector in a column
end

%Calculating mu vector
   row_sum = 0;
   mu = zeros(d,1);
   for i = 1:d
       for j = 1:n
           row_sum = 0;
           row_sum = row_sum + set(i,j);
       end
       mu(i) = row_sum/n;
   end

 C = zeros(d,n); %C is 4096X10 and has (X-mu).
 for i = 1:n
 C(:,i) = set(:,i) - mu;
 end

 img = set(:,2); % Original Image
 
% Testing for the following 5 images from the database
 
%  img = set(:,2);
%  img = set(:,75);
%  img = set(:,53);
%  img = set(:,96);
%  img = set(:,143);

 org_img = uint8(reshape(img,[],64));
 subplot(3,3,1);
 imshow(org_img);
 title('Original Image');
 
 % Constructing the covariance matrix
 covar_mat = C*C'/n;
 
  i = 1;  
 % Image reconstruction
 for l = [1, 10, 20, 40, 80, 160, 320, 640]
     % finding l significant Eigen vectors
     [Q,D] = eigs(covar_mat,l); 
        rec_img = zeros(d,1);

     for k = 1:l
         L = Q(:,k);
         rec_img = rec_img + mtimes(img',L)*L;
     end

      S = uint8(reshape(rec_img,[],64));
      subplot(3,3, i+1);
      imshow(S);
      title(['Reconstructed image for l = ' num2str(l) ''],'FontSize', 8);     
      i = i+1;
 end
 
 %Calculating the error of reconstructed image for l = 640
 erms = sqrt(sum((img - rec_img).^2))

