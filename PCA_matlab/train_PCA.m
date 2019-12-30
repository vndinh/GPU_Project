function [project_train_img, k_eig_vec, m] = train_PCA(train_matrix,k)

%% Arguments %%
% train_matrix: training image matrix with dimension of N*d 
% N: the number of training images
% d: is the dimension of one images
% k: number of PCs

%% Code %%
  [N,d] = size(train_matrix);

  % Mean of all images
  m = sum(train_matrix) / N;

  % Coveriance matrix
  X = train_matrix - repmat(m,N,1);
  %S = zeros(d,d);
%   for i = 1:N
%    S = S + X(i,:)' * X(i,:); 
%   end
  S = X' * X;

  % Calculate eigenvectors of the coveriance matrix
  [eig_vec,~] = eig(S);

  % Choose k eigenvectors corresponding k biggest eigenvalues
  k_eig_vec = eig_vec(:,4096:-1:(4096-k+1));

  % Projection training images in the subspace created by k chosen eigenvectors
  project_train_img = X * k_eig_vec;
end
