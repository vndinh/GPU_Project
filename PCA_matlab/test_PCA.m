function [project_test_img] = test_PCA(test_matrix,k_eig_vec,m)

%% Arguments %%
% test_matrix: test image matrix with dimension of N*d 
% N: the number of test images
% d: the dimension of one images
% k_eig_vec: k biggest eigen vectors from training matrix
% m: mean from training matrix

%% Code %%
  N = size(test_matrix,1);

  % Normalize test images
  X = test_matrix - repmat(m,N,1);

  % Projection of the test images in the subspace created by k biggest eigenvectors
  project_test_img = X * k_eig_vec;

end
