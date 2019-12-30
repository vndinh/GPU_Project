clear; clc; close all;
k = 50; % The number of PCs

%% Codes %%
[train_matrix,test_matrix] = createDataset();
[project_train_img, k_eig_vec, m] = train_PCA(train_matrix,k);
[project_test_img] = test_PCA(test_matrix,k_eig_vec,m);
[id] = identify(project_train_img,project_test_img);