function [train_matrix,test_matrix] = createDataset()
%% Arguments %%
% N_train is the number of training dataset
% N_test is the number of training dataset
% d is the dimension of training and test dataset
% train_matrix is the training data matrix
% test_matrix is the test data matrix

%% Codes %%
  addpath ./training_img
  addpath ./test_img

  N_train = 163;
  d = 64*64;
  train_matrix = zeros(N_train, d);   
  for i = 1:N_train
    fname = sprintf('training_img/%d.jpg',i);
    img = double(imread(fname));
    train_matrix(i,:) = (img(:))';
  end

  N_test = 50;
  test_matrix = zeros(N_test, d);   
  for i = 1:N_test
    fname = sprintf('test_img/%d.jpg',i);
    img = double(imread(fname));
    test_matrix(i,:) = (img(:))';
  end
end
