function [recognized_img] = identify(project_train_img,project_test_img)

%% Arguments %%
% project_train_img: training matrix represented by k PCs
% project_test_img: test matrix represented by k PCs

%% Code %%
  N_train = size(project_train_img,1);
  N_test = size(project_test_img,1);
  recognized_img = zeros(N_test,1);
  for i = 1:N_test
    % Calculate difference between training images and test images
    D = project_train_img - repmat(project_test_img(i,:),N_train,1);
    
    % Calculate Euclid distances
    distance = zeros(N_train,1);
    for j = 1:N_train
        distance(j) = norm(D(j,:));
    end
    
    % Choose the smallest distance and
    % index of the most similar training images
    [~, idx] = min(distance);
    recognized_img(i) = idx;
  end
end
