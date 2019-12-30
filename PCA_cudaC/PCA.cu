#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void gpuMatSub(float* A, float* B, float* C, int Nrows, int Ncols) {
  int tid, tx, ty;
  tx = threadIdx.x + blockIdx.x * blockDim.x;
  ty = threadIdx.y + blockIdx.y * blockDim.y;
  if ((tx < Ncols) && (ty < Nrows)) {
    tid = Ncols * ty + tx;
    C[tid] = A[tid] - B[tid];
  }
}

__global__ void gpuMatMul(float *A, float *B, float *C, int m, int n, int p) { 
  int row = blockIdx.y * blockDim.y + threadIdx.y; 
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  float S = 0;
  if (col < p && row < m) {
    for(int i = 0; i < n; i++) {
      S += A[row * n + i] * B[i * p + col];
    }
    C[row * p + col] = S;
  }
}

__global__ void gpuMatTranspose(float *A, float *B, int Nrows, int Ncols)
{
  unsigned int tx, ty, pos, trans_pos;
  tx = blockIdx.x * blockDim.x + threadIdx.x;
  ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx < Ncols && ty < Nrows) {
    pos = ty * Ncols + tx;
    trans_pos = tx * Nrows + ty;
    B[trans_pos] = A[pos];
  }
}

__global__ void gpuMatMulScalar(float *A, float *B, int Nrows, int Ncols, float lambda) {
  int tid, tx, ty;
  tx = threadIdx.x + blockIdx.x * blockDim.x;
  ty = threadIdx.y + blockIdx.y * blockDim.y;
  if ((tx < Ncols) && (ty < Nrows)) {
    tid = Ncols * ty + tx;
    B[tid] = A[tid] * lambda;
  }
}

__global__ void gpuNorm2(float *A, float *norm, int N) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int gid = tid + blockIdx.x * blockDim.x;

  if (gid < N) sdata[tid] = A[gid];
  else sdata[tid] = 0;
  __syncthreads();

  sdata[tid] = sdata[tid] * sdata[tid];
  __syncthreads();

  for (unsigned int stride = blockDim.x/2; stride > 0; stride = stride >> 1) {
    if (tid < stride) sdata[tid] = sdata[tid] + sdata[tid + stride];
    __syncthreads();
  }

  if (tid == 0) atomicAdd(norm, sdata[0]);
}

__global__ void gpuMeanVec(int *trainMat, int Ntrain, int dimension, float *mean) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float tmp = 0;
  int DestIndex;
  for (int i = 0; i < Ntrain; i++) {
    DestIndex = i * dimension + tid;
    tmp = tmp + trainMat[DestIndex];
  }
  mean[tid] = tmp / Ntrain;
}

__global__ void gpuSubMean(int *trainMat, float *mean, int Ntrain, int dimension, float *X) {
  int tid, tx, ty;
  tx = threadIdx.x + blockIdx.x * blockDim.x;
  ty = threadIdx.y + blockIdx.y * blockDim.y;
  if ((tx < dimension) && (ty < Ntrain)) {
    tid = dimension * ty + tx;
    X[tid] = trainMat[tid] - mean[tx];
  }
}

__global__ void gpuCovMat(float *A, float *B, int Nrows, int Ncols, float lambda) {
  int tid, tx, ty;
  tx = threadIdx.x + blockIdx.x * blockDim.x;
  ty = threadIdx.y + blockIdx.y * blockDim.y;
  if ((tx < Ncols) && (ty < Nrows)) {
    tid = Ncols * ty + tx;
    B[tid] = A[tid] / lambda;
  }
}

__global__ void gpuNormalize(float *A, float *norm, float *B, int N) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int gid = tid + blockIdx.x * blockDim.x;

  if (tid == 0) sdata[0] = norm[0];
  __syncthreads();

  if (gid < N) B[gid] = A[gid] / sdata[0];
}

__global__ void gpuEigVal(float *A, float *B, float *eigVal, int N) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int gid = tid + blockIdx.x * blockDim.x;

  if (gid < N) sdata[tid] = A[gid] * B[gid];
  else sdata[tid] = 0;
  __syncthreads();

  for (unsigned int stride = blockDim.x/2; stride > 0; stride = stride >> 1) {
    if (tid < stride) sdata[tid] = sdata[tid] + sdata[tid+stride];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(eigVal, sdata[0]);
}

float cpuNorm2(float *A, int Nrows, int Ncols) {
  float tmp = 0;
  float Norm;
  int DestIndex;
  for (int col = 0; col < Ncols; col++) {
    for (int row = 0; row < Nrows; row++) {
      DestIndex = row * Ncols + col;
      tmp = tmp + A[DestIndex] * A[DestIndex];
    }
  }
  Norm = sqrt(tmp);
  return Norm;
}

void cpuMatSub(float* A, float* B, int Nrows, int Ncols, float* C) {
  int col = 0;
  int row = 0;
  int DestIndex = 0;
  for (col = 0; col < Ncols; col++) {
    for (row = 0; row < Nrows; row++) {
      DestIndex = row * Ncols + col;
      C[DestIndex] = A[DestIndex] - B[DestIndex];
    }
  }
}

void identify(float *project_train_img, float *project_test_img, int Ntrain, int Ntest, int dimension, int *recognized_img) {
  int DestIndex;
  float *test;
  float *train;
  float *D;
  float distance = 0;
  float min;

  int buffer = dimension * sizeof(float);
  test = (float*)malloc(buffer);
  train = (float*)malloc(buffer);
  D = (float*)malloc(buffer);

  for (int i = 0; i < Ntest; i++) {
    min = 100000;
    for (int j = 0; j < dimension; j++) {
      DestIndex = i * dimension + j;
      test[j] = project_test_img[DestIndex];
    }

    for (int m = 0; m < Ntrain; m++) {
      for (int n = 0; n < dimension; n++) {
        DestIndex = m * dimension + n;
        train[n] = project_train_img[DestIndex];
      }
      cpuMatSub(train, test, 1, dimension, D);
      distance = cpuNorm2(D, 1, dimension);
      if (distance < min) {
        min = distance;
        recognized_img[i] = m + 1;
      }
    }
  }

  free(test);
  free(train);
  free(D);
}

int main() {
  FILE *fp_train;
  FILE *fp_test;
  FILE *fp_id;
  int *trainMat;
  int *testMat;
  int Ntrain = 163; // Number of training images
  int Ntest = 50;   // Number of test images
  int imgSize = 64; // Size of image
  int k = 50;       // Number of the principal components

  int dimension = imgSize * imgSize;  // The original dimension 4096
  int DestIndex;

  int trainMatBuffer = Ntrain * dimension * sizeof(int);
  int testMatBuffer = Ntest * dimension * sizeof(int);

  // Read training set
  trainMat = (int*)malloc(trainMatBuffer);
  fp_train = fopen("train_matrix.txt", "r");
  if (fp_train == NULL) {
    printf("Error open file\n");
    return 1;
  }
  for (int i = 0; i < Ntrain; i++) {
    for (int j = 0; j < dimension; j++) {
      DestIndex = i * dimension + j;
      fscanf(fp_train, "%d", &trainMat[DestIndex]);
    }
  }
  fclose(fp_train);

  // Read test set
  testMat = (int*)malloc(testMatBuffer);
  fp_test = fopen("test_matrix.txt", "r");
  if (fp_test == NULL) {
    printf("Error open file\n");
    return 1;
  }
  for (int i = 0; i < Ntest; i++) {
    for (int j = 0; j < dimension; j++) {
      DestIndex = i * dimension + j;
      fscanf(fp_test, "%d", &testMat[DestIndex]);
    }
  }
  fclose(fp_test);

  int *d_trainMat;
  int *d_testMat;
  float *d_mean;

  int imgBuffer = dimension * sizeof(float);

  cudaMalloc((void**)&d_trainMat, trainMatBuffer);
  cudaMalloc((void**)&d_testMat, testMatBuffer);
  cudaMalloc((void**)&d_mean, imgBuffer);

  // Copy the training and test matrices from Host to Device
  cudaMemcpy(d_trainMat, trainMat, trainMatBuffer, cudaMemcpyHostToDevice);
  cudaMemcpy(d_testMat, testMat, testMatBuffer, cudaMemcpyHostToDevice);

  // Calculate the mean vector of the training matrix
  gpuMeanVec<<<32, 128>>>(d_trainMat, Ntrain, dimension, d_mean);

  // Compute the difference between each image vector and the mean vector
  dim3 block1(128, 1);
  dim3 grid1(dimension/128, Ntrain);
  float *d_X;
  int Xbuffer = Ntrain * dimension * sizeof(float);
  cudaMalloc((void**)&d_X, Xbuffer);
  gpuSubMean<<<grid1, block1>>>(d_trainMat, d_mean, Ntrain, dimension, d_X);

  // Determine the covariance matrix
  int covMatBuffer = dimension * dimension * sizeof(float);
  float *d_Xt, *d_covMat;
  cudaMalloc((void**)&d_Xt, Xbuffer);
  cudaMalloc((void**)&d_covMat, covMatBuffer);
  gpuMatTranspose<<<grid1, block1>>>(d_X, d_Xt, Ntrain, dimension);
  dim3 block2(32, 16);
  dim3 grid2(dimension/32, dimension/16);
  gpuMatMul<<<grid2, block2>>>(d_Xt, d_X, d_covMat, dimension, Ntrain, dimension);

  // Find eigenvalues and eigenvectors of the covariance matrix
  float prev_lambda = 0;
  const float eps = 0.000001;

  float *Q, *normZ, *k_eig_vec;
  float *d_Q, *d_Z, *d_W, *d_normZ;

  int normBuffer = sizeof(float);
  int eigVecBuffer = k * imgBuffer;

  Q = (float*)malloc(imgBuffer);
  normZ = (float*)malloc(normBuffer);
  k_eig_vec = (float*)malloc(eigVecBuffer);

  cudaMalloc((void**)&d_Q, imgBuffer);
  cudaMalloc((void**)&d_Z, imgBuffer);
  cudaMalloc((void**)&d_W, covMatBuffer);
  cudaMalloc((void**)&d_normZ, normBuffer);

  int block3 = 32;
  int grid3 = dimension / block3;
  int sharedMemSize = block3 * sizeof(float);

  for (int i = 0; i < k; i++) {
    Q[0] = 1;
    for (int m = 1; m < dimension; m++) Q[m] = 0;
    cudaMemcpy(d_Q, Q, imgBuffer, cudaMemcpyHostToDevice);
    gpuMatMul<<<16, 256>>>(d_covMat, d_Q, d_Z, dimension, dimension, 1);

    // Power Method iteration
    for (int j = 0; j < 1000; j++) {
      normZ[0] = 0;
      cudaMemcpy(d_normZ, normZ, normBuffer, cudaMemcpyHostToDevice);
      gpuNorm2<<<grid3, block3, sharedMemSize>>>(d_Z, d_normZ, dimension);
      cudaThreadSynchronize();
      cudaMemcpy(normZ, d_normZ, normBuffer, cudaMemcpyDeviceToHost);
      normZ[0] = sqrt(normZ[0]);
      cudaMemcpy(d_normZ, normZ, normBuffer, cudaMemcpyHostToDevice);
      gpuNormalize<<<grid3, block3, sharedMemSize>>>(d_Z, d_normZ, d_Q, dimension);
      cudaThreadSynchronize();
      gpuMatMul<<<16, 256>>>(d_Q, d_covMat, d_Z, 1, dimension, dimension);
      cudaThreadSynchronize();
      normZ[0] = 0;
      cudaMemcpy(d_normZ, normZ, normBuffer, cudaMemcpyHostToDevice);
      gpuEigVal<<<grid3, block3, sharedMemSize>>>(d_Q, d_Z, d_normZ, dimension);
      cudaThreadSynchronize();
      cudaMemcpy(normZ, d_normZ, normBuffer, cudaMemcpyDeviceToHost);

      if (abs(prev_lambda-normZ[0]) < eps) {
        cudaMemcpy(Q, d_Q, imgBuffer, cudaMemcpyDeviceToHost);
        break;
      }
      prev_lambda = normZ[0];
    }

    // The new subspace created by k eigenvectors
    for (int p = 0; p < dimension; p++) {
      DestIndex = p * k + i;
      k_eig_vec[DestIndex] = Q[p];
    }

    // Calculate the new covariance matrix for the next eigenvalue and eigenvector
    gpuMatMul<<<grid2, block2>>>(d_Q, d_Q, d_W, dimension, 1, dimension);
    gpuMatMulScalar<<<grid2, block2>>>(d_W, d_W, dimension, dimension, normZ[0]);
    gpuMatSub<<<grid2, block2>>>(d_covMat, d_W, d_covMat, dimension, dimension);
  }

  // Project the training images in the new subspace
  float *d_kEigVec, *d_proj_train;
  int projTrainBuffer = Ntrain * k * sizeof(float);
  cudaMalloc((void**)&d_kEigVec, eigVecBuffer);
  cudaMalloc((void**)&d_proj_train, projTrainBuffer);
  cudaMemcpy(d_kEigVec, k_eig_vec, eigVecBuffer, cudaMemcpyHostToDevice);
  dim3 grid4((k+block2.x-1)/block2.x, (Ntrain+block2.y-1)/block2.y);
  gpuMatMul<<<grid4, block2>>>(d_X, d_kEigVec, d_proj_train, Ntrain, dimension, k);

  // Project the test images in the new subspace
  float *d_X1;
  cudaMalloc((void**)&d_X1, testMatBuffer);
  dim3 grid5(dimension/128, Ntest);
  gpuSubMean<<<grid5, block1>>>(d_testMat, d_mean, Ntest, dimension, d_X1);

  float *d_proj_test;
  int projTestBuffer = Ntest * k * sizeof(float);
  cudaMalloc((void**)&d_proj_test, projTestBuffer);
  dim3 grid6((k+block2.x-1)/block2.x, (Ntest+block2.y-1)/block2.y);
  gpuMatMul<<<grid6, block2>>>(d_X1, d_kEigVec, d_proj_test, Ntest, dimension, k);

  // Copy projected training and test matrices from Device to Host
  float *project_train_img, *project_test_img;
  project_train_img = (float*)malloc(projTrainBuffer);
  project_test_img = (float*)malloc(projTestBuffer);
  cudaMemcpy(project_train_img, d_proj_train, projTrainBuffer, cudaMemcpyDeviceToHost);
  cudaMemcpy(project_test_img, d_proj_test, projTestBuffer, cudaMemcpyDeviceToHost);

  cudaProfilerStop();

  // Choose the most similar images in the training set
  int *recognized_img;
  int recogBuffer = Ntest * sizeof(int);
  recognized_img = (int*)malloc(recogBuffer);
  identify(project_train_img, project_test_img, Ntrain, Ntest, k, recognized_img);

  fp_id = fopen("identify.txt", "wb");
  if (fp_id == NULL) {
    printf("Error open file\n");
    return 1;
  }
  fprintf(fp_id, "Test Image\t\tPrediction\n");
  for (int i = 0; i < Ntest; i++) {
    fprintf(fp_id, "%d \t\t\t\t %d\n", i + 1, recognized_img[i]);
  }
  fclose(fp_id);

  free(fp_train);
  free(fp_test);
  free(fp_id);
  free(trainMat);
  free(testMat);
  free(Q);
  free(normZ);
  free(k_eig_vec);
  free(project_train_img);
  free(project_test_img);
  free(recognized_img);

  cudaFree(d_trainMat);
  cudaFree(d_testMat);
  cudaFree(d_mean);
  cudaFree(d_X);
  cudaFree(d_covMat);
  cudaFree(d_Q);
  cudaFree(d_Z);
  cudaFree(d_W);
  cudaFree(d_normZ);
  cudaFree(d_kEigVec);
  cudaFree(d_proj_train);
  cudaFree(d_X1);
  cudaFree(d_proj_test);

  return 0;
}
