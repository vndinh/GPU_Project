#include "stdafx.h"
#include <stdio.h>
#include <math.h>
#include <iostream>

void cpuMatAdd(float* A, float* B, int Nrows, int Ncols, float* C) {
	int col = 0;
	int row = 0;
	int DestIndex = 0;
	for (col = 0; col < Ncols; col++) {
		for (row = 0; row < Nrows; row++) {
			DestIndex = row * Ncols + col;
			C[DestIndex] = A[DestIndex] + B[DestIndex];
		}
	}
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

void cpuMatMul(float *A, float *B, int m, int n, int p, float *C) {
	float tmp;
	int outidx;
	for (int col = 0; col < p; col++) {
		for (int row = 0; row < m; row++) {
			outidx = row * p + col;
			tmp = 0;
			for (int idx = 0; idx < n; idx++) {
				tmp += A[row*n + idx] * B[idx*p + col];
			}
			C[outidx] = tmp;
		}
	}
}

void cpuMatMulScalar(float *A, int Nrows, int Ncols, float lambda, float *B) {
	int DestIndex;
	for (int col = 0; col < Ncols; col++) {
		for (int row = 0; row < Nrows; row++) {
			DestIndex = row * Ncols + col;
			B[DestIndex] = lambda * A[DestIndex];
		}
	}
}

void cpuMatTranspose(float *A, int Nrows, int Ncols, float *B) {
	int col = 0;
	int row = 0;
	for (col = 0; col < Ncols; col++) {
		for (row = 0; row < Nrows; row++) {
			B[col * Nrows + row] = A[row * Ncols + col];
		}
	}
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

void PowerMethod(float *A, int n, float eps, float *eigVec, float *lambda) {
	float *Q;
	float *prevQ;
	float *Z;
	float *StepVec;
	float *QtA;
	float norm2z = 0;
	float dist = 1;
	int Buffer = n * sizeof(float);

	Q = (float*)malloc(Buffer);
	prevQ = (float*)malloc(Buffer);
	Z = (float*)malloc(Buffer);
	StepVec = (float*)malloc(Buffer);
	QtA = (float*)malloc(Buffer);

	Q[0] = 1;
	for (int i = 1; i < n; i++) {
		Q[i] = 0;
	}

	do {
		for (int i = 0; i < n; i++) {
			prevQ[i] = Q[i];
		}
		cpuMatMul(A, Q, n, n, 1, Z);
		norm2z = cpuNorm2(Z, n, 1);
		for (int i = 0; i < n; i++) {
			Q[i] = Z[i] / norm2z;
		}
		cpuMatSub(Q, prevQ, n, 1, StepVec);
		dist = cpuNorm2(StepVec, n, 1);
	} while (dist > eps);

	for (int i = 0; i < n; i++) {
		eigVec[i] = Q[i];
	}

	cpuMatMul(Q, A, 1, n, n, QtA);
	cpuMatMul(QtA, Q, 1, n, 1, lambda);

	free(Q);
	free(prevQ);
	free(Z);
	free(StepVec);
	free(QtA);
}

void trainPCA(int *trainMat, int size, int Ntrain, int k, float *project_train_img, float *k_eig_vec, float *mean) {
	int dimension = size * size;
	float tmp;
	int DestIndex;
	int trainMatBuffer = Ntrain * dimension * sizeof(int);

	for (int i = 0; i < dimension; i++) {
		tmp = 0;
		for (int j = 0; j < Ntrain; j++) {
			tmp = tmp + trainMat[j*dimension + i];
		}
		mean[i] = tmp / Ntrain;
	}

	// Subtract mean vector
	float *X;
	int Xbuffer = Ntrain * dimension * sizeof(float);
	X = (float*)malloc(Xbuffer);
	for (int i = 0; i < Ntrain; i++) {
		for (int j = 0; j < dimension; j++) {
			DestIndex = i * dimension + j;
			X[DestIndex] = trainMat[DestIndex] - mean[j];
		}
	}

	// Compute covariance matrix
	float *covMat;
	float *X1;
	int covMatBuffer = dimension * dimension * sizeof(float);
	covMat = (float*)malloc(covMatBuffer);
	X1 = (float*)malloc(Xbuffer);
	cpuMatTranspose(X, Ntrain, dimension, X1);
	cpuMatMul(X1, X, dimension, Ntrain, dimension, covMat);

	int imgBuffer;
	imgBuffer = dimension * sizeof(float);

	float *eigVal;
	float *eigVec;
	float *B;
	float *V;
	float *VVt;
	float *lambda;

	B = (float*)malloc(covMatBuffer);
	V = (float*)malloc(imgBuffer);
	VVt = (float*)malloc(covMatBuffer);
	eigVal = (float*)malloc(k * sizeof(float));
	eigVec = (float*)malloc(imgBuffer);
	lambda = (float*)malloc(sizeof(float));
	lambda[0] = 1;

	for (int i = 0; i < dimension; i++) {
		for (int j = 0; j < dimension; j++) {
			DestIndex = i * dimension + j;
			B[DestIndex] = covMat[DestIndex];
		}
	}
	for (int i = 0; i < k; i++) {
		PowerMethod(B, dimension, 0.00001, eigVec, lambda);

		for (int p = 0; p < dimension; p++) {
			DestIndex = p * k + i;
			k_eig_vec[DestIndex] = eigVec[p];
		}

		eigVal[i] = lambda[0];

		cpuMatMul(eigVec, eigVec, dimension, 1, dimension, VVt);
		cpuMatMulScalar(VVt, dimension, dimension, eigVal[i], VVt);
		cpuMatSub(B, VVt, dimension, dimension, B);
	}

	cpuMatMul(X, k_eig_vec, Ntrain, dimension, k, project_train_img);

	free(X);
	free(covMat);
	free(X1);
	free(eigVal);
	free(B);
	free(V);
	free(VVt);
}

void testPCA(int *testMat, float *k_eig_vec, float *mean, int Ntest, int imgSize, int k, float *project_test_img) {
	int DestIndex;
	int dimension = imgSize * imgSize;
	float *X;
	int Xbuffer = Ntest * dimension * sizeof(float);
	X = (float*)malloc(Xbuffer);
	for (int i = 0; i < Ntest; i++) {
		for (int j = 0; j < dimension; j++) {
			DestIndex = i * dimension + j;
			X[DestIndex] = testMat[DestIndex] - mean[j];
		}
	}

	cpuMatMul(X, k_eig_vec, Ntest, dimension, k, project_test_img);

	free(X);
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
	int Ntrain = 163;	// Number of training images
	int Ntest = 50;		// Number of test images
	int imgSize = 64;	// Size of images
	int k = 50;				// Number of principal components

	int dimension = imgSize * imgSize;	// The original dimension 4096
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


	float *project_train_img;
	float *k_eig_vec;
	float *mean;

	int projTrainBuffer = Ntrain * k * sizeof(float);
	int eigVecBuffer = dimension * k * sizeof(float);
	int meanBuffer = dimension * sizeof(float);

	project_train_img = (float*)malloc(projTrainBuffer);
	k_eig_vec = (float*)malloc(eigVecBuffer);
	mean = (float*)malloc(meanBuffer);

	// Traing PCA
	trainPCA(trainMat, imgSize, Ntrain, k, project_train_img, k_eig_vec, mean);

	float *project_test_img;
	int projTestBuffer = Ntest * k * sizeof(float);
	project_test_img = (float*)malloc(projTestBuffer);

	// Project test matrix in the new subspace
	testPCA(testMat, k_eig_vec, mean, Ntest, imgSize, k, project_test_img);

	int *recognized_img;
	int recogBuffer = Ntest * sizeof(int);
	recognized_img = (int*)malloc(recogBuffer);

	// Choose the most similar image in the training set
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
	free(project_train_img);
	free(project_test_img);
	free(k_eig_vec);
	free(mean);
	free(recognized_img);

	return 0;
}

