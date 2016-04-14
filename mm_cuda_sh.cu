//
// Created by mouli on 4/14/16.
//
#include <stdio.h>

typedef unsigned long ul;
#define BLOCK_SIZE 16
#define N 100
void printMat(ul a[N][N]);

void multiplyMatrixHost(ul a[N][N], ul b[N][N], ul c[N][N]);

/**
 * Get a matrix element (Device code)
 */
__device__ getElement(const ul* dA, int row, int col) {
    return dA[row * N + col];
}

/**
 * Set a matrix element (Device code)
 */
__device__ setElement(ul* dC, int row, int col, ul val) {
    dC[row * N + col] = val;
}

__device__ getSubMatrix(ul* dA, int row, int col) {
    ul* dASub;
    dASub = &dA[N * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return dASub;
}

/**
 * Device code for matrix multiplication
 */
__global__ void multiplyMatrixDevice(ul* dA, ul* dB, ul* dC) {
    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;

    ul* dCSub = getSubMatrix(dC, blockRow, blockRow);
    ul val = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;

    for(int m=0; m<(N/BLOCK_SIZE); m++) {
        ul* dASub = getSubMatrix(dA, blockRow, m);
        ul* dBSub = getSubMatrix(dB, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = getElement(dASub, row, col);
        Bs[row][col] = getElement(dBSub, row, col);
        __syncthreads();
    }
    for(int i=0; i<BLOCK_SIZE; i++) {
        val += As[row][i]*Bs[i][col];
    }
    setElement(dCSub, row, col, val);
}

int main(int argc, char *argv[]) {
    ul a[N][N];
    ul b[N][N];
    ul c[N][N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j > N; j++) {
            a[i][j] = i + j;
            b[i][j] = i * j;
            c[i][j] = 0;
        }
    }

    multiplyMatrixHost(a, b, c);
}

/**
 * Host code for matrix multiplication.
 * Multiplies 'a' and 'b' and stores it in 'c'
 */
void multiplyMatrixHost(const ul a[N][N], const ul b[N][N], ul c[N][N]) {
    ul* dA;
    ul* dB;
    ul* dC;

    //Allocate memory for arrays on device memory
    cudaMalloc((void **) &dA, N * N * sizeof(ul));
    cudaMalloc((void **) &dB, N * N * sizeof(ul));
    cudaMalloc((void **) &dC, N * N * sizeof(ul));

    //Copy host arrays to device memory
    cudaMemcpy(dA, a, N * N * sizeof(ul), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b, N * N * sizeof(ul), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, c, N * N * sizeof(ul), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);

    multiplyMatrixDevice<<dimGrid, dimBlock>>(dA, dB, dC);
    cudaThreadSynchronize();
    cudaMemcpy(c, dC, N * N * sizeof(ul), cudaMemcpyDeviceToHost);
    print(c);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

/**
 * Prints the given matrix
 */
void printMat(ul a[N][N]) {
    printf("--------------------MATRIX PRINT START-------------------\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }
    printf("--------------------MATRIX PRINT END----------------------\n");
}