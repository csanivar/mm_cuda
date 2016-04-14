//
// Created by mouli on 4/14/16.
//
#include <stdio.h>
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_device_runtime_api.h"

#define BLOCK_SIZE 16
#define N 100
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
typedef unsigned int ul;

void printMat(ul a[N][N]);

void multiplyMatrixHost(const ul a[N][N], const ul b[N][N], ul c[N][N]);

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

/**
 * Device code for matrix multiplication
 */
__global__ void multiplyMatrixDevice(ul* dA, ul* dB, ul* dC) {
    ul val = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > N || col > N) {
        return;
    }

    for (int i = 0; i < N; i++) {
        val += dA[row * N + i] * dB[i * N + col];
    }
    dC[row * N + col] = val;
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
    float time;
    cudaEvent_t start, stop;

    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    HANDLE_ERROR( cudaEventRecord(start, 0) );
    multiplyMatrixHost(a, b, c);
    HANDLE_ERROR( cudaEventRecord(stop, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
    printf("Time to generate:  %f ms \n", time);
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
    cudaMalloc((void**) &dA, N * N * sizeof(ul));
    cudaMalloc((void**) &dB, N * N * sizeof(ul));
    cudaMalloc((void**) &dC, N * N * sizeof(ul));

    //Copy host arrays to device memory
    cudaMemcpy(dA, a, N * N * sizeof(ul), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b, N * N * sizeof(ul), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, c, N * N * sizeof(ul), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N + dimBlock.x - 1 / dimBlock.x, N + dimBlock.y - 1 / dimBlock.y);

    multiplyMatrixDevice<<<dimGrid, dimBlock>>>(dA, dB, dC);
    cudaThreadSynchronize();
    cudaMemcpy(c, dC, N * N * sizeof(ul), cudaMemcpyDeviceToHost);
    printMat(c);

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
            printf("%u ", a[i][j]);
        }
        printf("\n");
    }
    printf("--------------------MATRIX PRINT END----------------------\n");
}