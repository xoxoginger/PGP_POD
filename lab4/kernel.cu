
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdio.h>

#include <thrust/swap.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h> //swap

using namespace std;

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

struct abs_comp 
{
    __host__ __device__ bool operator()(double x, double y) 
    {
        return fabs(x) < fabs(y);
    }
};


__global__ void kernel_swap(double* A, int N, int i, int max_idx) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = gridDim.x * blockDim.x;

    for (int k = idx; k < N; k += offsetX) 
    {
        //thrust::swap(A[i][k], A[max_idx][k]);
        thrust::swap(A[i + N * k], A[max_idx + N * k]);
    }
}
__global__ void kernel_LUP(double* A, int N, int i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;

    //compute LU
    for (int k = i + 1 + idx; k < N; k += offsetx)
    {
       // A[k][i] /= A[i][i];
        A[k + N * i] /= A[i + N * i];
        for (int j = i + 1 + idy; j < N; j += offsety) // iterate across rows
           // A[k][j] = A[k][j] - A[i][j] * A[k][i];
            A[k + N * j] = A[k + N * j] - A[i + N * j] * A[k + N * i];
    }

}

int main()
{   
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr); // - если cin

    //init
    int N;
    //cin >> N;
    scanf("%d", &N);
    double* A, * dev_A;
    CSC(cudaMalloc((void**)&dev_A, sizeof(double) * N * N));
    int* P = (int*)malloc(sizeof(int) * N);
    A = (double*)malloc(sizeof(double) * N * N);

    for (int i = 0; i < N; i++) 
    {
        P[i] = i;
        double t;
        for (int j = 0; j < N; j++) 
        {
            //cin >> A[i + N * j];
            scanf("%lf", &t);
            A[i + N * j] = t;
            //A[i + N * j] = (double)rand() * 0.1 + 0.3; - tests
        } 
    }

    //здесь измерять время 
    cudaEvent_t start, stop;
    float gpu_time = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    CSC(cudaMemcpy(dev_A, A, sizeof(double) * N * N, cudaMemcpyHostToDevice));

    int max_idx;
    abs_comp comp;
    thrust::device_ptr<double> data_ptr;
    thrust::device_ptr<double> max_ptr;

    for (int i = 0; i < N - 1; ++i) 
    {
        max_idx = i;

        data_ptr = thrust::device_pointer_cast(dev_A + i * N);// + i * N);
        max_ptr = thrust::max_element(data_ptr + i, data_ptr + N, comp); //+i, +N
        max_idx = max_ptr - data_ptr;
        

        P[i] = max_idx;
        if (max_idx != i) 
        {
            kernel_swap << <256, 256 >> > (dev_A, N, i, max_idx);
            CSC(cudaGetLastError());
        }

        kernel_LUP << <256, 256 >> > (dev_A, N, i);
        CSC(cudaGetLastError());
        CSC(cudaThreadSynchronize());

    }

    CSC(cudaMemcpy(A, dev_A, sizeof(double) * N * N, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_A));

     cudaEventRecord(stop, 0); //- tests
     cudaEventSynchronize(stop); //
     cudaEventElapsedTime(&gpu_time, start, stop); //
     printf("Time %f\n", gpu_time); //
    
     for (int i = 0; i < N; ++i)
     {
         for (int j = 0; j < N; ++j)
         {
             printf("%.10e ", A[i + N * j]);
         }
         printf("\n");
     }

    for (int j = 0; j < N; ++j) 
    {
        printf("%d ", P[j]);
    }

    free(A);
    free(P);
    
    return 0;
}
