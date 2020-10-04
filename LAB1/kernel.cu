#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <iomanip> 
#include <fstream>

using namespace std;

__global__ void kernelAdd(double* v1, double* v2, double* v3, unsigned long long n) {
    unsigned long long i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned long long offset = gridDim.x * blockDim.x;
    for (; i < n; i += offset) {
        v3[i] = v1[i] + v2[i];
    }
}


void MemFree(double* va, double* vb, double* vc, double* v1, double* v2, double* v3)
{
    cudaFree(va);
    cudaFree(vb);
    cudaFree(vc);
    delete[] v1;
    delete[] v2;
    delete[] v3;
}

int main() {
    
    ifstream fin("test1000000.txt");
    unsigned long long n = 0;
    fin >> n;

    double* v1;
    v1 = new double[n] {};
    double* v2;
    v2 = new double[n] {};
    double* v3;
    v3 = new double[n] {};


    for (int i = 0; i < n; i++)
    {
        fin >> v1[i];
    }
    for (int i = 0; i < n; i++)
    {
        fin >> v2[i];
    }

    /*cout << "Enter v1: ";
    for (int i = 0; i < n; i++)
        cin >> v1[i];
    cout << "Enter v2: ";
    for (int i = 0; i < n; i++)
        cin >> v2[i];*/

    double* va = 0, * vb = 0, * vc = 0;

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?:)";
        MemFree(va, vb, vc, v1, v2, v3);
        return 0;
    }
    
    cudaStatus = cudaMalloc((void**)&va, n * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        cout<< "cudaMalloc failed!";
        MemFree(va, vb, vc, v1, v2, v3);
        return 0;
    }

    cudaStatus = cudaMalloc((void**)&vb, n * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        cout << "cudaMalloc failed!";
        MemFree(va, vb, vc, v1, v2, v3);
        return 0;
    }

    cudaStatus = cudaMalloc((void**)&vc, n * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        cout << "cudaMalloc failed!";
        MemFree(va, vb, vc, v1, v2, v3);
        return 0;
    }


    cudaStatus = cudaMemcpy(va, v1, n * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cout << "cudaMemcpy failed!";
        MemFree(va, vb, vc, v1, v2, v3);
        return 0;
    }

    cudaStatus = cudaMemcpy(vb, v2, n * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        cout << "cudaMemcpy failed!";
        MemFree(va, vb, vc, v1, v2, v3);
        return 0;
    }

    
    int x = 1, y = 32;

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    kernelAdd << <x, y >> > (va, vb, vc, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    ofstream fout("output1000000.txt", ios::app);
    fout << endl << setprecision(10) << "x = " << x << " y = " << y << "\nGPU Time [ms] " << time << endl;
    x = 32;
    while (x <= 1024)
    {
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        kernelAdd << <x, y >> > (va, vb, vc, n);
        cudaStatus = cudaMemcpy(v3, vc, n * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            cout << "cudaMemcpy failed!";
            MemFree(va, vb, vc, v1, v2, v3);
            return 0;
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        ofstream fout("output1000000.txt", ios::app);
        fout << endl << setprecision(10) << "x = " << x << " y = " << y << "\nGPU Time [ms] " << time << endl;
        x *= 2;
        y *= 2;
    }

    cudaStatus = cudaMemcpy(v3, vc, n * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        cout << "cudaMemcpy failed!";
        MemFree(va, vb, vc, v1, v2, v3);
        return 0;
    }
    
    //kernelAdd << <256, 256 >> > (va, vb, vc, n);
    //16384, 512
    

    //for (int i = 0; i < n; i++)
        //cout << fixed << setprecision(10)<< v3[i] << ' ';
        //printf("%.10e ", v3[i]);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        cout << "cudaDeviceReset failed!";
        MemFree(va, vb, vc, v1, v2, v3);
        return 1;
    }

    MemFree(va, vb, vc, v1, v2, v3);
    return 0;
}
