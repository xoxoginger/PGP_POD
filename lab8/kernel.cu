//#pragma warning(disable : 4996)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <bits/stdc++.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"
#include "mpi.h"
#include <iostream>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
using namespace std;

// Индексация внутри блока
#define _i(i, j, k) (((k) + 1) * (dy + 2) * (dx + 2) + ((j) + 1) * (dx + 2) + (i) + 1)

// Индексация по блокам (процессам)
#define _ib(i, j, k) ((k) * bly * blx + (j) * blx + (i))
#define _ibz(id) ((id) / bly / blx)
#define _iby(id) (((id) % (bly * blx)) / blx)
#define _ibx(id) ((id) % blx)

#define CSC(call)                           \
do {                                \
  cudaError_t res = call;                     \
  if (res != cudaSuccess) {                   \
    fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
        __FILE__, __LINE__, cudaGetErrorString(res));   \
    exit(0);                          \
  }                               \
} while(0)

//struct
//{
    //int x;
    //int y;
    //int z;
//} dim;

//struct
//{
    //int x;
   // int y;
   // int z;
//} block;



//struct
//{
    //double x;
    //double y;
    //double z;
//} l; //area size

/*struct
{
    double down;
    double up;
    double left;
    double right;
    double front;
    double back;

} u; //border conditions
*/
__global__ void plzwork_computekernel(double* next, double* data, int dx, int dy, int dz, double hx, double hy, double hz) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsety = blockDim.y * gridDim.y;
    int offsetx = blockDim.x * gridDim.x;

    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetz = blockDim.z * gridDim.z;
   
    for (int i = idx; i < dx; i += offsetx)
    {
        for (int j = idy; j < dy; j += offsety)
        {
            for (int k = idz; k < dz; k += offsetz)
            {
                next[_i(i, j, k)] = ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
                    (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
                    (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
                    (2 * (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz)));
            }
        }
            
    }
        
}

__global__ void plzwork_errorkernel(double* next, double* data, int dx, int dy, int dz) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsety = blockDim.y * gridDim.y;
    int offsetx = blockDim.x * gridDim.x;

    int idz = blockIdx.z * blockDim.z + threadIdx.z;
    int offsetz = blockDim.z * gridDim.z;

    for (int i = idx - 1; i < dx + 1; i += offsetx)
    {
        for (int j = idy - 1; j < dy + 1; j += offsety)
        {
            for (int k = idz - 1; k < dz + 1; k += offsetz) 
            {
                data[_i(i, j, k)] = ((i != -1 && j != -1 && k != -1 && i != dx && j != dy && k != dz)) * fabs(next[_i(i, j, k)] - data[_i(i, j, k)]);
            }
        }      
    }    
}

__global__ void plzwork_zycopy(double* zy_edge, double* data, int dx, int dy, int dz, int i, int fl, double u) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsety = blockDim.y * gridDim.y;
    int offsetx = blockDim.x * gridDim.x;

   
    if (fl) 
    {
        for (int k = idy; k < dz; k += offsety)
        {
            for (int j = idx; j < dy; j += offsetx)
                zy_edge[k * dy + j] = data[_i(i, j, k)];
        }
            
    }
    else 
    {
        if (zy_edge) 
        {
            for (int k = idy; k < dz; k += offsety)
            {
                for (int j = idx; j < dy; j += offsetx)
                {
                    data[_i(i, j, k)] = zy_edge[k * dy + j];
                }  
            }     
        }
        else 
        {
            for (int k = idy; k < dz; k += offsety)
            {
                for (int j = idx; j < dy; j += offsetx)
                {
                    data[_i(i, j, k)] = u;
                }
                    
            }
                
        }
    }
}

__global__ void plzwork_zxcopy(double* zx_edge, double* data, int dx, int dy, int dz, int j, int fl, double u) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsety = blockDim.y * gridDim.y;
    int offsetx = blockDim.x * gridDim.x;

    if (fl) 
    {
        for (int k = idy; k < dz; k += offsety)
        {
            for (int i = idx; i < dx; i += offsetx)
            {
                zx_edge[k * dx + i] = data[_i(i, j, k)];
            }
                
        }
            
    }
    else 
    {
        if (zx_edge) 
        {
            for (int k = idy; k < dz; k += offsety)
            {
                for (int i = idx; i < dx; i += offsetx)
                {
                    data[_i(i, j, k)] = zx_edge[k * dx + i];
                }     
            }     
        }
        else 
        {
            for (int k = idy; k < dz; k += offsety)
            {
                for (int i = idx; i < dx; i += offsetx)
                {
                    data[_i(i, j, k)] = u;
                }
                    
            }     
        }
    }
}

__global__ void plzwork_xycopy(double* xy_edge, double* data, int dx, int dy, int dz, int k, int fl, double u) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsety = blockDim.y * gridDim.y;
    int offsetx = blockDim.x * gridDim.x;

    if (fl) 
    {
        for (int j = idy; j < dy; j += offsety)
        {
            for (int i = idx; i < dx; i += offsetx)
            {
                xy_edge[j * dx + i] = data[_i(i, j, k)];
            }
                
        }
           
    }
    else 
    {
        if (xy_edge) 
        {
            for (int j = idy; j < dy; j += offsety)
            {
                for (int i = idx; i < dx; i += offsetx)
                {
                    data[_i(i, j, k)] = xy_edge[j * dx + i];
                }
                    
            }
                
        }
        else 
        {
            for (int j = idy; j < dy; j += offsety)
            {
                for (int i = idx; i < dx; i += offsetx)
                {
                    data[_i(i, j, k)] = u;
                }       
            }       
        }
    }
}


int main(int argc, char* argv[])
{
    int id; //номер процесса, вызвавшего функцию
    int bx, by, bz; //ib, jb, kb
    int i, j, k;
    int numproc, proc_name_len; //число процессов
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    double eps, u0;
    double* temp;
    //string outFile; 
    char outFile[1024];
    double diff;
    double total_diff = 0;

    int dx, dy, dz; //dim
    int blx, bly, blz; //block
    double lx, ly, lz;
    double down;
    double up;
    double left;
    double right;
    double front;
    double back;
    MPI_Status status; //статус выполнения операций mpi
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc); //число процессов в области связи коммуникатора comm
    MPI_Comm_rank(MPI_COMM_WORLD, &id); //номер процесса, вызвавшего функцию
    MPI_Get_processor_name(proc_name, &proc_name_len);


    if (id == 0) //если главный процесс
    {
        //input data
        cin >> blx >> bly >> blz;          // Размер сетки блоков (процессов)
        cin >> dx >> dy >> dz; // Размер блока
        cin >> outFile;
        cin >> eps;
        cin >> lx >> ly >> lz;
        cin >> down >> up >> left >> right >> front >> back;
        cin >> u0;
    }

    // Передача параметров расчета всем процессам
    //Процесс с номером root(0) рассылает сообщение из своего буфера передачи всем процессам области связи коммуникаторa
    //то есть, передаем 3 эл-та из block всем процессам
    MPI_Bcast(&dx, 1, MPI_INT, 0, MPI_COMM_WORLD); 
    MPI_Bcast(&dy, 1, MPI_INT, 0, MPI_COMM_WORLD);    
    MPI_Bcast(&dz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bly, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&blz, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Bcast(&l, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //MPI_Bcast(&u, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(outFile, 1024, MPI_CHAR, 0, MPI_COMM_WORLD);

    //buf init
    double* data = (double*)malloc(sizeof(double) * (dx + 2) * (dy + 2) * (dz + 2));
    double* next = (double*)malloc(sizeof(double) * (dx + 2) * (dy + 2) * (dz + 2));
    double* buff = (double*)malloc(sizeof(double) * (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2));

    double* gpu_data;
    CSC(cudaMalloc(&gpu_data, sizeof(double) * (dx + 2) * (dy + 2) * (dz + 2)));
    double* gpu_next;
    CSC(cudaMalloc(&gpu_next, sizeof(double) * (dx + 2) * (dy + 2) * (dz + 2)));
    double* gpu_buff;
    CSC(cudaMalloc(&gpu_buff, sizeof(double) * (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2)));

    //zero iteration
    for (i = 0; i < dx; i++)
    {
        for (j = 0; j < dy; j++)
        {
            for (k = 0; k < dz; k++)
            {
                data[_i(i, j, k)] = u0;
            }

        }

    }


    bx = _ibx(id);    // Переход к 3-мерной индексации процессов
    by = _iby(id);
    bz = _ibz(id);

    double hx = lx / (dx * blx);
    double hy = ly / (dy * bly);
    double hz = lz / (dz * blz);

    dim3 blocks(32, 32);
    dim3 threads(32, 32);

    CSC(cudaMemcpy(gpu_data, data, sizeof(double) * (dx + 2) * (dy + 2) * (dz + 2), cudaMemcpyHostToDevice));
    
    do
    {
        if (bx < blx - 1)
        {
            plzwork_zycopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, dx - 1, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, gpu_buff, sizeof(double) * (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2), cudaMemcpyDeviceToHost));
            MPI_Send(buff, dy * dz, MPI_DOUBLE, _ib(bx + 1, by, bz), id, MPI_COMM_WORLD);
        }

        if (bx > 0)
        {
            MPI_Recv(buff, dy * dz, MPI_DOUBLE, _ib(bx - 1, by, bz), _ib(bx - 1, by, bz), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(gpu_buff, buff, sizeof(double) * (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2), cudaMemcpyHostToDevice));
            plzwork_zycopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, -1, false, 0.0);
        }
        else
        {
            plzwork_zycopy << <blocks, threads >> > (NULL, gpu_data, dx, dy, dz, -1, false, left);
        }

        if (by < bly - 1)
        {
            plzwork_zxcopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, dy - 1, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, gpu_buff, sizeof(double) * (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2), cudaMemcpyDeviceToHost));
            MPI_Send(buff, dz * dx, MPI_DOUBLE, _ib(bx, by + 1, bz), id, MPI_COMM_WORLD);
        }

        if (by > 0)
        {
            MPI_Recv(buff, dx * dz, MPI_DOUBLE, _ib(bx, by - 1, bz), _ib(bx, by - 1, bz), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(gpu_buff, buff, sizeof(double) * (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2), cudaMemcpyHostToDevice));
            plzwork_zxcopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, -1, false, 0.0);

        }
        else
        {
            plzwork_zxcopy << <blocks, threads >> > (NULL, gpu_data, dx, dy, dz, -1, false, front);
        }

        if (bz < blz - 1)
        {
            plzwork_xycopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, dz - 1, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, gpu_buff, sizeof(double)* (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2), cudaMemcpyDeviceToHost));
            //double r = _ib(bx, by, bz + 1);
            //cout << r;
            MPI_Send(buff, dy * dx, MPI_DOUBLE, _ib(bx, by, bz + 1), id, MPI_COMM_WORLD);
        }

        if (bz > 0)
        {
            MPI_Recv(buff, dx * dy, MPI_DOUBLE, _ib(bx, by, bz - 1), _ib(bx, by, bz - 1), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(gpu_buff, buff, sizeof(double) * (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2), cudaMemcpyHostToDevice));
            plzwork_xycopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, -1, false, 0.0);

        }
        else
        {
            plzwork_xycopy << <blocks, threads >> > (NULL, gpu_data, dx, dy, dz, -1, false, down);
        }

        if (bx > 0)
        {
            plzwork_zycopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, 0, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, gpu_buff, sizeof(double) * (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2), cudaMemcpyDeviceToHost));
            MPI_Send(buff, dz * dy, MPI_DOUBLE, _ib(bx - 1, by, bz), id, MPI_COMM_WORLD);
        }

        if (bx < blx - 1)
        {
            MPI_Recv(buff, dy * dz, MPI_DOUBLE, _ib(bx + 1, by, bz), _ib(bx + 1, by, bz), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(gpu_buff, buff, sizeof(double) * (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2), cudaMemcpyHostToDevice));
            plzwork_zycopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, dx, false, 0.0);
        }
        else
        {
            plzwork_zycopy << <blocks, threads >> > (NULL, gpu_data, dx, dy, dz, dx, false, right);
        }

        if (by > 0)
        {
            plzwork_zxcopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, 0, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, gpu_buff, sizeof(double)* (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2), cudaMemcpyDeviceToHost));
            MPI_Send(buff, dz * dx, MPI_DOUBLE, _ib(bx, by - 1, bz), id, MPI_COMM_WORLD);
        }

        if (by < bly - 1)
        {
            MPI_Recv(buff, dx * dz, MPI_DOUBLE, _ib(bx, by + 1, bz), _ib(bx, by + 1, bz), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(gpu_buff, buff, sizeof(double) * (max(dx, max(dy, dz)) * max(dx, max(dy, dz)) + 2), cudaMemcpyHostToDevice));
            plzwork_zxcopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, dy, false, 0.0);
        }
        else
        {
            plzwork_zxcopy << <blocks, threads >> > (NULL, gpu_data, dx, dy, dz, dy, false, back);

        }

        if (bz > 0)
        {
            plzwork_xycopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, 0, true, 0.0);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(buff, gpu_buff, sizeof(double)* (max(dx, max(dy, dz))* max(dx, max(dy, dz)) + 2), cudaMemcpyDeviceToHost));
            MPI_Send(buff, dy * dx, MPI_DOUBLE, _ib(bx, by, bz - 1), id, MPI_COMM_WORLD);
        }

        if (bz < blz - 1)
        {
            MPI_Recv(buff, dx * dy, MPI_DOUBLE, _ib(bx, by, bz + 1), _ib(bx, by, bz + 1), MPI_COMM_WORLD, &status);
            CSC(cudaMemcpy(gpu_buff, buff, sizeof(double)* (max(dx, max(dy, dz))* max(dx, max(dy, dz)) + 2), cudaMemcpyHostToDevice));
            plzwork_xycopy << <blocks, threads >> > (gpu_buff, gpu_data, dx, dy, dz, dz, false, 0.0);
        }
        else
        {
            plzwork_xycopy << <blocks, threads >> > (NULL, gpu_data, dx, dy, dz, dz, false, up);
        }

        plzwork_computekernel<< <dim3(8, 8, 8), dim3(32, 4, 4) >> > (gpu_next, gpu_data, dx, dy, dz, hx, hy, hz);
        CSC(cudaGetLastError());

        plzwork_errorkernel<< <dim3(8, 8, 8), dim3(32, 4, 4) >> > (gpu_next, gpu_data, dx, dy, dz);
        CSC(cudaGetLastError());

        diff = 0.0; //ыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыыы
        //double error = 0.0;
        thrust::device_ptr<double> p_arr = thrust::device_pointer_cast(gpu_data);
        thrust::device_ptr<double> res = thrust::max_element(p_arr, p_arr + (dx + 2) * (dy + 2) * (dz + 2));
        diff = *res;

        //std::cerr << total_diff << " ";
        temp = gpu_data;
        gpu_data = gpu_next;
        gpu_next = temp;

        total_diff = 0.0;
        double* diffs = (double*)malloc(sizeof(double) * blx * bly * blz);
        MPI_Allgather(&diff, 1, MPI_DOUBLE, diffs, 1, MPI_DOUBLE, MPI_COMM_WORLD);

        for (k = 0; k < blx * bly * blz; k++)
        {
            total_diff = max(total_diff, diffs[k]);
        }
        //std::cerr << total_diff << endl;


    } while (total_diff > eps); //ыыыыыыыыыыыыыыыыыы х2
    //std::cerr << total_diff << " ";

    CSC(cudaMemcpy(data, gpu_data, sizeof(double) * (dx + 2)* (dy + 2)* (dz + 2), cudaMemcpyDeviceToHost));
    std::cerr << data << endl;
    CSC(cudaFree(gpu_data));
    CSC(cudaFree(gpu_buff));
    CSC(cudaFree(gpu_next));

    int s_size = 14;
    char* out_buff = (char*)malloc(sizeof(char) * dx * dy * dz * s_size);

    memset(out_buff, ' ', dx* dy* dz* s_size * sizeof(char));
    
    for (k = 0; k < dz; k++) 
    {
        for (j = 0; j < dy; j++)
        {
            for (i = 0; i < dx; i++)
            {
                sprintf(out_buff + ((k * dx * dy) + j * dx + i) * s_size, "%.6e", data[_i(i, j, k)]);
            }
                
        }    
    }

    for (i = 0; i < dx * dy * dz * s_size; i++) 
    {
        if (out_buff[i] == '\0')
        {
            out_buff[i] = ' ';
            //fprintf(stderr, "%c", out_buff[i]);
        }     
    }


    
    MPI_Datatype cell;
    MPI_Type_contiguous(s_size, MPI_CHAR, &cell);
    MPI_Type_commit(&cell);

    MPI_Datatype subarray;
    int subarray_starts[3] = {0, 0, 0};
    int subarray_subsizes[3] = {dx, dy, dz};
    int subarray_bigsizes[3] = {dx, dy, dz};
    MPI_Type_create_subarray(3, subarray_bigsizes, subarray_subsizes, subarray_starts, MPI_ORDER_FORTRAN, cell, &subarray); // memtype
    MPI_Type_commit(&subarray);

    MPI_Datatype bigarray;
    int bigarray_starts[3] = { bx * dx, by * dy, bz * dz };
    int bigarray_subsizes[3] = { dx, dy, dz };
    int bigarray_bigsizes[3] = { dx * blx, dy * bly, dz * blz };
    MPI_Type_create_subarray(3, bigarray_bigsizes, bigarray_subsizes, bigarray_starts, MPI_ORDER_FORTRAN, cell, &bigarray); // memtype
    MPI_Type_commit(&bigarray);

    MPI_File out;
    MPI_File_delete(outFile, MPI_INFO_NULL);
    MPI_File_open(MPI_COMM_WORLD, outFile, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out);
    //std::cerr << "work";

    MPI_File_set_view(out, 0, MPI_CHAR, bigarray, "native", MPI_INFO_NULL);
    MPI_File_write_all(out, out_buff, 1, subarray, MPI_STATUS_IGNORE);
    MPI_File_close(&out);
    
    MPI_Finalize();

    free(data);
    free(buff);
    free(next);
    free(out_buff);

    return 0;
}



