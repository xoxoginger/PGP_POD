#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

typedef unsigned int uint;

__global__ void kernel_radixsort(uint* gpu_arr_prev, int arr_size, int k, uint* gpu_arr, uint* bits)
{
	//чисто формулка
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;

	for (int i = idx; i < arr_size; i += offsetx)
	{
		if (((gpu_arr_prev[i] >> k) & 1) == 0)
		{
			gpu_arr[i - (int)bits[i]] = gpu_arr_prev[i];
		}
		else
		{
			gpu_arr[(int)bits[i] + (arr_size - (int)bits[arr_size])] = gpu_arr_prev[i];
		}
	}
}

__global__ void kernel_bitsshift(uint* gpu_arr, int arr_size, int k, uint* bits)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;

	for (int i = idx; i < arr_size; i += offsetx)
		bits[i] = (gpu_arr[i] >> k) & 1;
}

__global__ void kernel_bitsshift_block(uint* bits, int arr_size, uint* lel, int threads)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;

	for (int i = idx + threads; i < arr_size; i += offsetx)
	{
		bits[i] += lel[i / threads - 1];
	}
}

__global__ void kernel_blellochscan_block(uint* bits, int arr_size, uint* lel, int threads)
{
	int idx = threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;

	__shared__ uint data[1024]; //threads

	uint lel_el, temp;
	int it;

	for (int ib = 0; ib < arr_size; ib += offsetx)
	{
		it = 1;
		if (idx + ib + blockIdx.x * blockDim.x < arr_size)
		{
			data[idx] = bits[idx + ib + blockIdx.x * blockDim.x];

			__syncthreads();

			for (int i = 1; i < blockDim.x; i *= 2, it++) //i <<= 1
			{
				if (((idx - i + 1) & ((1 << it) - 1)) == 0)
					data[idx + i] += data[idx];
				__syncthreads();
			}

			if (idx == 0)
			{
				lel_el = data[blockDim.x - 1];
				data[blockDim.x - 1] = 0;
			}
			__syncthreads();

			it = it - 1;
			for (int i = blockDim.x / 2; i > 0; i /= 2, it--) //i >>= 1
			{
				if (((idx - i + 1) & ((1 << it) - 1)) == 0)
				{
					temp = data[idx + i];
					data[idx + i] = data[idx] + data[idx + i];
					data[idx] = temp;
				}

				__syncthreads();
			}

			if (idx == 0)
			{
				lel[ib / blockDim.x + blockIdx.x] = lel_el;
				bits[ib + (blockIdx.x + 1) * blockDim.x - 1] = lel_el;
			}
			else
			{
				bits[ib + blockIdx.x * blockDim.x + idx - 1] = data[idx];
			}
		}
	}
}


void blellochscan(uint* bits, int arr_size, int blocks, int threads)
{
	int lel_size = arr_size / threads;
	int mult_block_lel = (lel_size + (threads - 1)) & (-threads);
	//size_bLastEls = (sizeLastEls % BLOCK_SIZE == 0) ? sizeLastEls : BLOCK_SIZE * ((int)sizeLastEls / BLOCK_SIZE + 1);

	uint* lel;
	cudaMalloc(&lel, mult_block_lel * sizeof(uint));
	cudaMemset(lel + lel_size, 0, (mult_block_lel - lel_size) * sizeof(uint));

	kernel_blellochscan_block << <blocks, threads >> > (bits, arr_size, lel, threads);

	if (lel_size > 1)
	{
		blellochscan(lel, mult_block_lel, blocks, threads);
		kernel_bitsshift_block << <blocks, threads >> > (bits, arr_size, lel, threads);
	}
	cudaFree(lel);
}

int main()
{
	int blocks = 1024;
	int threads = 1024;

	int arr_size;
	fread(&arr_size, sizeof(int), 1, stdin);
	//std::cin >> arr_size;

	uint* arr = new uint[arr_size];
	fread(arr, sizeof(uint), arr_size, stdin);
	//for (int i = 0; i < arr_size; i++)
	//{
		//std::cin >> arr[i];
	//}

	//bit mask
	int mult_block = (arr_size + (threads - 1)) & (-threads); //наибольшее кратное блоку

	uint* gpu_arr;
	CSC(cudaMalloc(&gpu_arr, arr_size * sizeof(uint)));
	CSC(cudaMemcpy(gpu_arr, arr, arr_size * sizeof(uint), cudaMemcpyHostToDevice));

	uint* gpu_arr_prev;
	CSC(cudaMalloc(&gpu_arr_prev, arr_size * sizeof(uint)));
	CSC(cudaMemcpy(gpu_arr_prev, arr, arr_size * sizeof(uint), cudaMemcpyHostToDevice));

	uint* bits; //bit array
	CSC(cudaMalloc(&bits, (mult_block + 1) * sizeof(uint)));

	for (int k = 0; k < 32; k++) //uint - 32 бита
	{
		kernel_bitsshift << <blocks, threads >> > (gpu_arr, arr_size, k, bits + 1); //k-тый бит
		CSC(cudaGetLastError());

		cudaMemset(bits, 0, sizeof(uint)); //1 эл-т в 0

		if (mult_block > 1) //если он равен единице, тогда не нужно ничего дополнять
		{
			//дополняем нулями до кратности, (mult_block - arr_size) - yekb
			cudaMemset(bits + arr_size + 1, 0, (mult_block - arr_size) * sizeof(uint));

			blellochscan(bits + 1, mult_block, blocks, threads);
		}

		uint* temp = gpu_arr;
		gpu_arr = gpu_arr_prev;
		gpu_arr_prev = temp;

		kernel_radixsort << <blocks, threads >> > (gpu_arr_prev, arr_size, k, gpu_arr, bits);
		CSC(cudaGetLastError());

	}

	CSC(cudaMemcpy(arr, gpu_arr, arr_size * sizeof(uint), cudaMemcpyDeviceToHost));

	fwrite(arr, sizeof(uint), arr_size, stdout);
	//for (int i = 0; i < arr_size; i++)
	//{
		//std::cout << std::endl << arr[i] << ' ';
	//}

	cudaFree(bits);
	cudaFree(gpu_arr);
	cudaFree(gpu_arr_prev);

	free(arr);

	return 0;
}
