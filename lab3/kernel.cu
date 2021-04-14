#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
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

typedef struct
{
	int x;
	int y;
} vector2;

typedef struct
{
	double x;
	double y;
	double z;
} vector3;

__constant__ vector3 gpu_avgs[32];
__constant__ vector3 gpu_norm_avgs[32];

vector3 copy_avgs[32];
vector3 copy_norm_avgs[32];

__device__ __host__ void RGBget(vector3& rgb, uchar4* pixel)
{
	rgb.x = pixel->x;
	rgb.y = pixel->y;
	rgb.z = pixel->z;
}

void _avgs_(vector<vector<vector2>>& vec, uchar4* data, int w, int h, int nc)
{
	vector<vector3> avgs(32);
	for (int i = 0; i < nc; i++)
	{
		avgs[i].x = 0;
		avgs[i].y = 0;
		avgs[i].z = 0;

		for (int j = 0; j < vec[i].size(); j++)
		{
			vector2 point = vec[i][j];
			uchar4 pixel = data[point.y * w + point.x];
			vector3 rgb;
			RGBget(rgb, &pixel);

			avgs[i].x += rgb.x;
			avgs[i].y += rgb.y;
			avgs[i].z += rgb.z;
		}

		avgs[i].x /= vec[i].size();
		avgs[i].y /= vec[i].size();
		avgs[i].z /= vec[i].size();
	}

	for (int i = 0; i < nc; i++)
		copy_avgs[i] = avgs[i];
	//return avgs;
}

void _norm_avgs_(int nc)
{
	for (int i = 0; i < nc; i++)
	{
		
		//cout << sqrt(pow(copy_avgs[i].x, 2) + pow(copy_avgs[i].y, 2) + pow(copy_avgs[i].z, 2));
		copy_norm_avgs[i].x = (double)copy_avgs[i].x / sqrt(pow(copy_avgs[i].x, 2) + pow(copy_avgs[i].y, 2) + pow(copy_avgs[i].z, 2));
		copy_norm_avgs[i].y = (double)copy_avgs[i].y / sqrt(pow(copy_avgs[i].x, 2) + pow(copy_avgs[i].y, 2) + pow(copy_avgs[i].z, 2));
		copy_norm_avgs[i].z = (double)copy_avgs[i].z / sqrt(pow(copy_avgs[i].x, 2) + pow(copy_avgs[i].y, 2) + pow(copy_avgs[i].z, 2));
		
	}
}

__device__ double find(uchar4 pixel, int c)
{
	vector3 rgb;
	RGBget(rgb, &pixel);
	//vector3 div_avgs;

	double res = 0;

	double trgb[3];
	trgb[0] = rgb.x;
	trgb[1] = rgb.y;
	trgb[2] = rgb.z;
	double tnorm[3];
	tnorm[0] = gpu_norm_avgs[c].x;
	tnorm[1] = gpu_norm_avgs[c].y;
	tnorm[2] = gpu_norm_avgs[c].z;
	for (int i = 0; i < 3; i++)
			res += trgb[i] * tnorm[i];
	return res;
}

__global__ void spectral_kernel(uchar4* data, int w, int h, int nc)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	
	for (y = idy; y < h; y += offsety)
		for (x = idx; x < w; x += offsetx)
		{
			uchar4 pixel = data[y * w + x];
			double sa = find(pixel, 0);
			int sac = 0;
			for (int i = 1; i < nc; i++)
			{
				double tsa = find(pixel, i);
				if (sa < tsa) //argmax
				{
					sa = tsa;
					sac = i;
				}
			}
			data[y * w + x].w = (unsigned char)sac;
		}
}
int main() 
{
	int w, h;
	string inFile;
	string outFile;

	std::cin >> inFile >> outFile;

	FILE* fp = fopen(inFile.c_str(), "rb");
	//FILE* fp = fopen("in.data", "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);
	int nc, np; //classes, pixels
	cin >> nc;
	vector<vector<vector2>> vec(nc); //coords
	for (int i = 0; i < nc; i++)
	{
		cin >> np;
		vec[i].resize(np);
		for (int j = 0; j < np; j++)
		{
			cin >> vec[i][j].x >> vec[i][j].y;
		}
	}

	_avgs_(vec, data, w, h, nc);
	_norm_avgs_(nc);

	CSC(cudaMemcpyToSymbol(gpu_avgs, copy_avgs, 32 * sizeof(vector3)));
	
	CSC(cudaMemcpyToSymbol(gpu_norm_avgs, copy_norm_avgs, 32 * sizeof(vector3)));

	uchar4* out;

	CSC(cudaMalloc(&out, sizeof(uchar4) * w * h));

	CSC(cudaMemcpy(out, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	spectral_kernel << <dim3(32, 32), dim3(32, 32) >> > (out, w, h, nc);

	CSC(cudaGetLastError());

	CSC(cudaMemcpy(data, out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

	CSC(cudaFree(out));

	fp = fopen(outFile.c_str(), "wb");
	//fp = fopen("out.data", "wb");
	fwrite(&w, sizeof(int), 1, fp);
	fwrite(&h, sizeof(int), 1, fp);
	fwrite(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	free(data);

	return 0;
}
