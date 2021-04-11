#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>

using namespace std;

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void bilinear_interp_kernel(uchar4* out, int w, int h, int wn, int hn)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y, x_n, y_n;
	double x2, y2;
	uchar4 p1, p2, p3, p4;
	char r, g, b;
	
	for (y = idy; y < hn; y += offsety)
		for (x = idx; x < wn; x += offsetx)
		{
			y_n = max(min(y, h), 1);
			x_n = max(min(x, w), 1);
			//x_n = (x + 0.5) * w / wn - 0.5;
			//y_n = (y + 0.5) * h / hn - 0.5;

			p1 = tex2D(tex, x_n, y_n); //01
			p2 = tex2D(tex, x_n + 1, y_n); //11
			p3 = tex2D(tex, x_n, y_n + 1); // 00
			p4 = tex2D(tex, x_n + 1, y_n + 1); //10

			
			r = (p1.x * (x_n + 1 - x) * (y_n + 1 - y) + p2.x * (x - x_n) * (y_n + 1 - y) + p3.x * (x_n + 1 - x) * (y - y_n) + p4.x * (x - x_n) * (y - y_n)); // ((x_n+1 - x_n) * (y_n +1 - y_n));
			g = (p1.y * (x_n + 1 - x) * (y_n + 1 - y) + p2.y * (x - x_n) * (y_n + 1 - y) + p3.y * (x_n + 1 - x) * (y - y_n) + p4.y * (x - x_n) * (y - y_n)); // ((x_n + 1 - x_n) * (y_n + 1 - y_n));
			b = (p1.z * (x_n + 1 - x) * (y_n + 1 - y) + p2.z * (x - x_n) * (y_n + 1 - y) + p3.z * (x_n + 1 - x) * (y - y_n) + p4.z * (x - x_n) * (y - y_n)); // ((x_n + 1 - x_n) * (y_n + 1 - y_n));
			

			out[y * wn + x] = make_uchar4(r, g, b, p1.w);
		}
}

int main() {
	int w, h, wn, hn;
	string inFile;
	string outFile;

	std::cin >> inFile >> outFile >> wn >> hn;

	FILE* fp = fopen(inFile.c_str(), "rb");

	//FILE* fp = fopen("in.data", "rb");
	fread(&w, sizeof(int), 1, fp);
	fread(&h, sizeof(int), 1, fp);
	uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
	uchar4* data_ = (uchar4*)malloc(sizeof(uchar4) * wn * hn);
	fread(data, sizeof(uchar4), w * h, fp);
	fclose(fp);

	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?:)";
		//memfree
		return 0;
	}
	// Подготовка данных для текстуры
	cudaArray* arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	cudaStatus = cudaMallocArray(&arr, &ch, w, h);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc failed!";
		//MemFree(va, vb, vc, v1, v2, v3);
		return 0;
	}

	cudaStatus = cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpy failed!";
		//MemFree(va, vb, vc, v1, v2, v3);
		return 0;
	}

	// Подготовка текстурной ссылки, настройка интерфейса работы с данными
	tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
	tex.normalized = false;						// Режим нормализации координат: без нормализации

	// Связываем интерфейс с данными
	cudaStatus = cudaBindTextureToArray(tex, arr, ch);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaBindTexture failed!";
		//MemFree(va, vb, vc, v1, v2, v3);
		return 0;
	}

	uchar4* out;

	cudaStatus = cudaMalloc(&out, sizeof(uchar4) * wn * hn);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMalloc failed!";
		//MemFree(va, vb, vc, v1, v2, v3);
		return 0;
	}

	//bool f;

	bilinear_interp_kernel << <dim3(16, 16), dim3(16, 16) >> > (out, w, h, wn, hn);

	cudaMemcpy(data_, out, sizeof(uchar4) * wn * hn, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaMemcpy failed!";
		//MemFree(va, vb, vc, v1, v2, v3);
		return 0;
	}

	// Отвязываем данные от текстурной ссылки
	cudaStatus = cudaUnbindTexture(tex);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaUnBindTexture failed!";
		//MemFree(va, vb, vc, v1, v2, v3);
		return 0;
	}

	cudaStatus = cudaFreeArray(arr);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaFreeArray failed!";
		//MemFree(va, vb, vc, v1, v2, v3);
		return 1;
	}

	cudaStatus = cudaFree(out);
	if (cudaStatus != cudaSuccess) {
		cout << "cudaFree failed!";
		//MemFree(va, vb, vc, v1, v2, v3);
		return 0;
	}

	fp = fopen(outFile.c_str(), "wb");
	fwrite(&wn, sizeof(int), 1, fp);
	fwrite(&hn, sizeof(int), 1, fp);
	//data = (uchar4*)malloc(sizeof(uchar4) * wn * hn);
	fwrite(data_, sizeof(uchar4), wn * hn, fp);
	fclose(fp);

	free(data);
	free(data_);
	return 0;
}
