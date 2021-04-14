#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>

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

// текстурная ссылка <тип элементов, размерность, режим нормализации>
texture<uchar4, 2, cudaReadModeElementType> tex;

__global__ void bilinear_interp_kernel(uchar4* out, int w, int h, int wn, int hn)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;
	int x, y;
	int x1, y1;
	uchar4 p1, p2, p3, p4;
	double r, g, b;
	double x2, y2;
	for (y = idy; y < hn; y += offsety)
		for (x = idx; x < wn; x += offsetx)
		{
			double h_d = (double)h / hn;
			double w_d = (double)w / wn;

			double x_n = (x + 0.5) * w_d - 0.5;
			double y_n = (y + 0.5) * h_d - 0.5;

			x1 = x_n; //x_r
			x2 = x_n - x1;//ceil(x_n);
			if (x2 < 0.0) //////////
			{
				x2++;
				x1--;
			}
			y1 = y_n;
			y2 = y_n - y1; //ceil(y_n);
			if (y2 < 0.0) /////////
			{ 
				y2++;
				y1--;
			}
				
			//printf("%f %f %f %f \n", x1, x2, y1, y2); +
			
		    p1 = tex2D(tex, x1, y1);
			p2 = tex2D(tex, x1 + 1, y1); 
			p3 = tex2D(tex, x1, y1 + 1); 
			p4 = tex2D(tex, x1 + 1, y1 + 1); 

			//r = (p1.x * (x2 - x_n) * (y2 - y_n) + p2.x * (x_n - x1) * (y2 - y_n) + p3.x * (x2 - x_n) * (y_n - y1) + p4.x * (x_n - x1) * (y_n - y1)) / ((x2 - x1) * (y2 - y1));
			//g = (p1.y * (x2 - x_n) * (y2 - y_n) + p2.y * (x_n - x1) * (y2 - y_n) + p3.y * (x2 - x_n) * (y_n - y1) + p4.y * (x_n - x1) * (y_n - y1)) / ((x2 - x1) * (y2 - y1));
			//b = (p1.z * (x2 - x_n) * (y2 - y_n) + p2.z * (x_n - x1) * (y2 - y_n) + p3.z * (x2 - x_n) * (y_n - y1) + p4.z * (x_n - x1) * (y_n - y1)) / ((x2 - x1) * (y2 - y1));

			r = p1.x * (1.0 - x2) * (1.0 - y2) + p2.x * x2 * (1.0 - y2) + p3.x * (1.0 - x2) * y2 + p4.x * x2 * y2;
			g = p1.y * (1.0 - x2) * (1.0 - y2) + p2.y * x2 * (1.0 - y2) + p3.y * (1.0 - x2) * y2 + p4.y * x2 * y2;
			b = p1.z * (1.0 - x2) * (1.0 - y2) + p2.z * x2 * (1.0 - y2) + p3.z * (1.0 - x2) * y2 + p4.z * x2 * y2;
			
			//if (r > 255)
			//{
				//printf("SHIT\n");
			//}

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

	CSC(cudaSetDevice(0));

	// Подготовка данных для текстуры
	cudaArray* arr;
	cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));

	CSC(cudaMemcpyToArray(arr, 0, 0, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

	// Подготовка текстурной ссылки, настройка интерфейса работы с данными
	tex.addressMode[0] = cudaAddressModeClamp;	// Политика обработки выхода за границы по каждому измерению
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;		// Без интерполяции при обращении по дробным координатам
	tex.normalized = false;						// Режим нормализации координат: без нормализации

	// Связываем интерфейс с данными
	CSC(cudaBindTextureToArray(tex, arr, ch));

	uchar4* out;

	CSC(cudaMalloc(&out, sizeof(uchar4) * wn * hn));

	bilinear_interp_kernel << <dim3(16, 16), dim3(16, 16) >> > (out, w, h, wn, hn);

	CSC(cudaMemcpy(data_, out, sizeof(uchar4) * wn * hn, cudaMemcpyDeviceToHost));

	// Отвязываем данные от текстурной ссылки
	CSC(cudaUnbindTexture(tex));

	CSC(cudaFreeArray(arr));

	CSC(cudaFree(out));

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
