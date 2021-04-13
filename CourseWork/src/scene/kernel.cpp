#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../core/primitives/vec3.hpp"
#include "../core/primitives/material.hpp"

int t = 0;
__global__
void kernel(material* mat)
{
	printf("%d %d %d\n", mat->tex_height, mat->tex_width, mat->texture[1].x);
}

int main()
{
	material* m = new material();
	m->color = {1,2,3,4};
	m->tex_height = 1;
	m->tex_width = 2;
	m->texture = new uchar4[2];
	m->texture[0] = {1,1,1,1};
	m->texture[1] = { 2,2,2,2 };
	material* devMat;
	cudaMalloc(&devMat, sizeof(material));
	//cudaMemcpy(devMat, m, sizeof(material), cudaMemcpyHostToDevice);
	cudaMaterialMemCpyToDevice(devMat, m);
	kernel << <1, 16 >> > (devMat);
}

void update()
{
	t++;
}