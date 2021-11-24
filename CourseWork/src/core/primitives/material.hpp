#ifndef MATERIAL_H
#define MATERIAL_H

#include "cuda_runtime.h"
#include "vec3.hpp"

struct material
{
	int id;
	double reflection;
	double refraction;
	//не только цвет, но и текстуру если есть
	uchar4 get_color(double u, double v);
	uchar4 color;
	int tex_width;
	int tex_height;
	uchar4* texture;
};

void cudaMaterialMemCpyToDevice(material* dst, const material* src);

#endif