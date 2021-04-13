#include "material.hpp"

uchar4 material::get_color(double u, double v)
{
	return {255,255,255,255};
}

void cudaMaterialMemCpyToDevice(material* dst, const material* src)
{
	//сначала копируем текстуру
	uchar4* cudaTexture = nullptr;
	material* temp = new material(*src);
	
	if (src->texture != nullptr)
	{
		cudaMalloc(&cudaTexture, src->tex_width * src->tex_height * sizeof(uchar4));
		cudaMemcpy(cudaTexture, src->texture, src->tex_width * src->tex_height * sizeof(uchar4), cudaMemcpyHostToDevice);
	}
	temp->texture = cudaTexture;

	cudaMemcpy(dst, temp, sizeof(material), cudaMemcpyHostToDevice);

	delete temp;
}
