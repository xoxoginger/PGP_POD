#ifndef VEC3_H
#define VEC3_H

#include "cuda_runtime.h"

struct vec3
{
	double x;
	double y;
	double z;

	__device__ __host__ vec3 norm();
	static const vec3 undefined;
};

__device__ __host__
vec3 operator+(const vec3& a, const vec3& b);
vec3 operator-(const vec3& a, const vec3& b);
vec3 operator*(const vec3& a, const vec3& b); // prod
vec3 operator*(const vec3& a, const double& b);
vec3 operator*(const double& a, const vec3& b);
double dot(vec3 a, vec3 b);

#endif