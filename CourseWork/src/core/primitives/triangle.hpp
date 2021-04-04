#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "cuda_runtime.h"
#include "vec3.hpp"

struct triangle
{
	vec3 a;
	vec3 b;
	vec3 c;
	__host__ __device__
	vec3 test(vec3 pos, vec3 dir);
};

#endif