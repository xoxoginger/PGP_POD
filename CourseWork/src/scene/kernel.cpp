#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../core/primitives/vec3.hpp"

int t = 0;
__global__
void kernel()
{
	vec3 a = { 1,2,3 };
	vec3 b = { -1,-2,-3 };
	vec3 c = a+b;
	printf("%f %f %f\n", c.x, c.y, c.z);
}

int main()
{
	kernel<<<1, 16>>>();
}

void update()
{
	t++;
}