#include "triangle.hpp"
#include <math.h>

//барицентрический тест
__host__ __device__
vec3 triangle::test(vec3 pos, vec3 dir)
{
	vec3 e1 = this->b - this->a;
	vec3 e2 = this->c - this->a;
	vec3 p = dir * e2;
	double div = dot(p, e1);

	if (fabs(div) < 1e-10)
	{
		return vec3::undefined;
	}

	vec3 t = pos - this->a;
	double u = dot(p, t) / div;

	if (u < 0.0 || u > 1.0)
	{
		return vec3::undefined;
	}

	vec3 q = t * e1;
	double v = dot(q, dir) / div;

	if (v < 0.0 || v + u > 1.0)
	{
		return vec3::undefined;
	}

	double ts = dot(q, e2) / div;

	if (ts < 0.0)
	{
		return vec3::undefined;
	}

	return dir * ts + pos;
}