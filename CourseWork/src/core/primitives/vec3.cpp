#include "vec3.hpp"
#include <math.h>

const vec3 vec3::undefined =
{
	NAN,
	NAN,
	NAN
};

vec3 operator+(const vec3& a, const vec3& b)
{
	return
	{
		a.x + b.x,
		a.y + b.y,
		a.z + b.z
	};
}

vec3 operator-(const vec3& a, const vec3& b)
{
	return
	{
		a.x - b.x,
		a.y - b.y,
		a.z - b.z
	};
}

vec3 operator*(const vec3& a, const double& b)
{
	return
	{
		a.y * b,
		a.z * b,
		a.x * b
	};
}

vec3 operator*(const double& b, const vec3& a)
{
	return
	{
		a.y * b,
		a.z * b,
		a.x * b
	};
}

vec3 operator*(const vec3& a, const vec3& b)
{
	return
	{
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}

double dot(vec3 a, vec3 b)
{
	return
		a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 vec3::norm()
{
	double l = sqrt(dot(*this, *this));

	return
	{
		x / l,
		y / l,
		z / l
	};
}

vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v)
{
	return
	{
		a.x * v.x + b.x * v.y + c.x * v.z,
		a.y * v.x + b.y * v.y + c.y * v.z,
		a.z * v.x + b.z * v.y + c.z * v.z
	};
}