#ifndef SCENE_OBJECT_H
#define SCENE_OBJECT_H

#include <vector>
#include "../primitives/triangle.hpp"
#include "../scene.hpp"

class scene_object
{
public:
	vec3 pos;
	void* parent;

	//вызывается на GPU или CPU в зависимости от выбранного режима
	__host__ __device__
	virtual void update(int t);
	scene_object(vec3 pos);
};

#endif