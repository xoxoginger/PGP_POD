#ifndef SCENE_OBJECT_H
#define SCENE_OBJECT_H

#include <functional>
#include "../primitives/triangle.hpp"
#include "../scene.hpp"

class scene_object
{
	std::function<void(int)> m_update;
	scene m_parent;
public:
	scene get_parent();
	vec3 pos;
	scene_object(vec3 pos, std::function<void(int)> update);
};

#endif