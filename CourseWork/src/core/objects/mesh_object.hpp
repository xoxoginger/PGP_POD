#ifndef MESH_OBJECT_H
#define MESH_OBJECT_H

#include <vector>
#include "../primitives/triangle.hpp"
#include "scene_object.hpp"

class mesh_object : scene_object
{
public:
	std::vector<triangle> mesh;
};

#endif