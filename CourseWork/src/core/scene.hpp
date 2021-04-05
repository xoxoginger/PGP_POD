#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <map>
#include "objects/light.hpp"
#include "objects/mesh_object.hpp"
#include "objects/camera.hpp"
#include "primitives/material.hpp"

class scene
{
public:
	std::vector<scene_object*> children;
	std::vector<light*> lights;
	std::vector<mesh_object*> meshes;
	std::vector<camera*> cameras;
	std::map<int, material*> mat_map;

	void add_light(light* obj);
	void add_mesh(mesh_object* obj);
	void add_camera(camera* obj);
	void add_object(scene_object* obj);
};

#endif