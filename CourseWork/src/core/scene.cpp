#include "scene.hpp"

void scene::add_light(light* obj)
{
	obj->parent = this;
	this->children.push_back(obj);
	this->lights.push_back(obj);
}

void scene::add_mesh(mesh_object* obj)
{
	obj->parent = this;
	this->children.push_back(obj);
	this->meshes.push_back(obj);
}

void scene::add_camera(camera* obj)
{
	obj->parent = this;
	this->children.push_back(obj);
	this->cameras.push_back(obj);
}

void scene::add_object(scene_object* obj)
{
	obj->parent = this;
	this->children.push_back(obj);
}