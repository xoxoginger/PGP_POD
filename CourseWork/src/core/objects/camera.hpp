#ifndef CAMERA_H
#define CAMERA_H

#include "../render/renderer.hpp"
#include "scene_object.hpp"

class camera : public scene_object
{
	renderer m_renderer;
public:
	vec3 view_dir;
	uchar4* getFrame();
};

#endif