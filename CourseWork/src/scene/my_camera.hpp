#ifndef MY_CAMERA_H
#define MY_CAMERA_H

#include "../core/objects/camera.hpp"

class my_camera : camera
{
public:
	void update(int t)
	{
		this->pos.y = sin(t);
	}
};

#endif