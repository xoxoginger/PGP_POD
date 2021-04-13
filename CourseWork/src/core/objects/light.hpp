#ifndef LIGHT_H
#define LIGHT_H

#include "scene_object.hpp"

class light : public scene_object
{
public:
	uchar4 color;
};

#endif