#pragma once
#include "Object3D.h"

namespace ark {
	/**
	* Class defining the data structure and basic methods of objects
	* derived from single frame.
	*/
	class FrameObject : public Object3D {
	public:
		FrameObject();
		~FrameObject();
	};
}