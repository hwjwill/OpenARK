#pragma once
#include "Object3D.h"

namespace ark {
	/**
	* Class defining the data structure and basic methods of objects
	* derived from multiple frames (a scene).
	*/
	class SceneObject : public Object3D {
	public:
		SceneObject();
		~SceneObject();
	};
}