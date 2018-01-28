#pragma once
// OpenCV Libraries
#include "stdafx.h"
#include "version.h"

// OpenARK Libraries
#include "DepthCamera.h"

// RealSense SDK
#include <librealsense/rs.hpp>

namespace ark {
	class R200Camera : public DepthCamera {
	public: 
		R200Camera();
		~R200Camera();

		bool nextFrame();

		void update() override;
		/**
		* Returns true if an RGB image is available from this camera.
		* @return true if an RGB image is available from this camera.
		*/
		bool hasRGBImage() const;

		/**
		* Returns true if an infrared (IR) image is available from this camera.
		* @return true if an infrared (IR) image is available from this camera.
		*/
		bool hasIRImage() const;
		/**
		* Gracefully closes the R200 camera.
		*/
		void destroyInstance();

	private:
		/**
		* Initializat the camera
		*/
		void initCamera();
		void fillInZCoords();
		void fillInRGBImg();
		void initializeImages();
		static const int REAL_WID = 640, REAL_HI = 480;
		rs::context mCtx;
		rs::device * mpDev;
		rs::intrinsics mDepth_intrin;
		rs::intrinsics mColor_intrin;
		float mDepthScale;
	};
}