#include "stdafx.h"
#include "version.h"
#include "R200Camera.h"
#include "Visualizer.h"

namespace ark {
	R200Camera::R200Camera() {
		X_DIMENSION = REAL_WID;
		Y_DIMENSION = REAL_HI;
		rs::log_to_console(rs::log_severity::warn);
		initCamera();
	}

	void R200Camera::initCamera() {
		if (mCtx.get_device_count() == 0) exit(EXIT_FAILURE);

		mpDev = mCtx.get_device(0);

		// Configure depth and color to run with the device's preferred settings
		mpDev->enable_stream(rs::stream::color, rs::preset::best_quality);
		mpDev->enable_stream(rs::stream::depth, rs::preset::largest_image);
		mpDev->start();
		
		mDepth_intrin = mpDev->get_stream_intrinsics(rs::stream::depth);
		mColor_intrin = mpDev->get_stream_intrinsics(rs::stream::color);
		mDepthScale = mpDev->get_depth_scale();

		depth_width = mDepth_intrin.width;
		depth_height = mDepth_intrin.height;
		rgb_width = mColor_intrin.width;
		rgb_height = mColor_intrin.height; // Can we confirm if rgb and depth dimension are always same?
	}

	R200Camera::~R200Camera() {}

	void R200Camera::update() {
		initializeImages();
		mpDev->wait_for_frames();
		fillInZCoords();
		fillInRGBImg();
	}

	void R200Camera::fillInZCoords() {
		uint16_t * depth_image = (uint16_t *)mpDev->get_frame_data(rs::stream::depth_aligned_to_rectified_color);
		// imD =  cv::Mat(mDepth_intrin.height, mDepth_intrin.width, CV_16SC1); // We may need to store depth map as well for SLAM

		// Populate xyzMap
		for (int dy = 0; dy < depth_height; ++dy) {
			for (int dx = 0; dx < depth_width; ++dx) {
				uint16_t depth_value = depth_image[dy * depth_width + dx];
				rs::float3 depth_point = mDepth_intrin.deproject({ (float)dx, (float)dy }, depth_value * mDepthScale);
				xyzMap.at<cv::Vec3f>(dy, dx) = cv::Vec3f(depth_point.x, depth_point.y, depth_point.z);
			}
		}
	}

	void R200Camera::fillInRGBImg() {
		uint8_t * color_image = (uint8_t *)mpDev->get_frame_data(rs::stream::rectified_color);
		std::memcpy(&rgbImage.datastart, color_image, rgb_height * rgb_width * 3 * sizeof(unsigned char));
	}

	void R200Camera::initializeImages() {
		xyzMap = cv::Mat(depth_height, depth_width, CV_32FC3);
		rgbImage = cv::Mat(rgb_height, rgb_width, CV_8UC3);
	}

	void R200Camera::destroyInstance() {
		mpDev->stop();
	}

	bool R200Camera::hasRGBImage() const {
		return true;
	}

	bool R200Camera::hasIRImage() const {
		return false;
	}
}
