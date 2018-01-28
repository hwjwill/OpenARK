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

		intrinsics = cv::Mat::zeros(3, 3, CV_32FC1);
		intrinsics.at<float>(0, 0) = mColor_intrin.fx;
		intrinsics.at<float>(0, 1) = 0.0f; //Wrong, need change to s
		intrinsics.at<float>(1, 1) = mColor_intrin.fy;
		intrinsics.at<float>(0, 2) = mColor_intrin.ppx;
		intrinsics.at<float>(1, 2) = mColor_intrin.ppy;
		intrinsics.at<float>(2, 2) = 1.0f;
	}

	R200Camera::~R200Camera() {}

	bool R200Camera::nextFrame() {
		update();
		return true;
	}

	void R200Camera::update() {
		initializeImages();
		mpDev->wait_for_frames();
		fillInZCoords();
		fillInRGBImg();
	}

	void R200Camera::fillInZCoords() {
		uint16_t * depth_image = (uint16_t *)mpDev->get_frame_data(rs::stream::depth_aligned_to_rectified_color);
		std::memcpy(&depthMap.datastart, depth_image, Y_DIMENSION * X_DIMENSION * sizeof(short));
		
		// Populate xyzMap
		for (int dy = 0; dy < Y_DIMENSION; ++dy) {
			for (int dx = 0; dx < X_DIMENSION; ++dx) {
				uint16_t depth_value = depth_image[dy * X_DIMENSION + dx];
				rs::float3 depth_point = mColor_intrin.deproject({ (float)dx, (float)dy }, depth_value * mDepthScale); // May have issue with mDepthScale
				xyzMap.at<cv::Vec3f>(dy, dx) = cv::Vec3f(depth_point.x, depth_point.y, depth_point.z);
			}
		}
	}

	void R200Camera::fillInRGBImg() {
		uint8_t * color_image = (uint8_t *)mpDev->get_frame_data(rs::stream::rectified_color);
		std::memcpy(&rgbImage.datastart, color_image, Y_DIMENSION * X_DIMENSION * 3 * sizeof(unsigned char));
	}

	void R200Camera::initializeImages() {
		xyzMap = cv::Mat(Y_DIMENSION, X_DIMENSION, CV_32FC3);
		rgbImage = cv::Mat(Y_DIMENSION, X_DIMENSION, CV_8UC3);
		depthMap = cv::Mat(Y_DIMENSION, X_DIMENSION, CV_16SC1);
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
