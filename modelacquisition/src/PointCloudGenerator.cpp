//
// Created by lucas on 1/28/18.
//

#include <chrono>
#include <mutex>
#include <Utils.h>
#include <MathUtils.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/fast_bilateral.h>
#include "PointCloudGenerator.h"

namespace ark {

    PointCloudGenerator::PointCloudGenerator(std::string strSettingsFile) {
        cv::FileStorage fSettings(strSettingsFile, cv::FileStorage::READ);

        fx_ = fSettings["Camera.fx"];
        fy_ = fSettings["Camera.fy"];
        cx_ = fSettings["Camera.cx"];
        cy_ = fSettings["Camera.cy"];
        width_ = fSettings["Camera.width"];
        height_ = fSettings["Camera.height"];
        depthfactor_ = fSettings["DepthMapFactor"];
        maxdepth_ = fSettings["MaxDepth"];

        mOctomap = new octomap::ColorOcTree(0.006);

        mKeyFrame.frameId = -1;
        mbRequestStop = false;
    }

    void PointCloudGenerator::Start() {
        mptRun = new std::thread(&PointCloudGenerator::Run, this);
    }

    void PointCloudGenerator::ShutDown() {

    }

    void PointCloudGenerator::RequestStop() {
        std::unique_lock<std::mutex> lock(mRequestStopMutex);
        mbRequestStop = true;
    }

    bool PointCloudGenerator::IsRunning() {
        std::unique_lock<std::mutex> lock(mRequestStopMutex);
        return mbRequestStop;
    }

    void PointCloudGenerator::Run() {
        ark::RGBDFrame currentKeyFrame;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mRequestStopMutex);
                if (mbRequestStop)
                    break;
            }


            {
                std::unique_lock<std::mutex> lock(mKeyFrameMutex);
                if (currentKeyFrame.frameId == mKeyFrame.frameId)
                    continue;
                mKeyFrame.imDepth.copyTo(currentKeyFrame.imDepth);
                mKeyFrame.imRGB.copyTo(currentKeyFrame.imRGB);
                mKeyFrame.mTcw.copyTo(currentKeyFrame.mTcw);
                currentKeyFrame.frameId = mKeyFrame.frameId;
            }

            cv::Mat Twc = mKeyFrame.mTcw.inv();

//            rmd::SE3<float> T_world_curr = convertPoseToRmd(Twc);

            Reproject(currentKeyFrame.imRGB, currentKeyFrame.imDepth, Twc);
        }
    }

    void PointCloudGenerator::Reproject(const cv::Mat &imRGB, const cv::Mat &imD, const cv::Mat &Twc) {
        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
        for (int y = 0; y < imD.rows; ++y) {
            for (int x = 0; x < imD.cols; ++x) {
                const cv::Vec3f f(x, y, imD.at<float>(y, x));
                if (imD.at<float>(y, x) > 0.0f && imD.at<float>(y, x) < maxdepth_) {
                    PointType p;
                    p.x = f[0];
                    p.y = f[1];
                    p.z = f[2];
                    const cv::Vec3b color = imRGB.at<cv::Vec3b>(y, x);
                    p.r = color[0];
                    p.g = color[1];
                    p.b = color[2];
                    cloud->push_back(p);
                } else {
                    PointType p;
                    p.x = NAN;
                    p.y = NAN;
                    p.z = NAN;
                    p.r = 0;
                    p.g = 0;
                    p.b = 0;
                    cloud->push_back(p);
                }
            }
        }

        cloud->width = 640;
        cloud->height = 480;

        if (!cloud->empty()) {
            pcl::PointCloud<PointType>::Ptr cloud_bilateral(new pcl::PointCloud<PointType>);

            pcl::FastBilateralFilter<PointType> filter;
            filter.setInputCloud(cloud);
            filter.setSigmaS(10);
            filter.setSigmaR(0.05);
            filter.applyFilter(*cloud_bilateral);

            pcl::PointCloud<PointType>::Ptr cloud_sor(new pcl::PointCloud<PointType>);
            pcl::StatisticalOutlierRemoval<PointType> sor;
            sor.setInputCloud(cloud_bilateral);
            sor.setMeanK(50);
            sor.setStddevMulThresh(0.1);
            sor.filter(*cloud_sor);

            cv::Matx33f Rwc;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    Rwc(i, j) = Twc.at<float>(i, j);

            cv::Vec3f twc;
            for (int i = 0; i < 3; ++i)
                twc[i] = Twc.at<float>(i, 3);

            pcl::PointCloud<PointType>::Ptr cloud_for_disp(new pcl::PointCloud<PointType>);
            for (auto p: cloud_sor->points) {
                if (!std::isnan(p.x)) {
                    const cv::Vec3f f((p.x - cx_) / fx_, (p.y - cy_) / fy_, 1.0f);
                    const cv::Vec3f xyz = transform(Rwc, twc, f * p.z);
                    PointType p_disp;
                    p_disp.x = xyz[0];
                    p_disp.y = xyz[1];
                    p_disp.z = xyz[2];
                    p_disp.r = p.r;
                    p_disp.g = p.g;
                    p_disp.b = p.b;
                    cloud_for_disp->push_back(p_disp);
                }
            }

            octomap::Pointcloud cloud_octo;
            for (auto p:cloud_for_disp->points)
                cloud_octo.push_back(p.x, p.y, p.z);

            mOctomap->insertPointCloud(cloud_octo,
                                       octomap::point3d(0, 0, 0));

            for (auto p:cloud_for_disp->points)
                mOctomap->integrateNodeColor(p.x, p.y, p.z, p.r, p.g, p.b);

        }
    }

    void PointCloudGenerator::SaveOccupancyGrid(std::string filename) {
        mOctomap->updateInnerOccupancy();
        mOctomap->write("tmp.ot");
    }

    void PointCloudGenerator::OnKeyFrameAvailable(const RGBDFrame &keyFrame) {
        if (mMapRGBDFrame.find(keyFrame.frameId) != mMapRGBDFrame.end())
            return;
        std::unique_lock<std::mutex> lock(mKeyFrameMutex);
        keyFrame.mTcw.copyTo(mKeyFrame.mTcw);
        keyFrame.imRGB.copyTo(mKeyFrame.imRGB);
        keyFrame.imDepth.copyTo(mKeyFrame.imDepth);

        mKeyFrame.frameId = keyFrame.frameId;
        mMapRGBDFrame[keyFrame.frameId] = ark::RGBDFrame();
    }

    void PointCloudGenerator::OnFrameAvailable(const RGBDFrame &frame) {
        std::cout << "OnFrameAvailable" << frame.frameId << std::endl;
    }

    void PointCloudGenerator::OnLoopClosureDetected() {
        std::cout << "LoopClosureDetected" << std::endl;
    }
}
