//
// Created by lucas on 1/28/18.
//

#ifndef OPENARK_POINTCLOUDGENERATOR_H
#define OPENARK_POINTCLOUDGENERATOR_H

#include "Utils.h"
#include <opencv2/opencv.hpp>
#include <mutex>
#include <thread>
#include <pcl/io/pcd_io.h>
#include "SE3.h"
#include "Octree.h"

namespace ark{

    class PointCloudGenerator{
    public:
        PointCloudGenerator(std::string strSettingsFile);

        void Start();

        void RequestStop();

        void ShutDown();

        bool IsRunning();

        void OnKeyFrameAvailable(const RGBDFrame &keyFrame);

        void OnFrameAvailable(const RGBDFrame &frame);

        void OnLoopClosureDetected();

        void Run();

        void SavePointCloud(std::string filename);

    private:

        void Reproject(cv::Mat &imRGB, cv::Mat &imD, rmd::SE3<float> &T_world_ref);

        //Main Loop thread
        std::thread *mptRun;

        //Octree of Global cloud
        Octree mOctree;

        //Current KeyFrame
        std::mutex mKeyFrameMutex;
        ark::RGBDFrame mKeyFrame;

        //Current Frame
        std::mutex mFrameMutex;
        ark::RGBDFrame mFrame;

        //Request Stop Status
        std::mutex mRequestStopMutex;
        bool mbRequestStop;

        //Camera params
        float fx_, fy_, cx_, cy_;
        float maxdepth_;
        int width_, height_;
        float depthfactor_;
    };
}

#endif //OPENARK_POINTCLOUDGENERATOR_H
