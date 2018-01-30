//
// Created by yang on 16-12-9.
//

/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <librealsense/rs.hpp>
#include <opencv2/core/core.hpp>

#include <ORBSLAMSystem.h>
#include <BridgeRSR200.h>
#include <PointCloudGenerator.h>

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << endl << "Usage: ./rgbd_realsense path_to_vocabulary path_to_settings" << endl;
        return 1;
    }


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ark::PointCloudGenerator pointCloudGenerator(argv[2]);
    ark::ORBSLAMSystem slam(argv[1], argv[2], ark::ORBSLAMSystem::RGBD, true);
    BridgeRSR200 bridgeRSR200;

    slam.AddKeyFrameAvailableHandler([&pointCloudGenerator](const ark::RGBDFrame& keyFrame){return pointCloudGenerator.OnKeyFrameAvailable(keyFrame);}, "PointCloudFusion");

    slam.Start();
    pointCloudGenerator.Start();
    bridgeRSR200.Start();

    // Main loop
    int tframe = 1;

    while (!slam.IsRunning()) {
        cv::Mat imRGB, imD;

        bridgeRSR200.GrabRGBDPair(imRGB, imD);

        // Pass the image to the SLAM system
        slam.PushFrame(imRGB, imD, tframe);

        // check map changed
        if (slam.MapChanged()) {
            std::cout << "map changed" << std::endl;
        }

        if(tframe++==1000)
        {
            slam.RequestStop();
            pointCloudGenerator.RequestStop();
        }
    }

    pointCloudGenerator.SaveOccupancyGrid("tmp.pcd");

    slam.ShutDown();
    pointCloudGenerator.ShutDown();
}