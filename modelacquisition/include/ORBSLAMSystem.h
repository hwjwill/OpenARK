//
// Created by lucas on 1/28/18.
//

#ifndef OPENARK_ORBSLAMSYSTEM_H
#define OPENARK_ORBSLAMSYSTEM_H

#include "SLAMSystem.h"
#include <thread>
#include <opencv2/core/core.hpp>

#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"

namespace ORB_SLAM2{
    class Viewer;
    class FrameDrawer;
    class Map;
    class Tracking;
    class LocalMapping;
    class LoopClosing;
    class FrameSelector;
    class OccupancyGrid;
}

namespace ark {

    class ORBSLAMSystem : public SLAMSystem {

    public:
        // Input sensor
        enum eSensor{
            MONOCULAR=0,
            STEREO=1,
            RGBD=2
        };

    public:
        ORBSLAMSystem(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true);

        void PushFrame(const cv::Mat &imRGB,const cv::Mat &imDepth, const double &timestamp);

        void Start();

        void RequestStop();

        void ShutDown();

        bool IsRunning();

    public:

        void ActivateLocalizationMode();

        void DeactivateLocalizationMode();

        bool MapChanged();

        void Reset();

    private:
        //Location of vocabulary file and settings file
        string mStrVocFile;

        string mStrSettingsFile;

        // Input sensor
        eSensor mSensor;

        // ORB vocabulary used for place recognition and feature matching.
        ORB_SLAM2::ORBVocabulary* mpVocabulary;

        // KeyFrame database for place recognition (relocalization and loop detection).
        ORB_SLAM2::KeyFrameDatabase* mpKeyFrameDatabase;

        // Map structure that stores the pointers to all KeyFrames and MapPoints.
        ORB_SLAM2::Map* mpMap;

        // Tracker. It receives a frame and computes the associated camera pose.
        // It also decides when to insert a new keyframe, create some new MapPoints and
        // performs relocalization if tracking fails.
        ORB_SLAM2::Tracking* mpTracker;

        // Local Mapper. It manages the local map and performs local bundle adjustment.
        ORB_SLAM2::LocalMapping* mpLocalMapper;

        // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
        // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
        ORB_SLAM2::LoopClosing* mpLoopCloser;

        // The viewer draws the map and the current camera pose. It uses Pangolin.
        ORB_SLAM2::Viewer* mpViewer;

        // OccupancyGrid Generator
        ORB_SLAM2::OccupancyGrid *mpOccupancyGrid;

        ORB_SLAM2::FrameDrawer* mpFrameDrawer;
        ORB_SLAM2::MapDrawer* mpMapDrawer;

        // System threads: Local Mapping, Loop Closing, Viewer.
        // The Tracking thread "lives" in the main execution thread that creates the System object.
        std::thread* mptLocalMapping;
        std::thread* mptLoopClosing;
        std::thread* mptViewer;
        std::thread* mptOccupancyGrid;

        // Reset flag
        std::mutex mMutexReset;
        bool mbReset;

        // Runing flag
        std::mutex mMutexRequestStop;
        bool mbRequestStop;

        // Change mode flags
        std::mutex mMutexMode;
        bool mbActivateLocalizationMode;
        bool mbDeactivateLocalizationMode;

        // Viewer mode flags;
        bool mbUseViewer;

        // Tracking state
        int mTrackingState;
        std::vector<ORB_SLAM2::MapPoint*> mTrackedMapPoints;
        std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
        std::mutex mMutexState;
    };
}

#endif //OPENARK_ORBSLAMSYSTEM_H
