//
// Created by lucas on 1/28/18.
//

#include "ORBSLAMSystem.h"

namespace ark {

    ORBSLAMSystem::ORBSLAMSystem(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
                                 const bool bUseViewer) :
            mStrVocFile(strVocFile),
            mStrSettingsFile(strSettingsFile),
            mSensor(sensor),
            mbUseViewer(bUseViewer),
            mpViewer(static_cast<ORB_SLAM2::Viewer *>(NULL)),
            mbReset(false), mbActivateLocalizationMode(false),
            mbDeactivateLocalizationMode(false) {
    }

    void ORBSLAMSystem::Start() {

        // Output welcome message
        cout << endl <<
             "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
             "This program comes with ABSOLUTELY NO WARRANTY;" << endl <<
             "This is free software, and you are welcome to redistribute it" << endl <<
             "under certain conditions. See LICENSE.txt." << endl << endl;

        cout << "Input sensor was set to: ";

        if (mSensor == MONOCULAR)
            cout << "Monocular" << endl;
        else if (mSensor == STEREO)
            cout << "Stereo" << endl;
        else if (mSensor == RGBD)
            cout << "RGB-D" << endl;

        //Check settings file
        cv::FileStorage fsSettings(mStrSettingsFile, cv::FileStorage::READ);
        if (!fsSettings.isOpened()) {
            cerr << "Failed to open settings file at: " << mStrSettingsFile << endl;
            exit(-1);
        }


        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

        mpVocabulary = new ORB_SLAM2::ORBVocabulary();
        bool bVocLoad = mpVocabulary->loadFromTextFile(mStrVocFile);
        if (!bVocLoad) {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << mStrVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        //Create KeyFrame Database
        mpKeyFrameDatabase = new ORB_SLAM2::KeyFrameDatabase(*mpVocabulary);

        //Create the Map
        mpMap = new ORB_SLAM2::Map();

        //Create Drawers. These are used by the Viewer
        mpFrameDrawer = new ORB_SLAM2::FrameDrawer(mpMap);
        mpMapDrawer = new ORB_SLAM2::MapDrawer(mpMap, mStrSettingsFile);

        //Initialize the Tracking thread
        //(it will live in the main thread of execution, the one that called this constructor)
        mpTracker = new ORB_SLAM2::Tracking(this, mMapKeyFrameAvailableHandler, mpVocabulary,
                                            mpFrameDrawer, mpMapDrawer, mpMap, mpKeyFrameDatabase, mStrSettingsFile, mSensor);

        //Initialize the Local Mapping thread and launch
        mpLocalMapper = new ORB_SLAM2::LocalMapping(mpMap, mSensor == MONOCULAR);
        mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);

        //Initialize the Loop Closing thread and launch
        mpLoopCloser = new ORB_SLAM2::LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor != MONOCULAR);
        mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);

        //Initialize the Viewer thread and launch
        if (mbUseViewer) {
            mpViewer = new ORB_SLAM2::Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker,
                                             mStrSettingsFile);
            mptViewer = new thread(&ORB_SLAM2::Viewer::Run, mpViewer);
            mpTracker->SetViewer(mpViewer);
        }

        //Set pointers between threads
        mpTracker->SetLocalMapper(mpLocalMapper);
        mpTracker->SetLoopClosing(mpLoopCloser);

        mpLocalMapper->SetTracker(mpTracker);
        mpLocalMapper->SetLoopCloser(mpLoopCloser);

        mpLoopCloser->SetTracker(mpTracker);
        mpLoopCloser->SetLocalMapper(mpLocalMapper);

        {
            unique_lock<mutex> lock(mMutexRequestStop);
            mbRequestStop = false;
        }
    }

    void ORBSLAMSystem::ShutDown() {
        mpLocalMapper->RequestFinish();
        mpLoopCloser->RequestFinish();

        if (mpViewer) {
            mpViewer->RequestFinish();
            while (!mpViewer->isFinished())
                usleep(5000);
        }

        // Wait until all thread have effectively stopped
        while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA()) {
            usleep(5000);
        }

        if (mpViewer)
            pangolin::BindToContext("ORB-SLAM2: Map Viewer");
    }

    void ORBSLAMSystem::RequestStop() {
        unique_lock<mutex> lock(mMutexRequestStop);
        mbRequestStop = true;
    }

    bool ORBSLAMSystem::IsRunning() {
        unique_lock<mutex> lock(mMutexRequestStop);
        return mbRequestStop;
    }

    void ORBSLAMSystem::PushFrame(const cv::Mat &imRGB, const cv::Mat &imDepth, const double &timestamp) {
        if (mSensor != RGBD) {
            cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
            exit(-1);
        }

        // Check mode change
        {
            unique_lock<mutex> lock(mMutexMode);
            if (mbActivateLocalizationMode) {
                mpLocalMapper->RequestStop();

                // Wait until Local Mapping has effectively stopped
                while (!mpLocalMapper->isStopped()) {
                    usleep(1000);
                }

                mpTracker->InformOnlyTracking(true);
                mbActivateLocalizationMode = false;
            }
            if (mbDeactivateLocalizationMode) {
                mpTracker->InformOnlyTracking(false);
                mpLocalMapper->Release();
                mbDeactivateLocalizationMode = false;
            }
        }

        // Check reset
        {
            unique_lock<mutex> lock(mMutexReset);
            if (mbReset) {
                mpTracker->Reset();
                mbReset = false;
            }
        }

        cv::Mat Tcw = mpTracker->GrabImageRGBD(imRGB, imDepth, timestamp);

        unique_lock<mutex> lock2(mMutexState);
        mTrackingState = mpTracker->mState;
        mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
        mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
        //return Tcw;
    }

    void ORBSLAMSystem::ActivateLocalizationMode() {
        unique_lock<mutex> lock(mMutexMode);
        mbActivateLocalizationMode = true;
    }

    void ORBSLAMSystem::DeactivateLocalizationMode() {
        unique_lock<mutex> lock(mMutexMode);
        mbDeactivateLocalizationMode = true;
    }

    bool ORBSLAMSystem::MapChanged() {
        static int n = 0;
        int curn = mpMap->GetLastBigChangeIdx();
        if (n < curn) {
            n = curn;
            return true;
        } else
            return false;
    }

    void ORBSLAMSystem::Reset() {
        unique_lock<mutex> lock(mMutexReset);
        mbReset = true;
    }

}