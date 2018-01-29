//
// Created by lucas on 1/28/18.
//

#ifndef OPENARK_SLAMSYSTEM_H
#define OPENARK_SLAMSYSTEM_H

#include <functional>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Utils.h"


namespace ark {

    typedef std::function<void(int)> KeyFrameAvailableHandler;
    typedef std::function<void(int)> FrameAvailableHandler;
    typedef std::function<void(void)> LoopClosureDetectedHandler;

    class SLAMSystem {
    public:

        virtual void PushFrame(const cv::Mat &imRGB,const cv::Mat &imDepth, const double &timestamp) = 0;

        virtual void Start() = 0;

        virtual void RequestStop() = 0;

        virtual void ShutDown() = 0;

        virtual bool IsRunning() = 0;

        virtual void SetKeyFrameAvailableHandler(KeyFrameAvailableHandler handler) {
            mKeyFrameAvailableHandler = std::move(handler);
        }

        virtual void SetFrameAvailableHandler(FrameAvailableHandler handler) {
            mFrameAvailableHandler = std::move(handler);
        }

        virtual void SetLoopClosureDetectedHandler(LoopClosureDetectedHandler handler) {
            mLoopClosureHandler = std::move(handler);
        }

        virtual ~SLAMSystem() = default;

    protected:
        KeyFrameAvailableHandler mKeyFrameAvailableHandler;
        FrameAvailableHandler mFrameAvailableHandler;
        LoopClosureDetectedHandler mLoopClosureHandler;
    };
}

#endif //OPENARK_SLAMSYSTEM_H
