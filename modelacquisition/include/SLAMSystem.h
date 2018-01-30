//
// Created by Kuan Lu on 1/28/18.
//

#ifndef OPENARK_SLAMSYSTEM_H
#define OPENARK_SLAMSYSTEM_H

#include <functional>
#include <iostream>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "Utils.h"


namespace ark {

    typedef std::function<void(const RGBDFrame &)> KeyFrameAvailableHandler;
    typedef std::function<void(const RGBDFrame &)> FrameAvailableHandler;
    typedef std::function<void(void)> LoopClosureDetectedHandler;
    typedef std::unordered_map<std::string, KeyFrameAvailableHandler> MapKeyFrameAvailableHandler;
    typedef std::unordered_map<std::string, FrameAvailableHandler> MapFrameAvailableHandler;
    typedef std::unordered_map<std::string, LoopClosureDetectedHandler> MapLoopClosureDetectedHandler;

    class SLAMSystem {
    public:

        virtual void PushFrame(const cv::Mat &imRGB,const cv::Mat &imDepth, const double &timestamp) = 0;

        virtual void Start() = 0;

        virtual void RequestStop() = 0;

        virtual void ShutDown() = 0;

        virtual bool IsRunning() = 0;

        virtual void AddKeyFrameAvailableHandler(KeyFrameAvailableHandler handler, std::string handlerName) {
            mMapKeyFrameAvailableHandler[handlerName] = handler;
        }

        virtual void RemoveKeyFrameAvailableHandler(std::string handlerName) {
            auto handler = mMapKeyFrameAvailableHandler.find(handlerName);
            if (handler != mMapKeyFrameAvailableHandler.end())
                mMapKeyFrameAvailableHandler.erase(handler);
        }

        virtual void AddFrameAvailableHandler(FrameAvailableHandler handler, std::string handlerName) {
            mMapFrameAvailableHandler[handlerName] = handler;
        }

        virtual void RemoveFrameAvailableHandler(std::string handlerName) {
            auto handler = mMapFrameAvailableHandler.find(handlerName);
            if (handler != mMapFrameAvailableHandler.end())
                mMapFrameAvailableHandler.erase(handler);
        }

        virtual void AddLoopClosureDetectedHandler(LoopClosureDetectedHandler handler, std::string handlerName) {
            mMapLoopClosureHandler[handlerName] = handler;
        }

        virtual void RemoveLoopClosureDetectedHandler(std::string handlerName) {
            auto handler = mMapLoopClosureHandler.find(handlerName);
            if (handler != mMapLoopClosureHandler.end())
                mMapLoopClosureHandler.erase(handler);
        }

        virtual ~SLAMSystem() = default;

    protected:
        MapKeyFrameAvailableHandler mMapKeyFrameAvailableHandler;
        MapFrameAvailableHandler mMapFrameAvailableHandler;
        MapLoopClosureDetectedHandler mMapLoopClosureHandler;
    };
}

#endif //OPENARK_SLAMSYSTEM_H
