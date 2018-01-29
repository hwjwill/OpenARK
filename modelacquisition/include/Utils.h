//
// Created by lucas on 1/28/18.
//

#ifndef OPENARK_UTILS_H
#define OPENARK_UTILS_H

#include <opencv2/opencv.hpp>

namespace ark{
    typedef struct {
        cv::Mat mTcw;
        cv::Mat imRGB;
        cv::Mat imDepth;
        int frameId;
    } RGBDFrame;
}


#endif //OPENARK_UTILS_H
