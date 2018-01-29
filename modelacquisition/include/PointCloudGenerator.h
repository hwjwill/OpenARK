//
// Created by lucas on 1/28/18.
//

#ifndef OPENARK_POINTCLOUDGENERATOR_H
#define OPENARK_POINTCLOUDGENERATOR_H

#include "Utils.h"

namespace ark{
    class PointCloudGenerator{
    public:
        PointCloudGenerator(){

        }
        void OnKeyFrameAvailable(int i) {
            std::cout << "OnKeyFrameAvailable " << i <<std::endl;
        }

        void OnFrameAvailable(int i) {
            std::cout << "OnFrameAvailable " << i <<std::endl;
        }

        void OnLoopClosureDetected() {
            std::cout << "OnLoopClosureDetected " <<std::endl;
        }
    };
}

#endif //OPENARK_POINTCLOUDGENERATOR_H
