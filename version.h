#pragma once

#define RSSDK_ENABLED
#define OPENARK_CAMERA_TYPE "sr300"
//#define PMDSDK_ENABLED
//#define OPENARK_CAMERA_TYPE "pmd" 

// Remove to disable visualizations (if building as library)
#define DEMO

// Uncomment to enable debug code
// #define DEBUG

// Uncomment to enable plane detection (warning: disables some hand constraints)
// #define PLANE_ENABLED

namespace ark {
    // OpenARK version number
    static const char * VERSION = "0.9.0";
}
