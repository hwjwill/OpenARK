# OpenARK - 3D Scanning Module

OpenARK is an open-source wearable augmented reality (AR) system founded at UC Berkeley in 2016. The C++ based software offers innovative core functionalities to power a wide range of off-the-shelf AR components, including see-through glasses, depth cameras, and IMUs. The open-source platform includes fundamental tools such as AR-based camera calibration and SLAM, and it also includes higher-level functions to aid human-computer interaction, such as 3D gesture recognition and multi-user collaboration. In addition, it also supports real time scene reconstruction and mesh simplification. Currently, it supports both PMD Pico Flexx and Intel RealSense SR300 cameras. OpenARK currently only supports Windows and we have tested our platform with Windows 10 and Visual Studio 2015 Community Edition.

At a Glance

  - **Technology stack**: C++, OpenCV, PCL, Boost, RealSense 3D SDK, Kinect SDK, CGL
## Support
Plase talk to Kuan (kuan_lu@berkeley.edu) for support on this section

## Dependencies
Hardware
- Depth Camera

Software
- CMake 3.8 or above
- OpenCV 3.0 or above
- PCL 1.8
- Boost 1.6.4

## Platform
- OS: Ubuntu 16.04.3 LTS
- Kernel version: Linux 4.10.0-28-generic x86_64
Ensure that the linux kernel version is under 4.13.0, higher kernel version may cause failure to login after installing NVIDIA driver, kernel version can be check by:
```sh
 uname -mrs
 ```
Installing 4.10.0-28-generic x86_64 can be executed via:
```sh
 sudo apt-get install linux-image-4.10.0-28-generic
 ```
If your ubuntu 16.04.3 comes with a kernel version different than `linux-image-4.10.0-28-generic`, uninstall that kernel via:
```sh
sudo apt-get purge linux-image-x.x.x-x-generic
```
or select `linux-image-4.10.0-28-generic` in the grub menu every time you boot


## Install CMake
```sh
sudo apt-get install cmake
```

### Installing OpenCV 3

1. Install prerequisites:

```sh
sudo apt-get install -y libopencv-dev libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev
sudo apt-get install -y libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev
sudo apt-get install -y libvorbis-dev libxvidcore-dev v4l-utils vtk6
sudo apt-get install -y liblapacke-dev libopenblas-dev libgdal-dev checkinstall
sudo apt-get install -y libssl-dev
```

2. Download the OpenCV sources (Minimum version 3.2) from their website: <https://opencv.org/releases.html>

3. Unpack the source zip using `unzip`. 

4. Enter the extracted directory and enter the following:

``` sh
mkdir build
cd build/
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D FORCE_VTK=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -D CUDA_NVCC_FLAGS="-D_FORCE_INLINES" -D WITH_GDAL=ON -D WITH_XINE=ON -D BUILD_EXAMPLES=ON ..
make -j4
```

You may replace '4' in the last line with any number of threads. The build process should not take too long.

5. `sudo make install` to install. You may be prompted for your password.

(Credits to: [https://github.com/BVLC/caffe/wiki/OpenCV-3.3-Installation-Guide-on-Ubuntu-16.04])


### Installing PCL
Referenced from <https://askubuntu.com/questions/916260/how-to-install-point-cloud-library-v1-8-pcl-1-8-0-on-ubuntu-16-04-2-lts-for?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa>

Install universal pre-requisites:
```sh
sudo apt -y install g++ cmake cmake-gui doxygen mpi-default-dev openmpi-bin openmpi-common libusb-1.0-0-dev libqhull* libusb-dev libgtest-dev
sudo apt -y install git-core freeglut3-dev pkg-config build-essential libxmu-dev libxi-dev libphonon-dev libphonon-dev phonon-backend-gstreamer
sudo apt -y install phonon-backend-vlc graphviz mono-complete qt-sdk libflann-dev 
```
For PCL v1.8, Ubuntu 16.04.2 input the following:
```sh
sudo apt -y install libflann1.8 libboost1.58-all-dev

cd ~/Downloads
wget http://launchpadlibrarian.net/209530212/libeigen3-dev_3.2.5-4_all.deb
sudo dpkg -i libeigen3-dev_3.2.5-4_all.deb
sudo apt-mark hold libeigen3-dev

wget http://www.vtk.org/files/release/7.1/VTK-7.1.0.tar.gz
tar -xf VTK-7.1.0.tar.gz
cd VTK-7.1.0 && mkdir build && cd build
cmake ..
make                                                                   
sudo make install

cd ~/Downloads
wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.0.tar.gz
tar -xf pcl-1.8.0.tar.gz
cd pcl-pcl-1.8.0 && mkdir build && cd build
cmake ..
make
sudo make install

cd ~/Downloads
rm libeigen3-dev_3.2.5-4_all.deb VTK-7.1.0.tar.gz pcl-1.8.0.tar.gz
sudo rm -r VTK-7.1.0 pcl-pcl-1.8.0
```
#### Validation
```sh
cd ~
mkdir pcl-test && cd pcl-test
```
Create a CMakeLists.txt file:
```
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(pcl-test)
find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcl-test main.cpp)
target_link_libraries(pcl-test ${PCL_LIBRARIES})

SET(COMPILE_FLAGS "-std=c++11")
add_definitions(${COMPILE_FLAGS})
```
Create a main.cpp file:
```cpp
#include <iostream>

int main() {
    std::cout << "hello, world!" << std::endl;
    return (0);
}
```
Compile:
```sh
mkdir build && cd build
cmake ..
make
```
Test:
```sh
./pcl-test
```
Output ->
```
hello, world!
```

### Install Pangolin
1. follow the installation guide here <https://github.com/stevenlovegrove/Pangolin>, need version 0.5

### Install CUDA version 8.0
A simple and clear install instruction of installing NVIDIA driver and CUDA can be found here <https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07#common-errors-and-solutions>. 

I personally suggest installing both NVIDIA driver and CUDA via runfile, which can be downloaded from NVIDIA website.

For ubuntu 16.04, the following version has been tested working:

NVIDIA Driver 384.90: <https://www.nvidia.com/download/driverResults.aspx/123918/en-us>

CUDA 8.0: <https://developer.nvidia.com/cuda-80-ga2-download-archive>

### Install librealsense v1.12.1
Use the legacy librealsense driver for ZR300 and R200 support <https://github.com/IntelRealSense/librealsense/releases/tag/v1.12.1>

Follow the installation guide here <https://github.com/IntelRealSense/librealsense/blob/v1.12.1/doc/installation.md>

### Install libfreenect version 0.5.7
If you want building the kinect examples, we used libfreenect <https://github.com/OpenKinect/libfreenect/tree/v0.5.7>, the installation guide is inside

## Build
Building can be done via build.sh inside the `OpenARK/modelacquisition` folder
```sh
cd modelacquisition
./build.sh
```
If anything goes wrong, or you want to specify the cmake configuration, you can manually build the project.
First we build the third party dependencies inside the `OpenARK/modelacquisition/Thirdparty`

1. Build DBow2
```sh
cd modelacquisition/Thirdparty
cd DBoW2
mkdir build && cd build
cmake ../
make -j8
```
2. Build g2o
```sh
cd modelacquisition/g2o
cd g2o
mkdir build && cd build
cmake ../
make -j8
```

3. Build TSDF
```sh
cd modelacquisition/TSDF
cd TSDF
mkdir build && cd build
cmake ../
make -j8
```

Then we build the library
```sh
cd modelacquisition
mkdir build && cd build
cmake ../
make -j8
```


# Colorizer
Colorizer is a separate build that takes input of a dense PLY mesh and simplify to 1% of original size, and apply texture mapping to reduce the size while retain the same visual quality.

## Support
Plase talk to Will (hwjwill@berkeley.edu) for support on this section

## Dependency
- PCL 1.8
- VTK
- FLANN
- TinyPly <https://github.com/ddiakopoulos/tinyply>, already included in this repo
- CGL, already included in this repo

## Build
```sh
cd colorizer/source
mkdir build && cd build
cmake ../..
make -j4
```

## To Use
```sh
./colorizer [Input model path] [texture resolution]
```
`Input model path` is the path to model generated in previous module. It will be a dense 3D mesh model with color information on each vertex.

Texture generated by colorizer will be a squre 2D picture, and its `texture resolution` parameter will be number of desired pixels for the texture map
