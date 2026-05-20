cmake_minimum_required(VERSION 3.16)
project(cpp_static_pose_optimizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED core imgproc calib3d highgui imgcodecs)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(static_pose_optimizer_example
    src/main.cpp
    src/StaticPoseOptimizer.cpp
)

target_link_libraries(static_pose_optimizer_example PRIVATE ${OpenCV_LIBS})
