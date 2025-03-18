cmake_minimum_required(VERSION 3.22)
project(main_class)

set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main
        RelativePoseUpdate.cpp)

target_link_libraries(main ${OpenCV_LIBS})
