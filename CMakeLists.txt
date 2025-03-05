cmake_minimum_required(VERSION 3.10)
project(FeatureExtraction)

find_package(OpenCV REQUIRED)

add_executable(matching matching.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

target_include_directories(matching PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(matching PRIVATE ${OpenCV_LIBS})
