cmake_minimum_required(VERSION 3.10)
project(FeatureExtraction)
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
find_package(OpenCV REQUIRED)

add_executable(feature_extraction feature_extraction.cpp)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

target_include_directories(feature_extraction PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(feature_extraction PRIVATE ${OpenCV_LIBS})
