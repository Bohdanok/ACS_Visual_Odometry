cmake_minimum_required(VERSION 3.10)
project(FeatureExtraction)
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 20)

#add_executable(corner_detection corner_detection.cpp corner_detection.h)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
#
##target_include_directories(corner_detection PRIVATE ${OpenCV_INCLUDE_DIRS})
##target_link_libraries(corner_detection PRIVATE ${OpenCV_LIBS})
#
#
##add_executable(FREAK_feature_description FREAK_feature_descriptor.cpp FREAK_feature_descriptor.h)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

#target_include_directories(FREAK_feature_description PRIVATE ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(FREAK_feature_description PRIVATE ${OpenCV_LIBS})
#


add_executable(test_feature_extraction  test_feature_extraction.cpp test_feature_extraction.h
        corner_detection.cpp corner_detection.h
        FREAK_feature_descriptor.cpp
        FREAK_feature_descriptor.h
        matching.cpp)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
#set(CMAKE_BUILD_TYPE Debug)  # Ensures debugging symbols are included
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")  # Adds debugging flags (no optimization)

target_include_directories(test_feature_extraction PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_feature_extraction PRIVATE ${OpenCV_LIBS})



# add_executable(test_orientation test_orientation.cpp test_orientation.h)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

# target_include_directories(test_orientation PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(test_orientation PRIVATE ${OpenCV_LIBS})


