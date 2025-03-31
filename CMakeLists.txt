cmake_minimum_required(VERSION 3.10)
project(FeatureExtraction)
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

add_executable(corner_detection feature_extraction/corner_detection.cpp feature_extraction/corner_detection.h)

target_include_directories(corner_detection PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(corner_detection PRIVATE ${OpenCV_LIBS})


add_executable(FREAK_feature_description feature_extraction/FREAK_feature_descriptor.cpp feature_extraction/FREAK_feature_descriptor.h)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

target_include_directories(FREAK_feature_description PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(FREAK_feature_description PRIVATE ${OpenCV_LIBS})



add_executable(test_feature_extraction_with_matching feature_extraction/test_feature_extraction.cpp feature_extraction/test_feature_extraction.h
        feature_extraction/corner_detection.cpp feature_extraction/corner_detection.h
        feature_extraction/FREAK_feature_descriptor.cpp
        feature_extraction/FREAK_feature_descriptor.h
        matching.cpp)

#set(CMAKE_BUILD_TYPE Debug)  # Ensures debugging symbols are included
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")  # Adds debugging flags (no optimization)

target_include_directories(test_feature_extraction_with_matching PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_feature_extraction_with_matching PRIVATE ${OpenCV_LIBS})



#add_executable(parallel_test_feature_extraction_with_matching feature_extraction_parallel/test_feature_extraction.cpp feature_extraction_parallel/test_feature_extraction.h
#        feature_extraction_parallel/corner_detection_parallel.cpp feature_extraction_parallel/corner_detection.h
#        feature_extraction_parallel/FREAK_feature_descriptor.cpp
#        feature_extraction_parallel/FREAK_feature_descriptor.h
#        matching.cpp)
add_executable(parallel_test_feature_extraction feature_extraction_parallel/test_feature_extraction_parallel.cpp
        feature_extraction_parallel/threadpool.cpp feature_extraction_parallel/threadpool.h
        )

target_include_directories(parallel_test_feature_extraction PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(parallel_test_feature_extraction PRIVATE ${OpenCV_LIBS})


add_executable(compare_feature_extraction_with_serial

        feature_extraction/test_feature_extraction.cpp feature_extraction/test_feature_extraction.h
        feature_extraction/corner_detection.cpp feature_extraction/corner_detection.h
        feature_extraction/FREAK_feature_descriptor.cpp
        feature_extraction/FREAK_feature_descriptor.h


#        feature_extraction_parallel/test_feature_extraction_parallel.cpp
        feature_extraction_parallel/threadpool.cpp feature_extraction_parallel/threadpool.h
        feature_extraction_parallel/corner_detection_parallel.cpp feature_extraction_parallel/corner_detection_parallel.h
        compare_with_serial.cpp
        feature_extraction_parallel/join_threads.h
        feature_extraction_parallel/ts_queue.h

)

target_include_directories(compare_feature_extraction_with_serial PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(compare_feature_extraction_with_serial PRIVATE ${OpenCV_LIBS})


