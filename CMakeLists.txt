cmake_minimum_required(VERSION 3.10)
project(FeatureExtraction)
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

#set(PROJECT_CORNER_DETECTION corner_detection)
#set(PROJECT_FREAK_FEATURE_DESCRIPTION FREAK_feature_description)
set(PROJECT_FEATURE_EXTRACTION_WITH_MATCHING feature_extraction_with_matching)
set(PROJECT_SERIAL_FEATURE_EXTRACTION test_serial)
set(PROJECT_PARALLEL_FEATURE_EXTRACTION test_parallel)
set(PROJECT_PARALLEL_FEATURE_EXTRACTION_WITH_MATCHING parallel_feature_extraction_with_matching)
set(BOHDAN_TEST bohdan_test)
set(PROJECT_TEST_SERIAL_MATCHING serial_feature_extraction_with_matching)


add_executable(${PROJECT_FEATURE_EXTRACTION_WITH_MATCHING} feature_extraction/test_feature_extraction.cpp feature_extraction/test_feature_extraction.h
        feature_extraction/corner_detection.cpp feature_extraction/corner_detection.h
        feature_extraction/FREAK_feature_descriptor.cpp
        feature_extraction/FREAK_feature_descriptor.h
        matching.cpp)


target_include_directories(${PROJECT_FEATURE_EXTRACTION_WITH_MATCHING} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_FEATURE_EXTRACTION_WITH_MATCHING} PRIVATE ${OpenCV_LIBS})



add_executable(${PROJECT_PARALLEL_FEATURE_EXTRACTION}

        feature_extraction_parallel/threadpool.h
        feature_extraction_parallel/join_threads.h
        feature_extraction_parallel/ts_queue.h
        feature_extraction_parallel/corner_detection_parallel.cpp feature_extraction_parallel/corner_detection_parallel.h
        feature_extraction_parallel/FREAK_feature_descriptor_parallel.cpp feature_extraction_parallel/FREAK_feature_descriptor_parallel.h
        feature_extraction_parallel/feature_extraction_parallel.cpp feature_extraction_parallel/feature_extraction_parallel.h
        compare_with_serial.cpp


)

target_include_directories(${PROJECT_PARALLEL_FEATURE_EXTRACTION} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_PARALLEL_FEATURE_EXTRACTION} PRIVATE ${OpenCV_LIBS})


add_executable(${PROJECT_PARALLEL_FEATURE_EXTRACTION_WITH_MATCHING}
        feature_extraction_parallel/corner_detection_parallel.cpp feature_extraction_parallel/corner_detection_parallel.h
        feature_extraction_parallel/FREAK_feature_descriptor_parallel.cpp feature_extraction_parallel/FREAK_feature_descriptor_parallel.h
        feature_extraction_parallel/feature_extraction_parallel.cpp feature_extraction_parallel/feature_extraction_parallel.h
        matching.cpp)

target_include_directories(${PROJECT_PARALLEL_FEATURE_EXTRACTION_WITH_MATCHING} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_PARALLEL_FEATURE_EXTRACTION_WITH_MATCHING} PRIVATE ${OpenCV_LIBS})

target_compile_definitions(${PROJECT_PARALLEL_FEATURE_EXTRACTION_WITH_MATCHING} PRIVATE PARALLEL_IMPLEMENTATION)


add_executable(${PROJECT_SERIAL_FEATURE_EXTRACTION}

        feature_extraction/corner_detection.cpp feature_extraction/corner_detection.h
        feature_extraction/FREAK_feature_descriptor.cpp feature_extraction/FREAK_feature_descriptor.h
        feature_extraction/test_feature_extraction.cpp feature_extraction/test_feature_extraction.h
)

target_include_directories(${PROJECT_SERIAL_FEATURE_EXTRACTION} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_SERIAL_FEATURE_EXTRACTION} PRIVATE ${OpenCV_LIBS})


add_executable(${PROJECT_TEST_SERIAL_MATCHING} feature_extraction/test_feature_extraction.cpp feature_extraction/test_feature_extraction.h
        feature_extraction/corner_detection.cpp feature_extraction/corner_detection.h
        feature_extraction/FREAK_feature_descriptor.cpp
        feature_extraction/FREAK_feature_descriptor.h
        matching_serial.cpp)

#set(CMAKE_BUILD_TYPE Debug)  # Ensures debugging symbols are included
#set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")  # Adds debugging flags (no optimization)

target_include_directories(${PROJECT_TEST_SERIAL_MATCHING} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_TEST_SERIAL_MATCHING} PRIVATE ${OpenCV_LIBS})
