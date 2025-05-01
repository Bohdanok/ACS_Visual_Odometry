cmake_minimum_required(VERSION 3.10)
project(FeatureExtraction)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

set(PROJECT_FEATURE_EXTRACTION_WITH_MATCHING feature_extraction_with_matching)
set(PROJECT_SERIAL_FEATURE_EXTRACTION test_serial)
set(PROJECT_PARALLEL_FEATURE_EXTRACTION test_parallel)
set(PROJECT_PARALLEL_FEATURE_EXTRACTION_WITH_MATCHING parallel_feature_extraction_with_matching)
set(PROJECT_PARALLEL_FEATURE_EXTRACTION_GPU_WITH_MATCHING GPU_parallel_feature_extraction_with_matching)
set(BOHDAN_TEST bohdan_test)
set(PROJECT_TEST_SERIAL_MATCHING serial_feature_extraction_with_matching)
set(PRECOMPILE_THE_SOURCE precompile_the_source)

add_executable(${PROJECT_FEATURE_EXTRACTION_WITH_MATCHING}
        feature_extraction/test_feature_extraction.cpp
        feature_extraction/corner_detection.cpp
        feature_extraction/FREAK_feature_descriptor.cpp
        matching.cpp)

target_include_directories(${PROJECT_FEATURE_EXTRACTION_WITH_MATCHING} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_FEATURE_EXTRACTION_WITH_MATCHING} PRIVATE ${OpenCV_LIBS})

add_executable(${PROJECT_PARALLEL_FEATURE_EXTRACTION}
        feature_extraction_parallel/threadpool.h
        feature_extraction_parallel/join_threads.h
        feature_extraction_parallel/ts_queue.h
        feature_extraction_parallel/corner_detection_parallel.cpp
        feature_extraction_parallel/FREAK_feature_descriptor_parallel.cpp
        feature_extraction_parallel/feature_extraction_parallel.cpp
        compare_with_serial.cpp)

target_include_directories(${PROJECT_PARALLEL_FEATURE_EXTRACTION} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_PARALLEL_FEATURE_EXTRACTION} PRIVATE ${OpenCV_LIBS})

add_executable(${PROJECT_PARALLEL_FEATURE_EXTRACTION_WITH_MATCHING}
        feature_extraction_parallel/corner_detection_parallel.cpp
        feature_extraction_parallel/FREAK_feature_descriptor_parallel.cpp
        feature_extraction_parallel/feature_extraction_parallel.cpp
        matching.cpp)

target_include_directories(${PROJECT_PARALLEL_FEATURE_EXTRACTION_WITH_MATCHING} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_PARALLEL_FEATURE_EXTRACTION_WITH_MATCHING} PRIVATE ${OpenCV_LIBS})
target_compile_definitions(${PROJECT_PARALLEL_FEATURE_EXTRACTION_WITH_MATCHING} PRIVATE PARALLEL_IMPLEMENTATION)

add_executable(${PROJECT_SERIAL_FEATURE_EXTRACTION}
        feature_extraction/corner_detection.cpp
        feature_extraction/FREAK_feature_descriptor.cpp
        feature_extraction/test_feature_extraction.cpp)

target_include_directories(${PROJECT_SERIAL_FEATURE_EXTRACTION} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_SERIAL_FEATURE_EXTRACTION} PRIVATE ${OpenCV_LIBS})

add_executable(${PROJECT_TEST_SERIAL_MATCHING}
        feature_extraction/test_feature_extraction.cpp
        feature_extraction/corner_detection.cpp
        feature_extraction/FREAK_feature_descriptor.cpp
        matching_serial.cpp)

target_include_directories(${PROJECT_TEST_SERIAL_MATCHING} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_TEST_SERIAL_MATCHING} PRIVATE ${OpenCV_LIBS})

# ////////////////////////

add_executable(${PROJECT_PARALLEL_FEATURE_EXTRACTION_GPU_WITH_MATCHING}
        feature_extraction_parallel_GPU/corner_detection_parallel_GPU.cpp
        feature_extraction_parallel_GPU/FREAK_feature_descriptor_parallel_GPU.cpp
        feature_extraction_parallel_GPU/feature_extraction_parallel_GPU.cpp
         matching.cpp
)

target_include_directories(${PROJECT_PARALLEL_FEATURE_EXTRACTION_GPU_WITH_MATCHING} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_PARALLEL_FEATURE_EXTRACTION_GPU_WITH_MATCHING} PRIVATE ${OpenCV_LIBS} OpenCL::OpenCL)
target_compile_definitions(${PROJECT_PARALLEL_FEATURE_EXTRACTION_GPU_WITH_MATCHING} PRIVATE GPU_IMPLEMENTATION)


add_executable(${BOHDAN_TEST}
        test_opencl.cpp)

target_include_directories(${BOHDAN_TEST} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${BOHDAN_TEST} PRIVATE ${OpenCV_LIBS} OpenCL::OpenCL)

add_executable(${PRECOMPILE_THE_SOURCE}
        feature_extraction_parallel_GPU/precompile_kernel_source_code.cpp)

target_include_directories(${PRECOMPILE_THE_SOURCE} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PRECOMPILE_THE_SOURCE} PRIVATE ${OpenCV_LIBS} OpenCL::OpenCL)



add_custom_target(run_precompile_kernel
        COMMAND ${CMAKE_COMMAND} -E echo "Running kernel precompilation step"
        COMMAND $<TARGET_FILE:${PRECOMPILE_THE_SOURCE}>
        DEPENDS ${PRECOMPILE_THE_SOURCE}
        COMMENT "Running ${PRECOMPILE_THE_SOURCE} to generate kernel binary"
)

add_dependencies(${PROJECT_PARALLEL_FEATURE_EXTRACTION_GPU_WITH_MATCHING} run_precompile_kernel)

add_dependencies(${BOHDAN_TEST} run_precompile_kernel)
