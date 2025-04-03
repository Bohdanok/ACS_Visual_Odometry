# ACS_Visual_Odometry

Visual odometry is a state estimation problem where the goal is to estimate the camera’s (relative) poses  in the 3D world from an image sequence. Visual odometry is meant to run in real-time and is crucial for autonomous navigation, mission planning, etc. The implementation is benchmarked against OpenCV-based solutions and designed to serve as a foundation for further developments in SLAM and sensor fusion.

**Authors**

<br>[Sofiia Sampara](https://github.com/sofiasampara76)<br> [Sofiia Popeniuk](https://github.com/SofiiaPop)<br> [Ostap Pavlyshyn](https://github.com/Ostik24)

**Requirements**

C++17

OpenCV 4.x

Eigen 3

CMake ≥ 3.15

POSIX-compatible thread support

**Compilation**

for feature extraction and matching (branch Bohdan_and_Sofia)

mkdir build

cd build

cmake ..

make

**Usage**
for feature extraction and matching (branch Bohdan_and_Sofia)

./feature_extraction_with_matching ../images/f1.png ../images/f2.png

**Results**

Feature extraction is based on detecting corners using the Harris and Shi-Tomasi methods. Images are first converted to grayscale and denoised using Gaussian smoothing to improve detection robustness. Gradients and structure matrices are computed to identify distinct keypoints. Feature description uses the FREAK descriptor, which is efficient and inspired by the retinal sampling pattern of the human eye. FREAK descriptors are binary and support rotation invariance, making them suitable for real-time visual tracking.

Feature matching is done using a brute-force matcher. For binary descriptors, the Hamming distance is applied to determine the best matches between feature points in consecutive frames.

Motion estimation is performed by computing the fundamental matrix via RANSAC to eliminate outliers. The estimation is refined before being used for pose updates.

Parallelization is introduced at the feature extraction and feature matching stages. A thread pool implementation ensures concurrent processing of image data while maintaining synchronization and avoiding data races. This significantly improves performance on high-resolution inputs.
