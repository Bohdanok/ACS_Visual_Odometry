# ACS_Visual_Odometry

Visual odometry is a state estimation problem where the goal is to estimate the camera’s (relative) poses  in the 3D world from an image sequence. Visual odometry is meant to run in real-time and is crucial for autonomous navigation, mission planning, etc.


**Installation**

**Compilation**
for feature extraction and matching (branch Bohdan_and_Sofia)
mkdir build
cd build
cmake ..
make

**Usage**
for feature extraction and matching (branch Bohdan_and_Sofia)
./feature_extraction_with_matching ../images/f1.png ../images/f2.png
