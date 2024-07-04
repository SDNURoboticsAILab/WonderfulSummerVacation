# Research and deployment of vins slam algorithm

## VINS Algorithm Introduction

[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) is a real-time SLAM framework for Monocular Visual-Inertial Systems. It uses an optimization-based sliding window formulation for providing high-accuracy visual-inertial odometry. It features efficient IMU pre-integration with bias correction, automatic estimator initialization, online extrinsic calibration, failure detection and recovery, loop detection, and global pose graph optimization, map merge, pose graph reuse, online temporal calibration, rolling shutter support. VINS-Mono is primarily designed for state estimation and feedback control of autonomous drones, but it is also capable of providing accurate localization for AR applications. This code runs on Linux, and is fully integrated with ROS. For iOS mobile implementation, please go to VINS-Mobile.

[VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) is an optimization-based multi-sensor state estimator, which achieves accurate self-localization for autonomous applications (drones, cars, and AR/VR). VINS-Fusion is an extension of VINS-Mono, which supports multiple visual-inertial sensor types (mono camera + IMU, stereo cameras + IMU, even stereo cameras only). We also show a toy example of fusing VINS with GPS. Features:

- multiple sensors support (stereo cameras / mono camera+IMU / stereo cameras+IMU)
- online spatial calibration (transformation between camera and IMU)
- online temporal calibration (time offset between camera and IMU)
- visual loop closure

## 