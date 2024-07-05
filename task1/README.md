# Research and deployment of vins slam algorithm

## Introduction

It is very meaningful for robots to build a map of their environment in real time through vision.

As stated above, both of the following VINS slam algorithms can do this.

Please refer to the following "**Procedure**" to complete the relevant work.

[VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) is a real-time SLAM framework for Monocular Visual-Inertial Systems. It uses an optimization-based sliding window formulation for providing high-accuracy visual-inertial odometry. It features efficient IMU pre-integration with bias correction, automatic estimator initialization, online extrinsic calibration, failure detection and recovery, loop detection, and global pose graph optimization, map merge, pose graph reuse, online temporal calibration, rolling shutter support. VINS-Mono is primarily designed for state estimation and feedback control of autonomous drones, but it is also capable of providing accurate localization for AR applications. This code runs on Linux, and is fully integrated with ROS. For iOS mobile implementation, please go to VINS-Mobile.

[VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) is an optimization-based multi-sensor state estimator, which achieves accurate self-localization for autonomous applications (drones, cars, and AR/VR). VINS-Fusion is an extension of VINS-Mono, which supports multiple visual-inertial sensor types (mono camera + IMU, stereo cameras + IMU, even stereo cameras only). We also show a toy example of fusing VINS with GPS. Features:

- multiple sensors support (stereo cameras / mono camera+IMU / stereo cameras+IMU)
- online spatial calibration (transformation between camera and IMU)
- online temporal calibration (time offset between camera and IMU)
- visual loop closure

## Procedure

### stage1

1. Deploy ROS
2. Learning Basic operations of ROS.

### stage2

1. Learn about SLAM
2. Read VINS related papers

### stage3 (optional)

1. Use a camera and deploy VINS-Mono

## Submit

Any details and learning about the deployment process are worth putting in your folder.
