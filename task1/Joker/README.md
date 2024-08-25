# ROS2 And SLAM

学习《ROS2机器人开发从入门到实践》第1-7章的内容，学习了ROS2开发环境的搭建、节点、话题的订阅与发布、ros2通信、一些常用的开发工具以及第六章的在仿真环境中创建自己的机器人和第七章的自主导航移动。

### 开发环境

1. Ubuntu22.04 LTS
2. ROS2 humble

### 文件目录
```
Joker
├─chapt6
│  └─chapt6_ws
│      ├─build
│      │  └─fishbot_description
│      │      ├─ament_cmake_core
│      │      │  └─stamps
│      │      ├─ament_cmake_environment_hooks
│      │      ├─ament_cmake_index
│      │      │  └─share
│      │      │      └─ament_index
│      │      │          └─resource_index
│      │      │              ├─packages
│      │      │              ├─package_run_dependencies
│      │      │              └─parent_prefix_path
│      │      ├─ament_cmake_package_templates
│      │      ├─ament_cmake_uninstall_target
│      │      ├─ament_flake8
│      │      ├─ament_lint_cmake
│      │      ├─ament_pep257
│      │      ├─ament_xmllint
│      │      └─CMakeFiles
│      │          ├─3.22.1
│      │          │  ├─CompilerIdC
│      │          │  │  └─tmp
│      │          │  └─CompilerIdCXX
│      │          │      └─tmp
│      │          ├─CMakeTmp
│      │          ├─fishbot_description_uninstall.dir
│      │          └─uninstall.dir
│      ├─install
│      │  └─fishbot_description
│      │      └─share
│      │          ├─ament_index
│      │          │  └─resource_index
│      │          │      ├─packages
│      │          │      ├─package_run_dependencies
│      │          │      └─parent_prefix_path
│      │          ├─colcon-core
│      │          │  └─packages
│      │          └─fishbot_description
│      │              ├─cmake
│      │              ├─config
│      │              │  └─rviz
│      │              ├─environment
│      │              ├─hook
│      │              ├─launch
│      │              │  └─__pycache__
│      │              ├─urdf
│      │              │  └─fishbot
│      │              │      ├─actuator
│      │              │      ├─plugins
│      │              │      └─sensor
│      │              └─world
│      │                  └─room
│      ├─log
│      │  ├─build_2024-08-23_14-37-52
│      │  │  └─fishbot_description
│      │  └─build_2024-08-23_14-45-39
│      │      └─fishbot_description
│      └─src
│          └─fishbot_description
│              ├─config
│              │  └─rviz
│              ├─launch
│              ├─urdf
│              │  └─fishbot
│              │      ├─actuator
│              │      ├─plugins
│              │      └─sensor
│              └─world
│                  └─room
└─chapt7
    └─chapt7_ws
        └─src
            ├─autopatrol_interfaces
            │  └─srv
            ├─autopatrol_robot
            │  ├─autopatrol_robot
            │  ├─config
            │  ├─launch
            │  ├─resource
            │  └─test
            ├─fishbot_application
            │  ├─fishbot_application
            │  ├─resource
            │  └─test
            ├─fishbot_application_cpp
            │  └─src
            ├─fishbot_description
            │  ├─config
            │  │  └─rviz
            │  ├─launch
            │  ├─urdf
            │  │  └─fishbot
            │  │      ├─actuator
            │  │      ├─plugins
            │  │      └─sensor
            │  └─world
            │      └─room
            ├─fishbot_navigation2
            │  ├─config
            │  ├─launch
            │  └─maps
            └─navigation2
```

### 作者

崔瑾栖 刘润泽

### 参考文献

在本项目中，我们参考了以下文献进行研究：

- ​	[Research on SLAM Algorithm and Navigation of Mobile Robot Based on ROS](https://ieeexplore.ieee.org/document/9512584)
- ​    [SLAM Self - Cruise Vehicle Based on ROS Platform](https://ieeexplore.ieee.org/document/9342204)  
- ​    [Research on SLAM navigation of wheeled mobile robot based on ROS*](https://ieeexplore.ieee.org/document/9230186)
- ​    [Implementation of SLAM and path planning for mobile robots under ROS framework](https://ieeexplore.ieee.org/document/9408882)
