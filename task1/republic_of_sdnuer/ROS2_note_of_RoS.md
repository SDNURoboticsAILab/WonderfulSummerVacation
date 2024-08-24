## 1) ROS2(humble) installation

- #### Set locale

  ```
  $ sudo apt update 
  $ sudo apt install locales
  $ sudo locale-gen en_US en_US.UTF-8
  $ sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  $ export LANG=en_US.UTF-8
  ```

- #### Setup sources

  ```
  ## 通过检查此命令的输出，确保已启用Ubuntu Universe存储库。
  $ apt-cache policy | grep universe
  ## 输出应如下：
  500 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages
       release v=22.04,o=Ubuntu,a=jammy-security,n=jammy,l=Ubuntu,c=universe,b=amd64
  ## 若没有看到像上面这样的输出行，依次执行如下命令：
  $ sudo apt install software-properties-common
  $ sudo add-apt-repository universe
  
  
  ## 继续执行如下命令：
  $ sudo apt update && sudo apt install curl gnupg lsb-release
  $ sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  $ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
  ```

- #### Install ROS2 packages

  ```
  $ sudo apt update
  $ sudo apt upgrade
  ## 推荐桌面版，比较推荐。
  $ sudo apt install ros-humble-desktop
  ## 安装时间可能较长，安心等待。
  ```

- #### Environment setup

  ```
  $ source /opt/ros/humble/setup.bash
  $ echo " source /opt/ros/humble/setup.bash" >> ~/.bashrcor
  ```

  ### or

- #### By *fishros*(recommend)

  ```
  wget http://fishros.com/install -O fishros && . fishros
  ```



### Test your env.

#### In two different terminal,

```
source /opt/ros/humble/setup.bash
```

#### in the first,

```
ros2 run turtlesim turtlesim_node
```

#### the other

```
ros2 run turtlesim  turtle_teleop_key
```

#### In the last terminal, use "↑ ↓ ← → " to check it out.



#### *You may need rosdep or something to continue, fishros's enough.*





## 2) Basic CLI tools

### Understanding 

- #### Nodes

  A node is responsible for a ***single modular purpose*** such as publishing the sensor data from a laser range-finder.

  Each node can send and receive data from other nodes via ***topics***, ***services***, ***actions***, or ***parameters***.

- #### Topics —— "publisher-subscriber model"

​		Topics act as a bus to exchange messages for nodes.

​		A node can ***publish*** data to ***any number*** of topics and at the same time ***subscribe*** ***any number*** of 		topics

- #### Services —— "call-and-response model"

​		Unlike topics that allow nodes to subscribe to data streams and get continual updates, services only 		***provide data*** when they are specifically ***called by a client***.

- #### Parameters

​		Briefly, parameters can regarded as node settings that could be stored as integers, float, booleans, 		strings, etc.

- #### Actions —— "client-server model"

​		A action consist of three parts: a goal, feedback, and a result.

​		Actions are built on topics and services. Their functionality is similar to services, except actions can 		be canceled. They provide steady feedback, as opposed to services which return a single response.

​		An ***action client*** node sends a goal to an ***action server*** node that acknowledges the goal and returns 		a stream of feedback and a result.

### Implement

- #### Creating work space

  A workspace is a directory containing ROS 2 packages.(Don't forget to source command though you have done it wwww)

  ```
  mkdir -p ~/ros2_ws/src
  cd ~/ros2_ws/src
  ```

  With already having some packages

  ```
  ##from the root of your workspace
  colcon build
  ```

  The console will return following message like:

  ```
  Starting >>> turtlesim
  Finished <<< turtlesim [5.49s]
  
  Summary: 1 package finished [5.58s]
  ```

  In the root of your workspace

  ```
  ls
  build  install  log  src
  ```

- #### Write a simple publisher and subsriber/service and client

- #### About custom interfaces

- #### Using parameters in a class

##### 	More information: [[Tutorials — ROS 2 Documentation: Humble documentation](https://docs.ros.org/en/humble/Tutorials.html)](https://docs.ros.org/en/humble/Tutorials.html)



## 3)Simulation

### Understanding

- #### URDF (Unified Robot Description Format)

  A file format for specifying the geometry and organization of robots in ROS.(.XML)

  For example, a simple URDF model be like:

  ```
  <?xml version="1.0"?>
  <robot name="fishbot">
    <link name="base_link">
      <visual>
        <geometry>
          <cylinder length="0.18" radius="0.06"/>
        </geometry>
      </visual>
    </link>
  </robot>
  ```

  Generally, URDF is made up of the **declaration** and two **Joint&Link**

  **Declaration**

  ```
  <?xml version="1.0"?>
  <robot name="fishbot">
   	<link></link>
   	<joint></joint>
    ......
  </robot>
  
  ```

  **Joint&Link**

  ```
  ...
  ```

### Implement

- #### Gazebo

  Install and launch

  ```
  sudo apt install ros-humble-gazebo-ros
  gazebo --verbose -s libgazebo_ros_init.so -s libgazebo_ros_factory.so 
  ```

  ```
  to be continued...
  ```

  A example to [try](https://github.com/fishros/fishbot/blob/navgation2/src/fishbot_description/urdf/fishbot_gazebo.urdf)
  
  In rqt,  select  service **/spawn_entity**, paste the URDF code in xml(don't forget to delete   ''   !!!)

- #### **Turtlebot4**

  **Install**

  Install rcm

  ```
  curl https://www.ncnynl.com/rcm.sh | bash -
  ```

  Install tb4 with rcm

  ```
  rcm -s install_tb4_humble_source
  ```

  Launch standard edition

  ```
  ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py
  ```

  or the lite

  ```
  ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py model:=lite
  ```

  **Test lidar and camera**

  Launch

  ```
  ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py
  ```

  In terminal

  ```
  ros2 topic echo /scan
  ```

  the result would be like:

  ```
  ---
  header:
    stamp:
      sec: 172
      nanosec: 816000000
    frame_id: turtlebot4/rplidar_link/rplidar
  angle_min: -3.1415927410125732
  angle_max: 3.1415927410125732
  angle_increment: 0.009832840412855148
  time_increment: 0.0
  scan_time: 0.0
  range_min: 0.164000004529953
  range_max: 12.0
  ranges:
  - 7.365904331207275
  - 3.948739528656006
  - 3.907142400741577
  - 8.883407592773438
  - 8.94767951965332
  - 9.012924194335938
  - 9.079118728637695
  - 9.146242141723633
  - 9.214299201965332
  - 9.82641887664795
  - 9.902118682861328
  - 9.978501319885254
  - 10.055573463439941
  - .inf
  - .inf
  - '...'
  intensities:
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - 0.0
  - '...'
  ---
  ```

  meanwhile, you could use **rviz** to check the lidar

  ```
  ros2 launch turtlebot3_viz view_robot.launch.py
  ```

  use image_view to check the camera

  ```
  ros2 run rqt_image_view rqt_image_view 
  ```

  ##### **Asynchronous mapping**

  ```
  ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py sync:=false slam:=true nav2:=true rviz:=true
  ```

  ##### **Synchronous mapping**
  
  ```
  ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py slam:=true rviz:=true
  ```

  ##### **Saving the map**
  
  ```
  mkdir ~/map
  ros2 run nav2_map_server map_saver_cli -f ~/map/slam_toolbox_async --ros-args -p save_map_timeout:=10000.00
  ```

  and
  
  ```
    eog ~/map/slam_toolbox_async.pgm
  ```
  
  ##### **Navigation**
  
  **single point:**
  
  ```
  ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py nav2:=true slam:=false localization:=true 
  ```
  
  ```
  ros2 launch turtlebot4_viz view_robot.launch.py use_sim_time:=true
  ```
  
  use 2D Pose Estimate to initialize, and then set destination with nav2 Goal.
  
  
  
  **multipoint:**
  
  ```
  ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py nav2:=true slam:=false localization:=true 
  ```
  
  ```
  ros2 launch turtlebot4_viz view_robot.launch.py use_sim_time:=true
  ```
  
  use 2D Pose Estimate to initialize, and then click waypoint/nav through poses mode, set destination with nav2 Goal. Final is Start Nav Through Poses. 
  
  
  
  **Merge mapping and navigation:**
  
  ```
  ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py slam:=true nav2:=true rviz:=true
  ```
  
  next steps are the same as above.
  
  
  
  #### **The problems you might encounter**
  
  1. **[error] (while the first launch)**
  
     ```
     Package 'turtlebot4_ignition_bringup' not found: "package 'turtlebot4_ignition_bringup' not found, searching: ['/opt/ros/humble']"
     ```
  
     Install ignition
  
     ```
     sudo apt update
     sudo apt -y install ignition
     ```
  
  2. **defaults of lidar and camera**
  
     ```
     As for lidar:
     ......
     ranges:
     0
     0
     intensities:
     1.0
     1.0
     ```
  
     You need to add an environment  variable
  
     ```
     sudo vim ~/.bashrc
     ```
  
     then,
  
     ```
     export LIBGL_ALWAYS_SOFTWARE=true
     ```
  
     go back to check if the lidar is working. 
  
     If so, however,  just **a little bit lagged**,  make sure that you already have the decent graphics driver.
  
     To start with, open **/ros2_tb4_ws/src/create3_sim/irobot_create_common/irobot_create_description/urdf/create3.urdf.xacro** . Check **Line 301** whether it is **ogre**. And you can find what you need from [here]([Ubuntu22.04安装显卡驱动(高速、避错版)-CSDN博客](https://blog.csdn.net/Eric_xkk/article/details/131800365)). You may need to set **Graphics Processing Unit Direct** in bios.
  
     

