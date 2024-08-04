[TOC]

### 一、ROS1与ROS2的区别

### 二、ROS2安装

#### 1.ubuntu22.04安装+VM虚拟机 

#### 2. ROS2安装

```bash
命令行输入：
# 设置命令行中的编码方式
sudo apt update && sudo apt install locales   
sudo locale-gen en_US en_US>UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 添加源
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
## 在输入上述命令时会出现下列问题：
curl: (7) Failed to connect to raw.githubusercontent.com port 443 after 16 ms: Connection refused
解决方法：https://www.guyuehome.com/37844
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 安装ROS2
# 安装
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop  **等待一会
*** ROS的安装位置 computer/opt/ros/humble ，如下图所示
# 环境变量
source /opt/ros/humble/setup.bash  **只在当前终端中有效
echo " source opt/ros/humble/setup.bash" >> ~/.bashrc   **所有终端中都生效了
```



![image-20240726120116063](day01_ROS2环境部署.assets/image-20240726120116063.png)

**一键安装：wget http://fishros.com/install -O fishros && . fishros**（鱼香ros)   yyds

#### 3. 安装测试

* （1）测试示例1

```bash
## 在两个终端分别输入
ros2 run demo_nodes_cpp talker
ros2 run demo_nodes_cpp listener
```

* （2）测试示例2

  ```
  ## 小海龟仿真
  ros2 run turtlesim turtlesim_node
  ## 在另一个终端可操控海龟
  ros2 run turtlesim turtle_teleop_key  ** 在输入ros2命令时可使用tab键补全
  ```

  ```
  ## 控制海龟运动
  ros2 topic pub --rate 1 /turtle1/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 2.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 1.8}}"
  ## 生成新的海龟
  ros2 service call /spawn turtlesim/srv/Spawn "{x: 2, y: 2, theta: 0.2, name: '新生成的海龟的名字'}" 
  
  ```
  
  ```
  ros2 bag 录制
  ```
  
  ### 三、ROS2开发环境配置
  
  #### 1.git
  
  ```
  sudo apt install git
  ```
  
  #### 2. VScode下载
  
  >搜索官网：https://code.vsualstudio.com
  >
  >下载deb版，即安装在ubuntu中的版本
  >
  >在下载里边找到该文件，在该位置打开终端输入：`sudo dpkg -i 文件名`
  >
  >插件安装

 









