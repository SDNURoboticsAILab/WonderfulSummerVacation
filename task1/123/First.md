# 在电脑上部署 ROS

## 1. 选择合适的 ROS 版本
- 根据你的操作系统选择对应的 ROS 版本。常见的有 ROS Noetic (适用于 Ubuntu 20.04)、ROS Melodic (适用于 Ubuntu 18.04)、ROS Kinetic (适用于 Ubuntu 16.04) 等。
- 可以查看 ROS 官网的[发行版列表](http://wiki.ros.org/Distributions)来选择合适的版本。

## 2. 安装 ROS
- 以 Ubuntu 20.04 和 ROS Noetic 为例,按照以下步骤进行安装:

  ```bash
  # 设置 ROS 软件源
  sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
  sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

  # 更新软件包列表并安装 ROS Noetic
  sudo apt update
  sudo apt install ros-noetic-desktop-full
## 3.初始化rosdep
- ROS 依赖一些系统库,需要使用 rosdep 命令进行初始化和更新
  ```bash
    sudo rosdep init
    rosdep update
## 4.设置ROS环境变量
- 每次打开新的终端时,需要执行以下命令以设置 ROS 环境变量:
  ```bash
    ource /opt/ros/noetic/setup.bash
- 为了避免每次都手动执行,可以将上述命令添加到 ~/.bashrc 文件中,这样每次打开终端时都会自动设置 ROS 环境变量。
## 5.创建ROS工作空间
 - ROS 的工作空间:ROS 的工作空间是一个用于组织和编译 ROS 软件包的目录结构。它包含以下几个重要组成部分:

    ### src 目录

    这个目录用于存放 ROS 软件包的源代码。每个 ROS 软件包都应该位于 src 目录下的一个子目录中。
在创建新的 ROS 软件包时,它们会被放置在这个目录下。
    ### build 目录
    这个目录用于存放 ROS 软件包编译后生成的中间文件。当你运行 catkin_make 命令时,编译结果会被存放在这个目录中。
    ### devel 目录

    这个目录用于存放编译后生成的可执行文件和库文件。
当你运行 catkin_make 命令时,编译结果会被复制到这个目录中。
每次打开新的终端时,都需要执行 source devel/setup.bash 来初始化 ROS 环境。
    ### CMakeLists.txt 文件
    这个文件用于配置 ROS 软件包的编译规则。它告诉 ROS 如何编译和链接你的代码。
    ### package.xml 文件
    这个文件用于描述 ROS 软件包的元数据,如名称、版本、依赖关系等。它为 ROS 系统提供了关于软件包的基本信息。
- ROS 中的代码通常放在工作空间中进行开发和编译。可以使用以下命令创建一个新的工作空间:
  ```bash
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws
    catkin_make