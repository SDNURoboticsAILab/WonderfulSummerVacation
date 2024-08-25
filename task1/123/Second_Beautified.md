ROS的安装与使用 
# apt方式安装
安装
说起ROS，可能大家现在或多或少都有所了解。现如今世界机器人发展之迅猛犹如几十年前计算机行业一样，机器人也逐渐进入到千家万户，大到工业机器人，小到家用的服务型机器人，各式各样，为各种人们生活所需的机器人以计算机技术的发展为基础的机器人也是如雨后春笋。机器人可主要分为硬件层和软件层两个大的主要方向。每一种类型的机器人都需要以硬件的实际情况编写符合用户需要的功能，渐渐的人们发现，这样的机器人代码的复用率很低，大大阻碍了机器人的发展。因此ROS便是为机器人在研发的过程中的代码复用提供支持.ROS是Robot Operating System（机器人操作系统）的简称.ROS开始于2007年，在斯坦福大学人工智能实验室斯坦福AI机器人项目的支持下开发了ROS。从2010年3月2日发布的第一版ROS Box Turtle至今（截止到2018年8月）已有12个版本。对 ROS 兼容性最好的当属 Ubuntu 操作系统了。其中三个长期支持版本，并对应着的Ubuntu的三个LTS版本具体如下：

ROS版本      发行时间       截止支持时间      对应的Ubuntu的版本

| ROS版本 | 发行时间 | 截止支持时间 | 对应的Ubuntu的版本 |
| --- | --- | --- | --- |
ROS Indigo    2014年7月22    2019年4月  Ubuntu 14.04

ROS Kinetic   2016年5月23    2021年4月  Ubuntu 16.04

ROS Melodic 2018年5月23    2023年4月  Ubuntu 18.04

 更多的请参考 [http://wiki.ros.org/Distributions](http://wiki.ros.org/Distributions) 列出的支持计划。

 

ROS支持的Python，C ++，JAVA等编程语言。因为ROS主要支持Ubuntu的操作系统，因此本教程也是在Ubuntu的系统下安装的。这里比较建议安装的Ubuntu系统或者的Windows的双系统，这样可以更便于ROS学习接 下来为大家详细讲解ROS的安装。

 

（1）配置的Ubuntu的系统

“软件和更新”界面中，打开后按照下图进行配置（确保你的"restricted"， "universe，" 和 "multiverse."前是打上勾的），最好将源设置为中国。

 

输入```
sudo 

```gedit /etc/apt/sources.list



deb-src [http://archive.ubuntu.com/ubuntu](http://archive.ubuntu.com/ubuntu) xenial main restricted #Added by software-properties  
deb [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial main restricted  
deb-src [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial main restricted multiverse universe #Added by software-properties  
deb [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial-updates main restricted  
deb-src [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial-updates main restricted multiverse universe #Added by software-properties  
deb [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial universe  
deb [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial-updates universe  
deb [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial multiverse  
deb [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial-updates multiverse  
deb [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial-backports main restricted universe multiverse  
deb-src [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial-backports main restricted universe multiverse #Added by software-properties  
deb [http://archive.canonical.com/ubuntu](http://archive.canonical.com/ubuntu) xenial partner  
deb-src [http://archive.canonical.com/ubuntu](http://archive.canonical.com/ubuntu) xenial partner  
deb [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial-security main restricted  
deb-src [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial-security main restricted multiverse universe #Added by software-properties  
deb [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial-security universe  
deb [http://mirrors.aliyun.com/ubuntu/](http://mirrors.aliyun.com/ubuntu/) xenial-security multiverse 
复制代码
 

（2）打开终端，快捷键是按Ctrl + Alt + T键，配置sources.list

设置计算机可以从ROS.org网站上接收软件。

```
sudo 
```sh -c 'echo "deb [http://packages.ros.org/ros/ubuntu](http://packages.ros.org/ros/ubuntu) $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
 

（3）设置秘钥

```
sudo 
```apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F42ED6FBAB17C654

xxxx@cheetah-Z2-R:~$ ```
sudo 
```apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
Executing: /tmp/apt-key-gpghome.VQHtUrQqXU/gpg.1.sh --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
gpg: 密钥 5523BAEEB01FA116：公钥 “ROS Builder <rosbuild@ros.org>” 已导入
gpg: 处理的总数：1
gpg:               已导入：1
xxxx@cheetah-Z2-R:~$ 
复制代码
以上错误以及类似的“E: 无法定位软件包 ros-melodic-desktop-full”的错误，是由于ROS的版本不对应导致的，每个不同的ubuntu系统对应着不同的ROS版本，如果装错了就会出现上述问题，在Ubuntu18.04.1环境下可以安装的是melodic版本的，安装教程可以参考官网安装教程，ROS有Melodic、Lunar、Kinetic不同的种类对应着不同的ubuntu版本，
Melodic 主要对应：Ubuntu Artful (17.10), Bionic (18.04 LTS) 以及Debian Stretch
Kinetic 主要对应：Ubuntu Wily (15.10) and Ubuntu Xenial (16.04 LTS),
在安装的过程中要根据自己的ubuntu系统版本选择对应的ROS类型进行安装，具体的ROS类型可以在官网分支类型介绍上查看。
解决方法：

打开软件与更新->其他软件，找到ros那个软件源，选定后点编辑，把发行版从disco 改成bionic （Ubuntu18）或xenial应该就行了。

 PS: 最好以对应的方式去安装，否则后面的麻烦事情很多。

 

 

（4）安装

现在读者已经将ROS版本库地址放在代码列表中，然后更新下软件包。

重新定向ROS服务器

```
sudo 
```apt-get update && ```
sudo 
```apt-get upgrade -y

正在读取软件包列表... 完成                                                                          
E: 仓库 “[http://packages.ros.org/ros/ubuntu](http://packages.ros.org/ros/ubuntu) disco Release” 没有 Release 文件。
N: 无法安全地用该源进行更新，所以默认禁用该源。
N: 参见 apt-secure(8) 手册以了解仓库创建和用户配置方面的细节。
..
cheetah@cheetah-Z2-R:~$ 
复制代码
 

然后我们就可以安装 ROS 啦，但是问题又出现了，ROS kinetic 也有很多版本，比如工业版，基础版，高级版，豪华版，至尊豪华..

ROS，那就安装至尊豪华全功能版吧，指令如下

```
sudo 
```apt-get install ros-kinetic-desktop-full
 或其他版本的ros等

```
sudo 
```apt-get install ros-melodic-desktop-full
但是又报错

下列软件包有未满足的依赖关系：


下列软件包有未满足的依赖关系：
 ros-melodic-desktop-full : 依赖: ros-melodic-desktop 但是它将不会被安装
                            依赖: ros-melodic-perception 但是它将不会被安装
                            依赖: ros-melodic-simulators 但是它将不会被安装
                            依赖: ros-melodic-urdf-sim-tutorial 但是它将不会被安装
E: 无法修正错误，因为您要求某些软件包保持现状，就是它们破坏了软件包间的依赖关系。
cheetah@cheetah-Z2-R:~$ 
复制代码
解决方法为：```
sudo 
```apt-get install xxx（上面报的缺少的东西）

 

 

 

安装完成后，可以用下面的命令来查看可使用的包：

apt-cache search ros-kinetic
 

（5）初始化ROS

在使用ROS之前，必须先安装和初始化rosdep命令行工具。这样便可以使你轻松的安装库和源代码时的系统依赖。与此同时，ROS中的一些核心组件也需要rosdep，因此rosdep默认安装在ROS中。可以使用下面的命令安装和初始化rosdep。

```
sudo 
```rosdep init

rosdep updata

（6）初始化环境变量

你已经成功的安装了ROS系统了。但是为了每次打开新的终端不用执行脚本来配置环境变量，需要在.bashrc中的脚本文件结束时添加脚本，用下面这个命令来完成。

echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

要是配置生效，必须使用下面的命令来执行这个文件

source ~/.bashrc

上面两句非常非常非常重要，很多小伙伴在日常的开发过程中，有的找不到 Package, 找不到node, 很多情况下都是没有添加source。

（7）安装rosinstall

rosinstall。是一种常用的命令行工具，使你可以使用一个命令轻松下载ROS包的许多源，输入下面的命令安装这个工具：

```
sudo 
```apt-get install python-rosinstall

到此为止，你已经在电脑上安装了一个较为完整的ROS系统了。

对了，这里要提醒一下，ros中很多的第三方插件的安装格式是

$ ```
sudo 
```apt-get install ros-kinetic-...

例如：

$ ```
sudo apt-get install ros-kinetic-turtlebot*

 

卸载ROS
```
sudo 
```apt-get purge ros-*

```
sudo 
```rm -rf /etc/ros

edit ~/.bashrc

 

 

## 官方方式安装
安装
按照 ROS安装说明（Ubuntu install of ROS Melodic）完成安装

注意: 如果你是使用类似apt这样的软件管理器来安装ROS的，那么安装后这些软件包将不具备写入权限，当前系统用户比如你自己也无法对这些软件包进行修改编辑。当你的开发涉及到ROS软件包源码层面的操作或者在创建一个新的ROS软件包时，你应该是在一个具备读写权限的目录下工作，就像在你当前系统用户的home目录下一样。

管理环境
安装ROS期间，你会看到提示说需要 source 多个setup.*sh文件中的某一个，或者甚至提示添加这条'source'命令到你的启动脚本里面。这些操作是必须的，因为ROS是依赖于某种组合空间的概念，而这种概念就是通过配置脚本环境来实现的。这可以让针对不同版本或者不同软件包集的开发更加容易。

如果你在查找和使用ROS软件包方面遇到了问题，请确保你已经正确配置了脚本环境。一个检查的好方法是确保你已经设置了像ROS_ROOT和ROS_PACKAGE_PATH这样的环境变量，可以通过以下命令查看：

$ export | grep ROS

如果发现没有配置，那这个时候你就需要'source'某些'setup.*sh’文件了。

ROS会帮你自动生成这些‘setup.*sh’文件，通过以下方式生成并保存在不同地方：

通过类似apt的软件包管理器安装ROS软件包时会生成setup.*sh文件。
在rosbuild workspaces中通过类似rosws的工具生成。
在编译 或 安装 catkin 软件包时自动生成。
 

注意： 在所有教程中你将会经常看到分别针对rosbuild 和 catkin的不同操作说明，这是因为目前有两种不同的方法可以用来组织和编译ROS应用程序。一般而言，rosbuild比较简单也易于使用，而catkin使用了更加标准的CMake规则，所以比较复杂，但是也更加灵活，特别是对于那些想整合外部现有代码或者想发布自己代码的人。关于这些如果你想了解得更全面请参阅catkin or rosbuild。

如果你是通过ubuntu上的 apt 工具来安装ROS的，那么你将会在'/opt/ros/<distro>/'目录中看到setup.*sh文件，然后你可以执行下面的source命令：

# source /opt/ros/
/setup.bash
请使用具体的ROS发行版名称代替<distro>。

比如你安装的是ROS Indigo，则上述命令改为：

$ source /opt/ros/indigo/setup.bash
在每次打开终端时你都需要先运行上面这条命令后才能运行ros相关的命令，为了避免这一繁琐过程，你可以事先在.bashrc文件（初学者请注意：该文件是在当前系统用户的home目录下。）中添加这条命令，这样当你每次登录后系统已经帮你执行这些命令配置好环境。这样做也可以方便你在同一台计算机上安装并随时切换到不同版本的ROS（比如fuerte和groovy）。

此外，你也可以在其它系统平台上相应的ROS安装目录下找到这些setup.*sh文件。

创建ROS工作空间
这些操作方法只适用于ROS Groovy及后期版本，对于ROS Fuerte及早期版本请选择rosbuild。

下面我们开始创建一个catkin 工作空间：

$ mkdir -p ~/catkin_ws/src

$ cd ~/catkin_ws/src

即使这个工作空间是空的（在'src'目录中没有任何软件包，只有一个CMakeLists.txt链接文件），你依然可以编译它：

$ cd ~/catkin_ws/

$ catkin_make

catkin_make命令在catkin 工作空间中是一个非常方便的工具。如果你查看一下当前目录应该能看到'build'和'devel'这两个文件夹。在'devel'文件夹里面你可以看到几个setup.*sh文件。source这些文件中的任何一个都可以将当前工作空间设置在ROS工作环境的最顶层，想了解更多请参考catkin文档。接下来首先source一下新生成的setup.*sh文件：

$ source devel/setup.bash

要想保证工作空间已配置正确需确保ROS_PACKAGE_PATH环境变量包含你的工作空间目录，采用以下命令查看：

ROS_PACKAGE_PATH

/home/<youruser>/catkin_ws/src:/opt/ros/indigo/share:/opt/ros/indigo/stacks

到此你的工作环境已经搭建完成，接下来可以继续学习 ROS文件系统教程.

 

三、测试
为了保险，重启一下，测试测试我们的ROS吧....

ROS是依靠终端来运行的系统。所以还需要打开终端。这里给大家演示一个基本的案例--turtlrsim小海龟。运行下面的命令你便可以看到一个有小海龟的界面。

Roscore

rosrun turtlrsimn turtlesim_node

上面的第二行命令需要从新打开一个终端来输入，因为你会发现当输入roscore运行后终端是无法输入有效的命令来运行的。

 

 

 

如何使用键盘是小海龟运动起来呢？就需要输入下面的命令。

rosrun turtlesim turtle_teleop_key