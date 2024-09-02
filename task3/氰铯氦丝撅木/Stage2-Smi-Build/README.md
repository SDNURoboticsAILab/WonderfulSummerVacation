# 安装 Isaac Sim

官方文档：[https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html](https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html)

文档中提出了安装 Isaac Sim 的[配置要求](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/requirements.html)

| Element | Minimum Spec                              | Good                                      | Ideal                                                        |
| ------- | ----------------------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| OS      | Ubuntu 20.04/22.04Windows 10/11           | Ubuntu 20.04/22.04Windows 10/11           | Ubuntu 20.04/22.04Windows 10/11                              |
| CPU     | Intel Core i7 (7th Generation)AMD Ryzen 5 | Intel Core i7 (9th Generation)AMD Ryzen 7 | Intel Core i9, X-series or higherAMD Ryzen 9, Threadripper or higher |
| Cores   | 4                                         | 8                                         | 16                                                           |
| RAM     | 32GB*                                     | 64GB*                                     | 64GB*                                                        |
| Storage | 50GB SSD                                  | 500GB SSD                                 | 1TB NVMe SSD                                                 |
| GPU     | GeForce RTX 3070                          | GeForce RTX 4080                          | RTX Ada 6000                                                 |
| VRAM    | 8GB*                                      | 16GB*                                     | 48GB*                                                        |

很不幸，我们队成员的电脑都不能满足这个配置要求，因此我们把目光转向了云 GPU 服务器，配置如下：

- **镜像**
  - Miniconda conda3
  - Python 3.10(ubuntu22.04)
  - Cuda 11.8
- **GPU**
  - RTX 3080(10GB) * 1升降配置
- **CPU**
  - 12 vCPU Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz
- **内存**
  - 40GB
- **硬盘**
  - 系统盘:30 GB
  - 数据盘:100 GB

云服务器厂商并没有为我们提供图形化界面，因此需要手动安装桌面环境。

篇幅原因，请看[这个文档](./01-de-install.md)。

---

## 通过 Workstation 安装 Omniverse Launcher

访问 [https://www.nvidia.com/en-us/omniverse/download/](https://www.nvidia.com/en-us/omniverse/download/) 填写问卷，在页面下方获取下载链接。

官方提供了一个 [Windows 的安装教程](https://docs.omniverse.nvidia.com/launcher/latest/installing_launcher.html)。在 Linux 中，安装包以 AppImage 的形式打包，使用 FUSE（[用户空间文件系统](https://en.wikipedia.org/wiki/Filesystem_in_Userspace) 是一个允许非root用户挂载文件系统的系统）进行运行。

### 安装 FUSE

```bash
sudo apt install fuse libfuse2
sudo modprobe fuse
sudo groupadd fuse

user="$(whoami)"
sudo usermod -a -G fuse $user
```

### AppImage，启动！

```bash
chmod u+x ./omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage --appimage-extract
cd squashfs-root
chmod u+x ./omniverse-launcher
./omniverse-launcher
```

注意，[以 root 身份带 `--no-sandbox` 参数运行 AppImage 是不被允许的](https://issues.chromium.org/issues/40480798)，需要创建一个有 sudo 权限的普通用户。

```bash
sudo adduser excnies
sudo usermod -aG sudo excnies
sudo apt install sudo
su - excnies
sudo -l
```

提示权限问题

```bash
excnies@autodl-container-ae0c4bafdd-217fb69d:~/squashfs-root$ ./omniverse-launcher
[22784:0830/163016.755799:FATAL:setuid_sandbox_host.cc(158)] The SUID sandbox helper binary was found, but is not configured correctly. Rather than run without sandboxing I'm aborting now. You need to make sure that /home/excnies/squashfs-root/chrome-sandbox is owned by root and has mode 4755.
Trace/breakpoint trap (core dumped)
```

更改`chrome-sandbox`file to root:文件到根目录

```bash
sudo chown root:root /home/excnies/squashfs-root/chrome-sandbox
```

设置正确的权限（模式4755），`chrome-sandbox`file:文件

```bash
sudo chmod 4755 /home/excnies/squashfs-root/chrome-sandbox
```

出现下面报错

```bash
excnies@autodl-container-ae0c4bafdd-217fb69d:~/squashfs-root$ ./omniverse-launcher
Failed to move to new namespace: PID namespaces supported, Network namespace supported, but failed: errno = Operation not permitted
Trace/breakpoint trap (core dumped)
```

Omniverse Launcher 在尝试移动到新的命名空间时失败了，特别是在网络命名空间方面。这通常与权限或容器环境配置有关。

尝试使用 `--no-sandbox` 标志禁用沙箱功能，无果。

```bash
export NO_SANDBOX=true
./omniverse-launcher
```

翻阅了一些资料，不少论坛和 issue 指出使用 `--priviledged` 启动容器即可，但我们根本没有这个权限。

```bash
echo 'kernel.unprivileged_userns_clone=1' > /etc/sysctl.d/00-local-userns.conf
service procps restart
```

```bash
excnies@autodl-container-ae0c4bafdd-217fb69d:~/squashfs-root$ ./omniverse-launcher --no-sandbox
[23343:0830/164700.086081:ERROR:bus.cc(399)] Failed to connect to the bus: Failed to connect to socket /run/dbus/system_bus_socket: No such file or directory
[23343:0830/164700.402609:ERROR:ozone_platform_x11.cc(240)] Missing X server or $DISPLAY
[23343:0830/164700.402637:ERROR:env.cc(255)] The platform failed to initialize.  Exiting.
[0830/164700.410267:ERROR:file_io_posix.cc(144)] open /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq: No such file or directory (2)
[0830/164700.410313:ERROR:file_io_posix.cc(144)] open /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq: No such file or directory (2)
Segmentation fault (core dumped)
```
这个问题已知为容器安全机制导致，是 Chromium 开发者的问题，暂时无解。

## 清理无用的软件包

### apt

```bash
sudo du -sh /var/cache/apt/archives #查看当前缓存的大小
sudo apt-get autoremove #清理未使用的包
sudo apt-get clean #清理缓存文件
sudo apt-get autoclean #清理过时的包
```

### pip

```bash
pip cache info #查看缓存大小
pip cache purge #清理缓存
```

### conda

```bash
conda clean --list #查看缓存大小
conda clean --packages #清理未使用的包
conda clean --index-cache #清理索引缓存
conda clean --all #清理所有缓存
```

## Python 环境安装

创建虚拟环境

Linux

```bash
python3.10 -m venv env_isaacsim
source env_isaacsim/bin/activate
```

Powershell

```powershell
& "C:\Program Files\Python310\python.exe" -m venv env_isaacsim
& "env_isaacsim\Scripts\Activate.ps1"
```

如果你在执行 `Activate.ps1` 脚本时遇到权限问题，可能需要调整 PowerShell 的执行策略。你可以通过以下命令临时更改执行策略：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 全局换源

```bash
mkdir -p ~/.pip
nano ~/.pip/pip.conf
```

### 安装 Isaac Sim 软件包

```bash
pip install isaacsim==4.1.0.0 --extra-index-url https://pypi.nvidia.com
```

速度太慢，更换清华源

```bash
pip install isaacsim==4.1.0.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 安装 Isaac Sim 相关依赖

```bash
pip install isaacsim-extscache-physics==4.1.0.0 isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://pypi.nvidia.com
```

由于服务器只有 30 G 空间，但是有 50 G 数据盘，将环境迁移到数据盘。

```bash
source env_isaacsim/bin/activate
rm -rf env_isaacsim/bin/activate
deactivate
cd autodl-tmp/
python3.10 -m venv env_isaacsim
source env_isaacsim/bin/activate
```

需要同意相关许可证

```python
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
```

## 安装 Isaac Lab

### 克隆仓库

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
```

### 测试

```bash
cd IsaacLab/
./isaaclab.sh --help

usage: isaaclab.sh [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-c] -- Utility to manage Isaac Lab.

optional arguments:
   -h, --help           Display the help content.
   -i, --install [LIB]  Install the extensions inside Isaac Lab and learning frameworks (rl-games, rsl-rl, sb3, skrl) as extra dependencies. Default is 'all'.
   -f, --format         Run pre-commit to format the code and check lints.
   -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
   -s, --sim            Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.
   -t, --test           Run all python unittest tests.
   -o, --docker         Run the docker container helper script (docker/container.sh).
   -v, --vscode         Generate the VSCode settings file from template.
   -d, --docs           Build the documentation from source using sphinx.
   -c, --conda [NAME]   Create the conda environment for Isaac Lab. Default name is 'isaaclab'.
```

### 创建 Isaac Sim 符号链接

```bash
cd IsaacLab
ln -s path_to_isaac_sim _isaac_sim
```

On Linux systems, by default, Isaac Sim is installed in the directory `${HOME}/.local/share/ov/pkg/isaac_sim-*`, with `*` corresponding to the Isaac Sim version.

### 安装

```bash
# these dependency are needed by robomimic which is not available on Windows
sudo apt install cmake build-essential #安装依赖
./isaaclab.sh --install rl_games  # or "./isaaclab.sh -i rl_games"
```

