# 基于 Docker 的 Ubuntu 22.04 LTS 桌面环境和 VNC 安装及问题排查

安装和运行 Isaac Sim 需要图形化界面，而我们使用的云 GPU 厂商 AutoDL 默认只有 CLI，因此我们需要安装桌面环境（DE），通过 VNC 连接尝试安装 Isaac Sim。

## 安装VNC和必要的一些图形显示库

```bash
# 安装基本的依赖包
apt update && apt install -y libglu1-mesa-dev mesa-utils xterm xauth x11-xkb-utils xfonts-base xkb-data libxtst6 libxv1

# 安装libjpeg-turbo和turbovnc
export TURBOVNC_VERSION=2.2.5
export LIBJPEG_VERSION=2.0.90
wget http://aivc.ks3-cn-beijing.ksyun.com/packages/libjpeg-turbo/libjpeg-turbo-official_${LIBJPEG_VERSION}_amd64.deb
wget http://aivc.ks3-cn-beijing.ksyun.com/packages/turbovnc/turbovnc_${TURBOVNC_VERSION}_amd64.deb
dpkg -i libjpeg-turbo-official_${LIBJPEG_VERSION}_amd64.deb
dpkg -i turbovnc_${TURBOVNC_VERSION}_amd64.deb
rm -rf *.deb

# 启动VNC服务端，这一步可能涉及vnc密码配置（注意不是实例的账户密码）。另外如果出现报错xauth未找到，那么使用apt install xauth再安装一次
rm -rf /tmp/.X1*  # 如果再次启动，删除上一次的临时文件，否则无法正常启动
USER=root /opt/TurboVNC/bin/vncserver :1 -desktop X -auth /root/.Xauthority -geometry 1920x1080 -depth 24 -rfbwait 120000 -rfbauth /root/.vnc/passwd -fp /usr/share/fonts/X11/misc/,/usr/share/fonts -rfbport 6006 # 启动 VNC，如果首次启动或者删除了临时文件，则提示设置 VNC 密码

# 检查是否启动，如果有vncserver的进程，证明已经启动
ps -ef | grep vnc
```

## 使用隧道代理 VNC

以上启动 Server 时，手动设置了 rfbport=6006 端口，下面通过 SSH 隧道将实例中的 6006 端口代理到本地：[SSH隧道](https://www.autodl.com/docs/ssh_proxy/)

### Windows（图形工具）

图形工具：[点击下载](https://autodl-public.ks3-cn-beijing.ksyuncs.com/tool/AutoDL-SSH-Tools.zip)

下载后解压，无需安装点击 .exe 可执行文件即可，输入要代理的端口号，如果有多个端口号用英文逗号`,`分隔

输入 ssh 指令和密码，点击开始代理即可

<img src="https://cfdn-img.hx-cn.top/file/61ac55b0ba9e2f3821965.png" alt="image-20240829200555819" style="zoom: 80%;" />

### SSH代理命令

使用SSH将实例中的端口代理到本地，具体步骤为：

**Step.1** 在**实例中**启动您的服务（比如您的服务监听6006端口，下面以6006端口为例）

**Step.2** 在**本地电脑**的终端(cmd / powershell / terminal等)中执行代理命令：

```
ssh -CNg -L 6006:127.0.0.1:6006 root@123.125.240.150 -p 42151
```

其中`root@123.125.240.150`和`42151`分别是实例中SSH指令的访问地址与端口，请找到自己实例的ssh指令**做相应替换**。`6006:127.0.0.1:6006`是指代理实例内`6006`端口到本地的`6006`端口。

> 注意：执行完这条ssh命令，没有任何日志是正常的，只要没有要求重新输入密码或错误退出
>
> **Windows下的cmd/powershell如果一直提示密码错误，是因为无法粘贴，手动输入即可（正常不会显示正在输入的密码)**

![image-20230313162654913](https://cfdn-img.hx-cn.top/file/d5a341e2bb7bfcdebdcdc.png)

**Step.3** 在本地浏览器中访问`http://127.0.0.1:6006`即可打开服务，注意这里的`6006`端口要和上述`6006:127.0.0.1:6006`中的端口保持一致

![image-20230313162636526](https://cfdn-img.hx-cn.top/file/abfbd51208f8265dee5bb.png)

## 安装桌面环境

Cinnamon 是一个免费使用的 X Window 系统桌面环境，最初是从 GNOME 桌面派生出来的。它是 Linux 桌面的最佳桌面环境之一，专为速度、灵活性和先进的创新功能而设计。

查看服务器信息，系统为 Ubuntu 22.04 LTS，Cinnamon 是一个轻量且兼容性强的解决方案。

```bash
root@autodl-container-a44e11a152-0614e516:~# cat /etc/os-release
PRETTY_NAME="Ubuntu 22.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="22.04"
VERSION="22.04.4 LTS (Jammy Jellyfish)"
VERSION_CODENAME=jammy
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=jammy
```

适用于 Ubuntu 的 Cinnamon 桌面环境包可在操作系统存储库中找到，只需更新缓存并开始安装。

```bash
sudo apt update
sudo apt install cinnamon-desktop-environment 
```

安装后重新启动系统

```bash
sudo reboot
```

但是当我们使用 VNC Viewer 等客户端连接到服务器，连接是黑屏的。

![865539097651806c3f3ea244b1ed3cf1](https://cfdn-img.hx-cn.top/file/4149329ba36b4d6563cbe.png)

## 原因排查

### 检查日志文件

查看VNC服务器的日志文件，通常位于`~/.vnc/`目录下，文件名通常是`hostname:display_number.log`（例如`localhost:1.log`）。检查日志文件以获取有关黑屏原因的线索。

```bash
root@autodl-container-a44e11a152-0614e516:~/.vnc# cat autodl-container-a44e11a152-0614e516\:1.log
TurboVNC Server (Xvnc) 64-bit v2.2.5 (build 20200507)
Copyright (C) 1999-2020 The VirtualGL Project and many others (see README.txt)
Visit http://www.TurboVNC.org for more information on TurboVNC

29/08/2024 16:17:47 Using security configuration file /etc/turbovncserver-security.conf
29/08/2024 16:17:47 Enabled security type 'tlsvnc'
29/08/2024 16:17:47 Enabled security type 'tlsotp'
29/08/2024 16:17:47 Enabled security type 'tlsplain'
29/08/2024 16:17:47 Enabled security type 'x509vnc'
29/08/2024 16:17:47 Enabled security type 'x509otp'
29/08/2024 16:17:47 Enabled security type 'x509plain'
29/08/2024 16:17:47 Enabled security type 'vnc'
29/08/2024 16:17:47 Enabled security type 'otp'
29/08/2024 16:17:47 Enabled security type 'unixlogin'
29/08/2024 16:17:47 Enabled security type 'plain'
29/08/2024 16:17:47 Desktop name 'X' (autodl-container-a44e11a152-0614e516:1)
29/08/2024 16:17:47 Protocol versions supported: 3.3, 3.7, 3.8, 3.7t, 3.8t
29/08/2024 16:17:47 Listening for VNC connections on TCP port 6006
29/08/2024 16:17:47   Interface 0.0.0.0
29/08/2024 16:17:47 Listening for HTTP connections on TCP port 5801
29/08/2024 16:17:47   URL http://autodl-container-a44e11a152-0614e516:5801
29/08/2024 16:17:47   Interface 0.0.0.0
29/08/2024 16:17:47 Framebuffer: BGRX 8/8/8/8
29/08/2024 16:17:47 New desktop size: 1920 x 1080
29/08/2024 16:17:47 New screen layout:
29/08/2024 16:17:47   0x00000040 (output 0x00000040): 1920x1080+0+0
29/08/2024 16:17:47 Maximum clipboard transfer size: 1048576 bytes
29/08/2024 16:17:47 VNC extension running!

29/08/2024 16:18:07 Got connection from client 127.0.0.1
29/08/2024 16:18:07 Using protocol version 3.8
29/08/2024 16:18:11 Full-control authentication enabled for 127.0.0.1
29/08/2024 16:18:11 rfbProcessClientNormalMessage: ignoring unknown encoding 24 (18)
29/08/2024 16:18:11 Using ZRLE encoding for client 127.0.0.1
29/08/2024 16:18:11 rfbProcessClientNormalMessage: ignoring unknown encoding 22 (16)
29/08/2024 16:18:11 rfbProcessClientNormalMessage: ignoring unknown encoding 21 (15)
29/08/2024 16:18:11 rfbProcessClientNormalMessage: ignoring unknown encoding 15 (f)
29/08/2024 16:18:11 rfbProcessClientNormalMessage: ignoring unknown encoding -314 (fffffec6)
29/08/2024 16:18:11 Enabling full-color cursor updates for client 127.0.0.1
29/08/2024 16:18:11 Enabling Desktop Size protocol extension for client 127.0.0.1
29/08/2024 16:18:11 Pixel format for client 127.0.0.1:
29/08/2024 16:18:11   8 bpp, depth 6
29/08/2024 16:18:11   true colour: max r 3 g 3 b 3, shift r 4 g 2 b 0
29/08/2024 16:18:12 Using raw encoding for client 127.0.0.1
29/08/2024 16:18:12 rfbProcessClientNormalMessage: ignoring unknown encoding 24 (18)
29/08/2024 16:18:12 rfbProcessClientNormalMessage: ignoring unknown encoding 22 (16)
29/08/2024 16:18:12 rfbProcessClientNormalMessage: ignoring unknown encoding 21 (15)
29/08/2024 16:18:12 rfbProcessClientNormalMessage: ignoring unknown encoding 15 (f)
29/08/2024 16:18:12 rfbProcessClientNormalMessage: ignoring unknown encoding -314 (fffffec6)
29/08/2024 16:18:12 Enabling full-color cursor updates for client 127.0.0.1
29/08/2024 16:18:12 rfbProcessClientNormalMessage: ignoring unknown encoding 24 (18)
29/08/2024 16:18:12 Using ZRLE encoding for client 127.0.0.1
29/08/2024 16:18:12 rfbProcessClientNormalMessage: ignoring unknown encoding 22 (16)
29/08/2024 16:18:12 rfbProcessClientNormalMessage: ignoring unknown encoding 21 (15)
29/08/2024 16:18:12 rfbProcessClientNormalMessage: ignoring unknown encoding 15 (f)
29/08/2024 16:18:12 rfbProcessClientNormalMessage: ignoring unknown encoding -314 (fffffec6)
29/08/2024 16:18:12 Enabling full-color cursor updates for client 127.0.0.1
29/08/2024 16:18:12 Pixel format for client 127.0.0.1:
29/08/2024 16:18:12   32 bpp, depth 24, little endian
29/08/2024 16:18:12   true colour: max r 255 g 255 b 255, shift r 16 g 8 b 0
29/08/2024 16:18:12   no translation needed
```

从日志文件来看，VNC服务器（TurboVNC）已经成功启动，并且有客户端连接到它。日志中显示了多个安全类型的启用、协议版本的使用、编码方式的选择以及客户端的连接信息。然而，日志中也有一些关于未知编码的警告信息，这些可能是由于客户端和服务器之间的兼容性问题。

尝试在启动VNC服务器时指定特定的编码方式，例如：

```bash
/opt/TurboVNC/bin/vncserver :1 -desktop X -auth /root/.Xauthority -geometry 1920x1080 -depth 24 -rfbwait 120000 -rfbauth /root/.vnc/passwd -fp /usr/share/fonts/X11/misc/,/usr/share/fonts -rfbport 6006 -encodings "copyrect hextile zlib corre rre raw"
```

返回报错，这种启动方式无效。

### 检查Xorg服务器

确保Xorg服务器正确启动并且没有错误。在启动VNC服务器时添加`-verbose`选项来获取更多详细信息：

```bash
/opt/TurboVNC/bin/vncserver :1 -desktop X -auth /root/.Xauthority -geometry 1920x1080 -depth 24 -rfbwait 120000 -rfbauth /root/.vnc/passwd -fp /usr/share/fonts/X11/misc/,/usr/share/fonts -rfbport 6006 -verbose
```

返回以下日志

```bash
c# DISPLAY=:1 cinnamon-session
cinnamon-session[1512]: WARNING: t+0.00436s: Failed to connect to system bus: Could not connect: No such file or directory
cinnamon-session[1512]: WARNING: t+0.00454s: Could not get session id for session. Check that logind is properly installed and pam_systemd is getting used at login.
cinnamon-session[1512]: WARNING: t+0.01026s: Error while executing session-migration: Failed to close file descriptor for child process (Operation not permitted)
cinnamon-session[1512]: WARNING: t+0.01546s: Failed to start app: Unable to start application: Failed to close file descriptor for child process (Operation not permitted)
```

从日志信息来看，Cinnamon桌面环境在尝试启动时遇到了一些问题，主要与系统总线（system bus）连接失败和文件描述符关闭失败有关。

### 检查DBus服务

DBus是Linux系统中用于进程间通信的重要服务。确保DBus服务正在运行：

```
systemctl status dbus
```

如果DBus服务没有运行，可以尝试启动它：

```
systemctl start dbus
```

```bash
root@autodl-container-a44e11a152-0614e516:~# systemctl status dbus
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to connect to bus: Host is down
root@autodl-container-a44e11a152-0614e516:~# systemctl start dbus
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to connect to bus: Host is down
root@autodl-container-a44e11a152-0614e516:~# systemctl status systemd-logind
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to connect to bus: Host is down
root@autodl-container-a44e11a152-0614e516:~# systemctl start systemd-logind
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to connect to bus: Host is down
```

从输出信息来看，系统似乎没有使用 systemd 作为 init 系统（PID 1）。这通常意味着服务器正在一个不支持 systemd 的容器环境中运行，例如 Docker 容器。在这种情况下，systemd 服务（如 dbus 和 systemd-logind）将不可用。

在非systemd环境中，可以使用其他方法来管理服务和会话。

### 使用替代的会话管理工具

`consolekit`或`elogind`。安装并配置这些工具来管理用户会话。

- 安装`consolekit`：

  ```bash
  apt-get install consolekit
  ```

- 安装`elogind`：

  ```bash
  apt-get install elogind
  ```

这是 GPT 给出的一个解决方案，但是我们服务器的软件源中没有这两个软件包，PPA 仓库和 snap 也无法安装这两个软件包

> 1. **添加PPA仓库**：
>    你可以添加一个包含elogind的PPA（个人包档案）仓库。例如，你可以使用`ppa:levente-szabo/elogind`仓库：
>
>    ```bash
>    sudo add-apt-repository ppa:levente-szabo/elogind
>    sudo apt update
>    sudo apt install elogind
>    ```
>
> 2. **使用Snap包**：
>    另一种方法是使用Snap包管理器来安装elogind。首先，确保你已经安装了Snap
>
>    ```bash
>    sudo apt install snapd
>    ```
>
>    然后，你可以尝试安装elogind的Snap包：
>
>    ```bash
>    sudo snap install elogind
>    ```
>
> 3. **手动编译安装**：
>    如果上述方法都不适用，你可以从源代码手动编译和安装elogind。这需要一些额外的步骤和依赖项。你可以从elogind的GitHub仓库获取源代码：
>
> 请注意，手动编译和安装可能需要更多的系统配置和依赖项管理。确保你了解这些步骤并准备好处理可能出现的问题。

我们决定使用源码安装：
```bash
sudo apt install build-essential libsystemd-dev pkg-config
git clone https://github.com/elogind/elogind.git
cd elogind
meson build
ninja -C build
sudo ninja -C build install
```

根据报错安装缺少的依赖

```bash
sudo apt update
sudo apt install meson ninja-build gperf libcap-dev libmount-dev libudev-dev
```

编译成功，但是服务不能启动

```bash
ninja: Entering directory `build'
[543/543] Linking target test-bus-watch-bind
ninja: Entering directory `build'
[1/2] Installing files.
Installing po/be/LC_MESSAGES/elogind.mo to /usr/share/locale/be/LC_MESSAGES
Installing po/be@latin/LC_MESSAGES/elogind.mo to /usr/share/locale/be@latin/LC_MESSAGES
Installing po/bg/LC_MESSAGES/elogind.mo to /usr/share/locale/bg/LC_MESSAGES
Installing po/ca/LC_MESSAGES/elogind.mo to /usr/share/locale/ca/LC_MESSAGES
Installing po/cs/LC_MESSAGES/elogind.mo to /usr/share/locale/cs/LC_MESSAGES
Installing po/da/LC_MESSAGES/elogind.mo to /usr/share/locale/da/LC_MESSAGES
Installing po/de/LC_MESSAGES/elogind.mo to /usr/share/locale/de/LC_MESSAGES
Installing po/el/LC_MESSAGES/elogind.mo to /usr/share/locale/el/LC_MESSAGES
Installing po/es/LC_MESSAGES/elogind.mo to /usr/share/locale/es/LC_MESSAGES
Installing po/et/LC_MESSAGES/elogind.mo to /usr/share/locale/et/LC_MESSAGES
Installing po/fi/LC_MESSAGES/elogind.mo to /usr/share/locale/fi/LC_MESSAGES
Installing po/fr/LC_MESSAGES/elogind.mo to /usr/share/locale/fr/LC_MESSAGES
Installing po/gl/LC_MESSAGES/elogind.mo to /usr/share/locale/gl/LC_MESSAGES
Installing po/hr/LC_MESSAGES/elogind.mo to /usr/share/locale/hr/LC_MESSAGES
Installing po/hu/LC_MESSAGES/elogind.mo to /usr/share/locale/hu/LC_MESSAGES
Installing po/id/LC_MESSAGES/elogind.mo to /usr/share/locale/id/LC_MESSAGES
Installing po/it/LC_MESSAGES/elogind.mo to /usr/share/locale/it/LC_MESSAGES
Installing po/ja/LC_MESSAGES/elogind.mo to /usr/share/locale/ja/LC_MESSAGES
Installing po/ka/LC_MESSAGES/elogind.mo to /usr/share/locale/ka/LC_MESSAGES
Installing po/kab/LC_MESSAGES/elogind.mo to /usr/share/locale/kab/LC_MESSAGES
Installing po/ko/LC_MESSAGES/elogind.mo to /usr/share/locale/ko/LC_MESSAGES
Installing po/lt/LC_MESSAGES/elogind.mo to /usr/share/locale/lt/LC_MESSAGES
Installing po/nl/LC_MESSAGES/elogind.mo to /usr/share/locale/nl/LC_MESSAGES
Installing po/pa/LC_MESSAGES/elogind.mo to /usr/share/locale/pa/LC_MESSAGES
Installing po/pl/LC_MESSAGES/elogind.mo to /usr/share/locale/pl/LC_MESSAGES
Installing po/pt/LC_MESSAGES/elogind.mo to /usr/share/locale/pt/LC_MESSAGES
Installing po/pt_BR/LC_MESSAGES/elogind.mo to /usr/share/locale/pt_BR/LC_MESSAGES
Installing po/ro/LC_MESSAGES/elogind.mo to /usr/share/locale/ro/LC_MESSAGES
Installing po/ru/LC_MESSAGES/elogind.mo to /usr/share/locale/ru/LC_MESSAGES
Installing po/si/LC_MESSAGES/elogind.mo to /usr/share/locale/si/LC_MESSAGES
Installing po/sk/LC_MESSAGES/elogind.mo to /usr/share/locale/sk/LC_MESSAGES
Installing po/sr/LC_MESSAGES/elogind.mo to /usr/share/locale/sr/LC_MESSAGES
Installing po/sv/LC_MESSAGES/elogind.mo to /usr/share/locale/sv/LC_MESSAGES
Installing po/tr/LC_MESSAGES/elogind.mo to /usr/share/locale/tr/LC_MESSAGES
Installing po/uk/LC_MESSAGES/elogind.mo to /usr/share/locale/uk/LC_MESSAGES
Installing po/zh_CN/LC_MESSAGES/elogind.mo to /usr/share/locale/zh_CN/LC_MESSAGES
Installing po/zh_TW/LC_MESSAGES/elogind.mo to /usr/share/locale/zh_TW/LC_MESSAGES
Installing po/eu/LC_MESSAGES/elogind.mo to /usr/share/locale/eu/LC_MESSAGES
Installing po/he/LC_MESSAGES/elogind.mo to /usr/share/locale/he/LC_MESSAGES
Installing src/libelogind/libelogind.pc to /usr/lib/x86_64-linux-gnu/pkgconfig
Installing src/shared/libelogind-shared-255.so to /usr/lib/x86_64-linux-gnu/elogind
Installing libelogind.so.0.38.0 to /usr/lib/x86_64-linux-gnu
Installing symlink pointing to libelogind.so.0.38.0 to /usr/lib/x86_64-linux-gnu/libelogind.so.0
Installing symlink pointing to libelogind.so.0 to /usr/lib/x86_64-linux-gnu/libelogind.so
Installing src/login/logind.conf to /etc/elogind
Installing src/login/elogind-user to /usr/lib/pam.d
Installing src/login/org.freedesktop.login1.service to /usr/share/dbus-1/system-services
Installing busctl to /usr/bin
Installing elogind-cgroups-agent to /usr/libexec
Installing elogind to /usr/libexec
Installing loginctl to /usr/bin
Installing elogind-inhibit to /usr/bin
Installing elogind-uaccess-command to /usr/libexec
Installing rules.d/71-seat.rules to /usr/lib/udev/rules.d
Installing rules.d/73-seat-late.rules to /usr/lib/udev/rules.d
Installing /root/elogind/src/systemd/sd-bus.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-bus-protocol.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-bus-vtable.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-daemon.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-device.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-event.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-gpt.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-hwdb.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-id128.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-journal.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-login.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-messages.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/sd-path.h to /usr/include/elogind/systemd
Installing /root/elogind/src/systemd/_sd-common.h to /usr/include/elogind/systemd
Installing new directory /usr/libexec/system-shutdown
Installing new directory /usr/libexec/system-sleep
Installing new directory /var/lib/elogind
Installing /root/elogind/src/sleep/sleep.conf to /etc/elogind
Installing /root/elogind/src/sleep/10-elogind.conf to /etc/elogind/sleep.conf.d
Installing /root/elogind/src/login/org.freedesktop.login1.conf to /usr/share/dbus-1/system.d
Installing /root/elogind/src/login/org.freedesktop.login1.policy to /usr/share/polkit-1/actions
Installing /root/elogind/rules.d/70-power-switch.rules to /usr/lib/udev/rules.d
Installing /root/elogind/shell-completion/bash/busctl to /usr/share/bash-completion/completions
Installing /root/elogind/shell-completion/bash/loginctl to /usr/share/bash-completion/completions
Installing /root/elogind/shell-completion/zsh/_busctl to /usr/share/zsh/site-functions
Installing /root/elogind/shell-completion/zsh/_loginctl to /usr/share/zsh/site-functions
Installing /root/elogind/LICENSE.GPL2 to /usr/share/doc/elogind
Installing /root/elogind/LICENSE.LGPL2.1 to /usr/share/doc/elogind
Installing /root/elogind/README.md to /usr/share/doc/elogind
Installing /root/elogind/docs/CODING_STYLE.md to /usr/share/doc/elogind
Running custom install script '/root/elogind/tools/meson-symlink_headers.sh /usr/include sd-bus.h sd-bus-protocol.h sd-bus-vtable.h sd-daemon.h sd-device.h sd-event.h sd-gpt.h sd-hwdb.h sd-id128.h sd-journal.h sd-login.h sd-messages.h sd-path.h _sd-common.h'
root@autodl-container-a44e11a152-0614e516:~/elogind# sudo systemctl enable elogind
sudo systemctl start elogind
Failed to enable unit, unit elogind.service does not exist.
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to connect to bus: Host is down
```


### 手动启动 DBus

在非systemd环境中，可以手动启动DBus守护进程：

```bash
root@autodl-container-a44e11a152-0614e516:~# /usr/bin/dbus-daemon --system --fork
dbus-daemon[1555]: Failed to start message bus: Failed to bind socket "/run/dbus/system_bus_socket": No such file or directory
```

1. **创建必要的目录和文件**：
   确保`/run/dbus`目录存在并且有正确的权限。运行以下命令来创建这个目录：

   ```bash
   sudo mkdir -p /run/dbus
   sudo chown messagebus:messagebus /run/dbus
   sudo chmod 755 /run/dbus
   ```

2. **安装D-Bus**：
   确保D-Bus已经正确安装。使用包管理器来安装D-Bus：

   ```bash
   sudo apt update
   sudo apt install dbus
   ```

3. **启动D-Bus服务**：
   如果你使用的是systemd，尝试启动D-Bus服务：

   ```bash
   sudo systemctl start dbus
   ```

```bash
root@autodl-container-a44e11a152-0614e516:~# sudo systemctl start dbus
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to connect to bus: Host is down
```

系统没有使用systemd作为init系统，因此无法通过`systemctl`命令来启动D-Bus服务。如果系统使用的是其他init系统（例如SysVinit或OpenRC），需要使用相应的命令来启动D-Bus服务。

对于 SysVinit，可以使用`service`命令来启动D-Bus服务：

```bash
root@autodl-container-a44e11a152-0614e516:~# DISPLAY=:1 cinnamon-session

** (cinnamon-session:1869): WARNING **: 21:18:14.574: Cannot open display:
root@autodl-container-a44e11a152-0614e516:~# sudo service dbus start
 * Starting system message bus dbus                                                                              [ OK ]
```

### 检查Cinnamon桌面环境的配置：

确保Cinnamon桌面环境的配置文件正确，并且没有依赖于systemd的功能。你可以尝试手动启动Cinnamon会话：

```
DISPLAY=:1 cinnamon-session
```

#### 检查环境变量：

确保环境变量正确设置，特别是`DISPLAY`变量：

```
export DISPLAY=:1
```

#### 检查Xorg服务器：

确保Xorg服务器正确启动并且没有错误。你可以尝试手动启动Xorg服务器：

```
root@autodl-container-a44e11a152-0614e516:~# Xorg :1

X.Org X Server 1.21.1.4
X Protocol Version 11, Revision 0
Current Operating System: Linux autodl-container-a44e11a152-0614e516 5.4.0-126-generic #142-Ubuntu SMP Fri Aug 26 12:12:57 UTC 2022 x86_64
Kernel command line: BOOT_IMAGE=/boot/vmlinuz-5.4.0-126-generic root=UUID=b986dc3b-6b82-44d5-acb8-6cbad5e357d5 ro net.ifnames=0 biosdevname=0 console=ttyS0,115200 console=tty0 panic=5 intel_idle.max_cstate=1 intel_pstate=disable processor.max_cstate=1 amd_iommu=on iommu=pt crashkernel=2G-16G:512M,16G-:768M cgroup_enable=memory swapaccount=1
xorg-server 2:21.1.4-2ubuntu1.7~22.04.11 (For technical support please see http://www.ubuntu.com/support)
Current version of pixman: 0.40.0
        Before reporting problems, check http://wiki.x.org
        to make sure that you have the latest version.
Markers: (--) probed, (**) from config file, (==) default setting,
        (++) from command line, (!!) notice, (II) informational,
        (WW) warning, (EE) error, (NI) not implemented, (??) unknown.
(==) Log file: "/var/log/Xorg.1.log", Time: Thu Aug 29 21:23:15 2024
(==) Using system config directory "/usr/share/X11/xorg.conf.d"
xf86EnableIO: failed to enable I/O ports 0000-03ff (Operation not permitted)
vesa: Ignoring device with a bound kernel driver
(EE)
Fatal server error:
(EE) no screens found(EE)
(EE)
Please consult the The X.Org Foundation support
         at http://wiki.x.org
 for help.
(EE) Please also check the log file at "/var/log/Xorg.1.log" for additional information.
(EE)
(EE) Server terminated with error (1). Closing log file.
```

查看日志

```bash
root@autodl-container-a44e11a152-0614e516:~# cat /var/log/Xorg.1.log
[3040311.469]
X.Org X Server 1.21.1.4
X Protocol Version 11, Revision 0
[3040311.469] Current Operating System: Linux autodl-container-a44e11a152-0614e516 5.4.0-126-generic #142-Ubuntu SMP Fri Aug 26 12:12:57 UTC 2022 x86_64
[3040311.469] Kernel command line: BOOT_IMAGE=/boot/vmlinuz-5.4.0-126-generic root=UUID=b986dc3b-6b82-44d5-acb8-6cbad5e357d5 ro net.ifnames=0 biosdevname=0 console=ttyS0,115200 console=tty0 panic=5 intel_idle.max_cstate=1 intel_pstate=disable processor.max_cstate=1 amd_iommu=on iommu=pt crashkernel=2G-16G:512M,16G-:768M cgroup_enable=memory swapaccount=1
[3040311.469] xorg-server 2:21.1.4-2ubuntu1.7~22.04.11 (For technical support please see http://www.ubuntu.com/support)
[3040311.469] Current version of pixman: 0.40.0
[3040311.469]   Before reporting problems, check http://wiki.x.org
        to make sure that you have the latest version.
[3040311.469] Markers: (--) probed, (**) from config file, (==) default setting,
        (++) from command line, (!!) notice, (II) informational,
        (WW) warning, (EE) error, (NI) not implemented, (??) unknown.
[3040311.469] (==) Log file: "/var/log/Xorg.1.log", Time: Thu Aug 29 21:23:15 2024
[3040311.470] (==) Using system config directory "/usr/share/X11/xorg.conf.d"
[3040311.470] (==) No Layout section.  Using the first Screen section.
[3040311.470] (==) No screen section available. Using defaults.
[3040311.470] (**) |-->Screen "Default Screen Section" (0)
[3040311.470] (**) |   |-->Monitor "<default monitor>"
[3040311.470] (==) No monitor specified for screen "Default Screen Section".
        Using a default monitor configuration.
[3040311.470] (==) Automatically adding devices
[3040311.470] (==) Automatically enabling devices
[3040311.470] (==) Automatically adding GPU devices
[3040311.470] (==) Automatically binding GPU devices
[3040311.470] (==) Max clients allowed: 256, resource mask: 0x1fffff
[3040311.470] (WW) The directory "/usr/share/fonts/X11/cyrillic" does not exist.
[3040311.470]   Entry deleted from font path.
[3040311.470] (WW) The directory "/usr/share/fonts/X11/100dpi/" does not exist.
[3040311.470]   Entry deleted from font path.
[3040311.470] (WW) The directory "/usr/share/fonts/X11/75dpi/" does not exist.
[3040311.470]   Entry deleted from font path.
[3040311.470] (WW) The directory "/usr/share/fonts/X11/100dpi" does not exist.
[3040311.470]   Entry deleted from font path.
[3040311.470] (WW) The directory "/usr/share/fonts/X11/75dpi" does not exist.
[3040311.470]   Entry deleted from font path.
[3040311.470] (==) FontPath set to:
        /usr/share/fonts/X11/misc,
        /usr/share/fonts/X11/Type1,
        built-ins
[3040311.470] (==) ModulePath set to "/usr/lib/xorg/modules"
[3040311.470] (II) The server relies on udev to provide the list of input devices.
        If no devices become available, reconfigure udev or disable AutoAddDevices.
[3040311.470] (II) Loader magic: 0x555c06a02020
[3040311.470] (II) Module ABI versions:
[3040311.470]   X.Org ANSI C Emulation: 0.4
[3040311.470]   X.Org Video Driver: 25.2
[3040311.470]   X.Org XInput driver : 24.4
[3040311.470]   X.Org Server Extension : 10.0
[3040336.495] (EE) systemd-logind: failed to get session: Did not receive a reply. Possible causes include: the remote application did not send a reply, the message bus security policy blocked the reply, the reply timeout expired, or the network connection was broken.
[3040336.496] (II) xfree86: Adding drm device (/dev/dri/card0)
[3040336.496] (II) Platform probe for /sys/devices/pci0000:00/0000:00:1c.3/0000:04:00.0/0000:05:00.0/drm/card0
[3040336.496] (II) xfree86: Adding drm device (/dev/dri/card1)
[3040336.496] (II) Platform probe for /sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0/0000:3c:00.0/0000:3d:00.0/0000:3e:00.0/0000:3f:00.0/drm/card1
[3040336.496] (II) xfree86: Adding drm device (/dev/dri/card2)
[3040336.496] (II) Platform probe for /sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0/0000:3c:00.0/0000:3d:00.0/0000:3e:10.0/0000:40:00.0/drm/card2
[3040336.496] (II) xfree86: Adding drm device (/dev/dri/card3)
[3040336.496] (II) Platform probe for /sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0/0000:3c:04.0/0000:41:00.0/0000:42:00.0/0000:43:00.0/drm/card3
[3040336.496] (II) xfree86: Adding drm device (/dev/dri/card4)
[3040336.496] (II) Platform probe for /sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0/0000:3c:08.0/0000:45:00.0/0000:46:10.0/0000:47:00.0/drm/card4
[3040336.497] (II) xfree86: Adding drm device (/dev/dri/card5)
[3040336.497] (II) Platform probe for /sys/devices/pci0000:85/0000:85:00.0/0000:86:00.0/0000:87:00.0/0000:88:00.0/0000:89:00.0/0000:8a:00.0/drm/card5
[3040336.497] (II) xfree86: Adding drm device (/dev/dri/card6)
[3040336.497] (II) Platform probe for /sys/devices/pci0000:85/0000:85:00.0/0000:86:00.0/0000:87:00.0/0000:88:00.0/0000:89:10.0/0000:8b:00.0/drm/card6
[3040336.497] (II) xfree86: Adding drm device (/dev/dri/card7)
[3040336.497] (II) Platform probe for /sys/devices/pci0000:85/0000:85:00.0/0000:86:00.0/0000:87:04.0/0000:8c:00.0/0000:8d:00.0/0000:8e:00.0/drm/card7
[3040336.497] (II) xfree86: Adding drm device (/dev/dri/card8)
[3040336.497] (II) Platform probe for /sys/devices/pci0000:85/0000:85:00.0/0000:86:00.0/0000:87:08.0/0000:90:00.0/0000:91:10.0/0000:92:00.0/drm/card8
[3040336.528] (--) PCI:*(5@0:0:0) 1a03:2000:1a03:2000 rev 65, Mem @ 0x9c000000/16777216, 0x9d000000/131072, I/O @ 0x00002000/128, BIOS @ 0x????????/131072
[3040336.528] (--) PCI: (63@0:0:0) 10de:2216:10de:1539 rev 161, Mem @ 0xb3000000/16777216, 0x6ffa0000000/268435456, 0x6ffb0000000/33554432, I/O @ 0x00005000/128, BIOS @ 0x????????/524288
[3040336.528] (--) PCI: (64@0:0:0) 10de:2216:10de:1539 rev 161, Mem @ 0xb1000000/16777216, 0x6ff80000000/268435456, 0x6ff90000000/33554432, I/O @ 0x00004000/128, BIOS @ 0x????????/524288
[3040336.528] (--) PCI: (67@0:0:0) 10de:2216:10de:1539 rev 161, Mem @ 0xb7000000/16777216, 0x6ffe0000000/268435456, 0x6fff0000000/33554432, I/O @ 0x00007000/128, BIOS @ 0x????????/524288
[3040336.528] (--) PCI: (71@0:0:0) 10de:2216:10de:1539 rev 161, Mem @ 0xb5000000/16777216, 0x6ffc0000000/268435456, 0x6ffd0000000/33554432, I/O @ 0x00006000/128, BIOS @ 0x????????/524288
[3040336.528] (--) PCI: (138@0:0:0) 10de:2216:10de:1539 rev 161, Mem @ 0xdb000000/16777216, 0x9ffa0000000/268435456, 0x9ffb0000000/33554432, I/O @ 0x0000d000/128, BIOS @ 0x????????/524288
[3040336.528] (--) PCI: (139@0:0:0) 10de:2216:10de:1539 rev 161, Mem @ 0xd9000000/16777216, 0x9ff80000000/268435456, 0x9ff90000000/33554432, I/O @ 0x0000c000/128, BIOS @ 0x????????/524288
[3040336.528] (--) PCI: (142@0:0:0) 10de:2216:10de:1539 rev 161, Mem @ 0xdd000000/16777216, 0x9ffe0000000/268435456, 0x9fff0000000/33554432, I/O @ 0x0000b000/128, BIOS @ 0x????????/524288
[3040336.528] (--) PCI: (146@0:0:0) 10de:2216:10de:1539 rev 161, Mem @ 0xdf000000/16777216, 0x9ffc0000000/268435456, 0x9ffd0000000/33554432, I/O @ 0x0000e000/128, BIOS @ 0x????????/524288
[3040336.528] (II) LoadModule: "glx"
[3040336.528] (II) Loading /usr/lib/xorg/modules/extensions/libglx.so
[3040336.529] (II) Module glx: vendor="X.Org Foundation"
[3040336.529]   compiled for 1.21.1.4, module version = 1.0.0
[3040336.529]   ABI class: X.Org Server Extension, version 10.0
[3040336.529] (==) Matched ast as autoconfigured driver 0
[3040336.529] (==) Matched modesetting as autoconfigured driver 1
[3040336.529] (==) Matched fbdev as autoconfigured driver 2
[3040336.529] (==) Matched vesa as autoconfigured driver 3
[3040336.529] (==) Assigned the driver to the xf86ConfigLayout
[3040336.529] (II) LoadModule: "ast"
[3040336.529] (WW) Warning, couldn't open module ast
[3040336.529] (EE) Failed to load module "ast" (module does not exist, 0)
[3040336.529] (II) LoadModule: "modesetting"
[3040336.529] (II) Loading /usr/lib/xorg/modules/drivers/modesetting_drv.so
[3040336.529] (II) Module modesetting: vendor="X.Org Foundation"
[3040336.529]   compiled for 1.21.1.4, module version = 1.21.1
[3040336.529]   Module class: X.Org Video Driver
[3040336.529]   ABI class: X.Org Video Driver, version 25.2
[3040336.529] (II) LoadModule: "fbdev"
[3040336.529] (II) Loading /usr/lib/xorg/modules/drivers/fbdev_drv.so
[3040336.529] (II) Module fbdev: vendor="X.Org Foundation"
[3040336.529]   compiled for 1.21.1.3, module version = 0.5.0
[3040336.529]   Module class: X.Org Video Driver
[3040336.529]   ABI class: X.Org Video Driver, version 25.2
[3040336.529] (II) LoadModule: "vesa"
[3040336.529] (II) Loading /usr/lib/xorg/modules/drivers/vesa_drv.so
[3040336.529] (II) Module vesa: vendor="X.Org Foundation"
[3040336.529]   compiled for 1.21.1.3, module version = 2.5.0
[3040336.529]   Module class: X.Org Video Driver
[3040336.529]   ABI class: X.Org Video Driver, version 25.2
[3040336.529] (==) Matched ast as autoconfigured driver 0
[3040336.529] (==) Matched modesetting as autoconfigured driver 1
[3040336.529] (==) Matched fbdev as autoconfigured driver 2
[3040336.530] (==) Matched vesa as autoconfigured driver 3
[3040336.530] (==) Assigned the driver to the xf86ConfigLayout
[3040336.530] (II) LoadModule: "ast"
[3040336.530] (WW) Warning, couldn't open module ast
[3040336.530] (EE) Failed to load module "ast" (module does not exist, 0)
[3040336.530] (II) LoadModule: "modesetting"
[3040336.530] (II) Loading /usr/lib/xorg/modules/drivers/modesetting_drv.so
[3040336.530] (II) Module modesetting: vendor="X.Org Foundation"
[3040336.530]   compiled for 1.21.1.4, module version = 1.21.1
[3040336.530]   Module class: X.Org Video Driver
[3040336.530]   ABI class: X.Org Video Driver, version 25.2
[3040336.530] (II) UnloadModule: "modesetting"
[3040336.530] (II) Unloading modesetting
[3040336.530] (II) Failed to load module "modesetting" (already loaded, 0)
[3040336.530] (II) LoadModule: "fbdev"
[3040336.530] (II) Loading /usr/lib/xorg/modules/drivers/fbdev_drv.so
[3040336.530] (II) Module fbdev: vendor="X.Org Foundation"
[3040336.530]   compiled for 1.21.1.3, module version = 0.5.0
[3040336.530]   Module class: X.Org Video Driver
[3040336.530]   ABI class: X.Org Video Driver, version 25.2
[3040336.530] (II) UnloadModule: "fbdev"
[3040336.530] (II) Unloading fbdev
[3040336.530] (II) Failed to load module "fbdev" (already loaded, 0)
[3040336.530] (II) LoadModule: "vesa"
[3040336.530] (II) Loading /usr/lib/xorg/modules/drivers/vesa_drv.so
[3040336.530] (II) Module vesa: vendor="X.Org Foundation"
[3040336.530]   compiled for 1.21.1.3, module version = 2.5.0
[3040336.530]   Module class: X.Org Video Driver
[3040336.530]   ABI class: X.Org Video Driver, version 25.2
[3040336.530] (II) UnloadModule: "vesa"
[3040336.530] (II) Unloading vesa
[3040336.530] (II) Failed to load module "vesa" (already loaded, 0)
[3040336.530] (II) modesetting: Driver for Modesetting Kernel Drivers: kms
[3040336.530] (II) FBDEV: driver for framebuffer: fbdev
[3040336.530] (II) VESA: driver for VESA chipsets: vesa
[3040336.530] xf86EnableIO: failed to enable I/O ports 0000-03ff (Operation not permitted)
[3040336.530] (EE) open /dev/dri/card0: No such file or directory
[3040336.530] (WW) Falling back to old probe method for modesetting
[3040336.530] (EE) open /dev/dri/card0: No such file or directory
[3040336.530] (II) Loading sub module "fbdevhw"
[3040336.530] (II) LoadModule: "fbdevhw"
[3040336.530] (II) Loading /usr/lib/xorg/modules/libfbdevhw.so
[3040336.530] (II) Module fbdevhw: vendor="X.Org Foundation"
[3040336.530]   compiled for 1.21.1.4, module version = 0.0.2
[3040336.530]   ABI class: X.Org Video Driver, version 25.2
[3040336.530] (EE) Unable to find a valid framebuffer device
[3040336.530] (WW) Falling back to old probe method for fbdev
[3040336.530] (II) Loading sub module "fbdevhw"
[3040336.530] (II) LoadModule: "fbdevhw"
[3040336.530] (II) Loading /usr/lib/xorg/modules/libfbdevhw.so
[3040336.530] (II) Module fbdevhw: vendor="X.Org Foundation"
[3040336.530]   compiled for 1.21.1.4, module version = 0.0.2
[3040336.530]   ABI class: X.Org Video Driver, version 25.2
[3040336.530] (EE) open /dev/fb0: No such file or directory
[3040336.530] vesa: Ignoring device with a bound kernel driver
[3040336.530] (WW) VGA arbiter: cannot open kernel arbiter, no multi-card support
[3040336.530] (EE) Screen 0 deleted because of no matching config section.
[3040336.530] (II) UnloadModule: "modesetting"
[3040336.530] (EE) Screen 0 deleted because of no matching config section.
[3040336.530] (II) UnloadModule: "fbdev"
[3040336.530] (II) UnloadSubModule: "fbdevhw"
[3040336.530] (EE) Screen 0 deleted because of no matching config section.
[3040336.530] (II) UnloadModule: "vesa"
[3040336.530] (EE) Device(s) detected, but none match those in the config file.
[3040336.530] (EE)
Fatal server error:
[3040336.530] (EE) no screens found(EE)
[3040336.530] (EE)
Please consult the The X.Org Foundation support
         at http://wiki.x.org
 for help.
[3040336.530] (EE) Please also check the log file at "/var/log/Xorg.1.log" for additional information.
[3040336.530] (EE)
[3040336.531] (EE) Server terminated with error (1). Closing log file.
```

> 从日志中可以看出，Xorg服务器在启动时遇到了多个问题，导致无法找到可用的屏幕。以下是一些关键点和可能的解决方案：
>
> 1. **缺少 `/dev/dri/card0` 文件**：
>    ```
>    (EE) open /dev/dri/card0: No such file or directory
>    ```
>    这表明Xorg无法找到显卡设备。你需要确保显卡驱动正确安装并且设备文件存在。
>
> 2. **无法启用I/O端口**：
>    ```
>    xf86EnableIO: failed to enable I/O ports 0000-03ff (Operation not permitted)
>    ```
>    这可能是由于权限问题或内核配置问题。确保你有权限访问这些I/O端口。
>
> 3. **无法找到有效的帧缓冲设备**：
>    ```
>    (EE) Unable to find a valid framebuffer device
>    ```
>    这表明Xorg无法找到可用的帧缓冲设备。你需要确保系统中有可用的帧缓冲设备，并且它们被正确配置。
>
> 4. **VGA arbiter无法打开**：
>    ```
>    (WW) VGA arbiter: cannot open kernel arbiter, no multi-card support
>    ```
>    这表明系统无法启用VGA arbiter，可能是由于内核配置或权限问题。
>
> 5. **没有匹配的配置部分**：
>    ```
>    (EE) Screen 0 deleted because of no matching config section.
>    ```
>    这表明Xorg无法找到与硬件匹配的配置部分。你需要确保Xorg配置文件正确并且包含适当的配置部分。
>
> 以下是一些可能的解决方案：
>
> 1. **安装显卡驱动**：
>    确保你的系统已经安装了正确的显卡驱动。你可以使用以下命令来安装开源的Nouveau驱动（适用于NVIDIA显卡）：
>    
>    ```bash
>    sudo apt update
>    sudo apt install xserver-xorg-video-nouveau
>    ```
>    
>    如果你使用的是AMD显卡，可以安装开源的AMD驱动：
>    ```bash
>    sudo apt update
>    sudo apt install xserver-xorg-video-amdgpu
>    ```
>    
> 2. **检查Xorg配置文件**：
>    确保Xorg的配置文件位于正确的位置并且包含正确的配置。Xorg的配置文件通常位于`/etc/X11/xorg.conf`或`/usr/share/X11/xorg.conf.d/`目录下。你可以检查以下文件：
>    - `/etc/X11/xorg.conf`
>    - `/usr/share/X11/xorg.conf.d/10-nvidia.conf`（适用于NVIDIA显卡）
>    - `/usr/share/X11/xorg.conf.d/10-amdgpu.conf`（适用于AMD显卡）
>
> 3. **创建自定义Xorg配置文件**：
>    如果默认配置文件不存在或不正确，你可以尝试创建一个自定义的Xorg配置文件。你可以使用以下命令来生成一个默认的配置文件：
>    ```bash
>    sudo Xorg -configure
>    ```
>
>    这将生成一个名为`xorg.conf.new`的文件。你可以将其复制到`/etc/X11/xorg.conf`：
>    ```bash
>    sudo cp xorg.conf.new /etc/X11/xorg.conf
>    ```
>
> 4. **检查权限**：
>    确保你有权限访问显卡设备。你可以使用以下命令来检查权限：
>    ```bash
>    ls -l /dev/dri/card0
>    ```
>
>    如果权限不正确，你可以使用以下命令来更改权限：
>    ```bash
>    sudo chmod 666 /dev/dri/card0
>    ```
>
> 5. **查看日志**：
>    查看Xorg日志文件以获取更多关于启动失败的信息。你可以使用以下命令来查看日志：
>    
>    ```bash
>    cat /var/log/Xorg.1.log
>    ```
>
> 如果以上步骤都无法解决问题，你可能需要进一步检查系统的配置和日志，或者寻求更详细的帮助。希望这些建议能帮助你解决问题。如果有任何问题，请随时告诉我！

### 检查权限和文件描述符：

确保当前用户有足够的权限运行Cinnamon桌面环境，并且文件描述符没有被不正确地锁定。你可以尝试以root用户身份运行Cinnamon会话：

```
sudo DISPLAY=:1 cinnamon-session
```

---

## 安装 Gnome

```bash
sudo apt install vanilla-gnome-desktop vanilla-gnome-default-settings
```

验证 VNC：

```bash
>>> import cv2
>>> import numpy as np
>>> h = 500
>>> w = 500
>>> img = 255 * np.ones((h ,w , 3), dtype=np.uint8)
>>> cv2.imshow("", img)
qt.qpa.xcb: could not connect to display
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/root/miniconda3/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb.

Aborted (core dumped)
root@autodl-container-3b9d4a8a31-968453db:~# cv2.waitKey(0)
-bash: syntax error near unexpected token `0'
```

您遇到的错误信息表明 OpenCV 的 `imshow` 函数试图使用 Qt 后端来显示图像，但由于无法连接到 X 服务器（Linux 用于图形界面的显示服务器），因此失败了。这很可能是因为您运行脚本的环境没有设置 X 服务器，或者 `DISPLAY` 环境变量没有正确设置。

要检查当前会话是使用 Xorg 还是其他显示服务器（如 Wayland），您可以通过以下几种方法来确定：

1. **检查 `echo $XDG_SESSION_TYPE` 环境变量：**
   `XDG_SESSION_TYPE` 环境变量会告诉你当前会话使用的显示服务器类型。

   ```bash
   echo $XDG_SESSION_TYPE
   ```

   如果输出是 `x11`，则表示使用的是 Xorg；如果是 `wayland`，则表示使用的是 Wayland。

2. **检查 `loginctl` 命令输出：**
   `loginctl` 命令可以显示当前登录会话的详细信息，包括使用的显示服务器类型。

   ```bash
   loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p Type
   ```

   这将输出类似 `Type=x11` 或 `Type=wayland` 的信息。

3. **检查 `ps` 命令输出：**
   您可以通过 `ps` 命令来查找正在运行的显示服务器进程。

   ```bash
   ps -e | grep -E 'Xorg|Xwayland'
   ```

   如果输出包含 `Xorg`，则表示使用的是 Xorg；如果输出包含 `Xwayland`，则表示使用的是 Wayland。

4. **检查桌面环境的配置文件：**
   某些桌面环境（如 GNOME）的配置文件可能会指明它们正在使用的显示服务器。例如，在 GNOME 中，您可以检查 `gsettings` 设置：

   ```bash
   gsettings get org.gnome.desktop.session session-name
   ```

   这将返回当前会话的名称，可能包含 `x11` 或 `wayland`。

5. **检查 `/var/log/Xorg.0.log` 文件：**
   如果 Xorg 正在运行，您可以在 `/var/log/` 目录下找到 Xorg 的日志文件。检查这些日志文件可以确认 Xorg 是否正在使用。

   ```bash
   ls /var/log/ | grep Xorg
   ```

通过这些方法，您可以确定当前会话是使用 Xorg 还是其他显示服务器，如 Wayland。这对于配置图形应用程序和调试显示问题非常有用。

```bash
root@autodl-container-3b9d4a8a31-968453db:~# echo $XDG_SESSION_TYPE

root@autodl-container-3b9d4a8a31-968453db:~# loginctl show-session $(loginctl | grep $(whoami) | awk '{print $1}') -p Type
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to connect to bus: Host is down
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to connect to bus: Host is down
root@autodl-container-3b9d4a8a31-968453db:~# gsettings get org.gnome.desktop.session session-name
'gnome'
root@autodl-container-3b9d4a8a31-968453db:~# gsettings get org.gnome.desktop.session session-name
'gnome'
```

前文已知这个服务器基于SysVinit而不是systemd，所以上述方法无效。

#### 检查是否安装会话服务器

```bash
root@autodl-container-3b9d4a8a31-968453db:~# dpkg -l | grep -E 'xorg|wayland'
ii  libwayland-client0:amd64                   1.20.0-1ubuntu0.1                       amd64        wayland compositor infrastructure - client library
ii  libwayland-cursor0:amd64                   1.20.0-1ubuntu0.1                       amd64        wayland compositor infrastructure - cursor library
ii  libwayland-egl1:amd64                      1.20.0-1ubuntu0.1                       amd64        wayland compositor infrastructure - EGL library
ii  libwayland-server0:amd64                   1.20.0-1ubuntu0.1                       amd64        wayland compositor infrastructure - server library
ii  python3-xkit                               0.5.0ubuntu5                            all          library for the manipulation of xorg.conf files (Python 3)
ii  xorg                                       1:7.7+23ubuntu2                         amd64        X.Org X Window System
ii  xorg-docs-core                             1:1.7.1-1.2                             all          Core documentation for the X.org X Window System
ii  xorg-sgml-doctools                         1:1.11-1.1                              all          Common tools for building X.Org SGML documentation
ii  xserver-xorg                               1:7.7+23ubuntu2                         amd64        X.Org X server
ii  xserver-xorg-core                          2:21.1.4-2ubuntu1.7~22.04.11            amd64        Xorg X server - core server
ii  xserver-xorg-input-all                     1:7.7+23ubuntu2                         amd64        X.Org X server -- input driver metapackage
ii  xserver-xorg-input-libinput                1.2.1-1                                 amd64        X.Org X server -- libinput input driver
ii  xserver-xorg-input-wacom                   1:1.0.0-3ubuntu1                        amd64        X.Org X server -- Wacom input driver
ii  xserver-xorg-legacy                        2:21.1.4-2ubuntu1.7~22.04.11            amd64        setuid root Xorg server wrapper
ii  xserver-xorg-video-all                     1:7.7+23ubuntu2                         amd64        X.Org X server -- output driver metapackage
ii  xserver-xorg-video-amdgpu                  22.0.0-1ubuntu0.2                       amd64        X.Org X server -- AMDGPU display driver
ii  xserver-xorg-video-ati                     1:19.1.0-2ubuntu1                       amd64        X.Org X server -- AMD/ATI display driver wrapper
ii  xserver-xorg-video-fbdev                   1:0.5.0-2build1                         amd64        X.Org X server -- fbdev display driver
ii  xserver-xorg-video-intel                   2:2.99.917+git20210115-1                amd64        X.Org X server -- Intel i8xx, i9xx display driver
ii  xserver-xorg-video-nouveau                 1:1.0.17-2build1                        amd64        X.Org X server -- Nouveau display driver
ii  xserver-xorg-video-qxl                     0.1.5+git20200331-3                     amd64        X.Org X server -- QXL display driver
ii  xserver-xorg-video-radeon                  1:19.1.0-2ubuntu1                       amd64        X.Org X server -- AMD/ATI Radeon display driver
ii  xserver-xorg-video-vesa                    1:2.5.0-1build4                         amd64        X.Org X server -- VESA display driver
ii  xserver-xorg-video-vmware                  1:13.3.0-3build1                        amd64        X.Org X server -- VMware display driver
ii  xwayland                                   2:22.1.1-1ubuntu0.13                    amd64        X server for running X clients under Wayland
```

运行 Xorg

```bash
root@autodl-container-3b9d4a8a31-968453db:~# startx

X.Org X Server 1.21.1.4
X Protocol Version 11, Revision 0
Current Operating System: Linux autodl-container-3b9d4a8a31-968453db 5.15.0-57-generic #63-Ubuntu SMP Thu Nov 24 13:43:17 UTC 2022 x86_64
Kernel command line: BOOT_IMAGE=/boot/vmlinuz-5.15.0-57-generic root=UUID=8e57dfb2-fbde-4670-bb0f-2809e09d5423 ro crashkernel=512M-:384M net.ifnames=0 biosdevname=0 clocksource=tsc tsc=reliable console=tty0 console=ttyS0,115200
xorg-server 2:21.1.4-2ubuntu1.7~22.04.11 (For technical support please see http://www.ubuntu.com/support)
Current version of pixman: 0.40.0
        Before reporting problems, check http://wiki.x.org
        to make sure that you have the latest version.
Markers: (--) probed, (**) from config file, (==) default setting,
        (++) from command line, (!!) notice, (II) informational,
        (WW) warning, (EE) error, (NI) not implemented, (??) unknown.
(==) Log file: "/var/log/Xorg.0.log", Time: Fri Aug 30 00:00:31 2024
(==) Using system config directory "/usr/share/X11/xorg.conf.d"
xf86EnableIO: failed to enable I/O ports 0000-03ff (Operation not permitted)
vesa: Ignoring device with a bound kernel driver
xf86EnableIO: failed to enable I/O ports 0000-03ff (Operation not permitted)
vesa: Ignoring device with a bound kernel driver
(EE)
Fatal server error:
(EE) no screens found(EE)
(EE)
Please consult the The X.Org Foundation support
         at http://wiki.x.org
 for help.
(EE) Please also check the log file at "/var/log/Xorg.0.log" for additional information.
(EE)
(EE) Server terminated with error (1). Closing log file.
xinit: giving up
```

#### VNC 配置

默认情况下，VNC 服务器可能不会启动 GNOME 会话。您需要编辑 VNC 的启动脚本以启动 GNOME。

```
nano ~/.vnc/xstartup.turbovnc
```

将内容修改为：

```
#!/bin/bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
exec gnome-session &
```

然后给脚本执行权限：

```
chmod +x ~/.vnc/xstartup
```

### 重启 VNC

1. 停止当前的VNC服务器：

   ```
   /opt/TurboVNC/bin/vncserver -kill :1
   ```

2. 删除旧的密码文件：

   ```
   rm /root/.vnc/passwd
   ```

3. 重新启动VNC服务器，系统会再次提示你设置新的VNC密码：

   ```
   USER=root /opt/TurboVNC/bin/vncserver :1 -desktop X -auth /root/.Xauthority -geometry 1920x1080 -depth 24 -rfbwait 120000 -rfbauth /root/.vnc/passwd -fp /usr/share/fonts/X11/misc/,/usr/share/fonts -rfbport 6006
   ```

---

1.安装xfce和[vncserver](https://so.csdn.net/so/search?q=vncserver&spm=1001.2101.3001.7020)

```bash
apt-get update
apt-get install xfce4 xfce4-goodies
apt-get install tightvncserver
```

2.启动vncserver

```bash
vncserver
```

初次启动需要设置密码，需要二次确认，然后会跳出一个选项，选n即可

3.配置.[vnc](https://so.csdn.net/so/search?q=vnc&spm=1001.2101.3001.7020)/xstartup

将.vnc/xstartup中的内容替换为：

```bash
#!/bin/sh
# Uncomment the following two lines for normal desktop:
# unset SESSION_MANAGER
# exec /etc/X11/xinit/xinitrc
 
[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
xsetroot -solid grey
vncconfig -iconic &
x-terminal-emulator -geometry 80x24+10+10 -ls -title "$VNCDESKTOP Desktop" &
x-window-manager &
 
#gnome-terminal &
 
sesion-manager & xfdesktop & xfce4-panel &
xfce4-menu-plugin &
xfsettingsd &
xfconfd &
xfwm4 &
```

4.重新启动vncserver，并指定屏幕分辨率

```bash
vncserver -kill :1
vncserver -geometry 1920x1080
```

5. 如果遇到下面类似的报错

```bash
(base) root@autodl-container-ae0c4bafdd-217fb69d:~# vncserver -geometry 1920x1080

Warning: autodl-container-ae0c4bafdd-217fb69d:1 is taken because of /tmp/.X11-unix/X1
Remove this file if there is no X server autodl-container-ae0c4bafdd-217fb69d:1

Warning: autodl-container-ae0c4bafdd-217fb69d:2 is taken because of /tmp/.X11-unix/X2
Remove this file if there is no X server autodl-container-ae0c4bafdd-217fb69d:2

New 'X' desktop is autodl-container-ae0c4bafdd-217fb69d:4

Starting applications specified in /root/.vnc/xstartup
Log file is /root/.vnc/autodl-container-ae0c4bafdd-217fb69d:4.log
```

可以这样解决

```bash
vncserver -kill :1
rm /tmp/.X11-unix/X1 /tmp/.X11-unix/X2
rm ~/.vnc/passwd
vncserver -geometry 1920x1080
```

---

### 1.安装KDE桌面

```
apt update
apt upgrade
apt install task-kde-desktop
```

#### 1.1 设置界面语言和编码

```
localectl set-locale LANG=en_US.UTF-8
```

### 2.安装VNC服务器

```
apt install tigervnc-standalone-server tigervnc-common -y
```

#### 2.1 如果需要，添加用户

```
adduser username
su - username
vncpasswd
exit
```

#### 2.2 我们将以root身份连接，因此我们将设置一个密码

```
vncpasswd

Would you like to enter a view-only password (y/n)?, 答案： n (否).
```

### 3. 创建图形 shell 的配置文件

```
vim ~/.vnc/xstartup

#!/bin/sh
unset SESSION_MANAGER
exec /etc/X11/xinit/xinitrc
[ -x /etc/vnc/xstartup ] && exec /etc/vnc/xstartup
[ -r $HOME/.Xresources ] && xrdb $HOME/.Xresources
xsetroot -solid grey
vncconfig -iconic &
/usr/bin/startkde &
```

#### 3.1 复制文件

```
cp /etc/X11/Xresources/x11-common ~/.Xresources
```

#### 3.2 重启服务器

```
reboot
```

---



## 参考资料

[AutoDL帮助文档](https://www.autodl.com/docs/gui/)

[AutoDL帮助文档](https://www.autodl.com/docs/ssh_proxy/)

[VNC连接服务器实现远程桌面 --以AutoDL云服务器为例_autodl vnc-CSDN博客](https://blog.csdn.net/qq_44114055/article/details/134299963)

