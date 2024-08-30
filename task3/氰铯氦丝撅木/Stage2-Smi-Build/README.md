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