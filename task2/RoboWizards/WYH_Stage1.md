### Stage1

### 构建推理引擎

1.**安装Anaconda**
本人电脑已安装旧版本Anaconda，也可去官网https://www.anaconda.com/download下载最新版，打开安装包一直下一步即可
2.**创建虚拟环境**
打开Anaconda Prompt ,
输入命令行conda create -n wtransformer python=3.9 -y创建虚拟环境
conda activate wtransformer切换虚拟环境
3.**安装pytorch**
可参考pytorch官网https://pytorch.org，根据自身电脑情况安装pytorch，
本人用conda安装pytorch，输入命令行pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
，也可用pip等方式安装，
4.**安装transformers库**
pip install transformers
也可用清华源pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple


**电脑配置：**
GPU：NVIDIA GeForce RTX 3060 Laptop GPU
GPU计算能力：8.6
GPU内存：5.99951171875GB
多处理器数量：30个
CPU：12th Gen Intel(R) Core(TM) i9-12900H
CPU核心数：20个

**吞吐量**
本人运用 bert-base-uncased 模型，对 Transformer 吞吐量进行测试，吞吐量一般 130~170 samples/second 左右，优化还在测试中。