## **第一次训练(8:2划分数据集)**

​	得到的F1,P,R,PR曲线

​						<img src=".\asset\trainB\第一次训练\F1_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainB\第一次训练\P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainB\第一次训练\R_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainB\第一次训练\PR_curve.png" alt="P_curve" style="zoom:25%;" />

​	对结果进行分析，发现motor最不理想，可能是因为图像太暗，导致识别不准确。

## **第二次训练(SCI夜间增强技术处理照片)**

	class EnhanceNetwork(nn.Module):
	    def __init__(self, layers, channels):
	        # 调用父类的构造函数，完成初始化
	        super(EnhanceNetwork, self).__init__()
	        # 卷积核大小为3*3、膨胀设为1、通过计算，得到合适的填充大小。
	        kernel_size = 3
	        dilation = 1
	        padding = int((kernel_size - 1) / 2) * dilation
	        # 输入卷积层
	        self.in_conv = nn.Sequential(
	            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
	            nn.ReLU()
	        )
	        # 中间卷积层
	        self.conv = nn.Sequential(
	            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
	            nn.BatchNorm2d(channels),
	            nn.ReLU()
	        )
	        # 模块列表
	        self.blocks = nn.ModuleList()
	        for i in range(layers):
	            self.blocks.append(self.conv)
	        # 输出卷积层
	        self.out_conv = nn.Sequential(
	            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
	            nn.Sigmoid()
	        )


​	 
​	    def forward(self, input):
​	        # 将输入数据通过输入卷积层 self.in_conv，得到中间特征 fea。
​	        fea = self.in_conv(input)
​	        # 通过多个中间卷积层 self.blocks 进行迭代，
​	        for conv in self.blocks:
​	            # 每次迭代中将当前特征 fea 与中间卷积层的输出相加。
​	            fea = fea + conv(fea)
​	        fea = self.out_conv(fea)
​	        # 将输出特征与输入数据相加，得到增强后的图像。
​	        illu = fea + input
​	        # 通过 torch.clamp 函数将图像的像素值限制在 0.0001 到 1 之间，以确保输出在有效范围内。
​	        illu = torch.clamp(illu, 0.0001, 1)
​	 
	        return illu

​	通过多次堆叠卷积块，来学习图像的特征。增强后的图像通过与输入相加并进行截断，以确保像素值在合理范围内。

​	首先，通过__init__初始化超参数和网络层，参数设置看注释。

​	然后，将输入图像通过3*3的卷积层，得到特征 fea，对特征 fea 多次应用相同的卷积块进行叠加，通过输出卷积层获得最终的特征 	fea。

​	接着，将生成的特征与输入图像相加，得到增强后的图像，通过 clamp 函数将图像像素值限制在 0.0001 和 1 之间。

​	最后，返回增强后的图像 illu。

```
class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers
 
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
 
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)
 
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
 
    def forward(self, input):
        # 将输入数据通过输入卷积层，得到中间特征 fea。
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)
 
        fea = self.out_conv(fea)
        # 计算输入与输出的差异，得到增益调整的值。
        delta = input - fea
 
        return delta
```

​	这段代码主要作用是定义了一个校准网络

​        与上一段代码的区别是在前向传播时，输入经过一系列卷积操作后，对于最终的特征 fea再与原始输入相减，得到最终的增益调整结	果delta。

```
class Network(nn.Module):
 
    def __init__(self, stage=3):
        super(Network, self).__init__()
        # 将传入的 stage 参数保存到类的实例变量中
        self.stage = stage
        # 增强和校准
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self.calibrate = CalibrateNetwork(layers=3, channels=16)
        # 创建了一个损失函数实例，用于计算损失
        self._criterion = LossFunction()
 
    # 权重使用正态分布初始化
    def weights_init(self, m):
        
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
 
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)
 
    def forward(self, input):
        # 初始化空列表 ilist, rlist, inlist, attlist 用于存储不同阶段的结果
        ilist, rlist, inlist, attlist = [], [], [], []
        # 将输入数据保存到 input_op 变量中。
        input_op = input
        for i in range(self.stage):
            # 将当前输入添加到输入列表中
            inlist.append(input_op) 
            # 使用图像增强网络 self.enhance 处理当前输入，得到增强后的图像
            i = self.enhance(input_op)
            # 计算增强前后的比例
            r = input / i
            # 将比例值限制在 [0, 1] 范围内
            r = torch.clamp(r, 0, 1)
            # 使用校准网络 self.calibrate 对比例进行校准，得到校准值
            att = self.calibrate(r)
            # 将原始输入与校准值相加，得到下一阶段的输入
            input_op = input + att
            # 分别将当前阶段的增强图像、比例、输入和校准值的绝对值添加到对应的列表中
            ilist.append(i)
            rlist.append(r)
            attlist.append(torch.abs(att))
 
        return ilist, rlist, inlist, attlist
 
    def _loss(self, input):
        i_list, en_list, in_list, _ = self(input)
        loss = 0
        for i in range(self.stage):
            loss += self._criterion(in_list[i], i_list[i])
        return loss
```

​	这段代码主要作用是组合了刚刚我们提到的图像增强网络 (EnhanceNetwork) 和校准网络 (CalibrateNetwork)，并多次执行这两个操	作。

​	1.首先，初始化网络结构，并创建EnhanceNetwork、CalibrateNetwork以及loss损失函数的实例。

​	2.接着，定义权重初始化的方法。

​	3.然后，通过多次迭代，每次迭代中进行以下步骤：

​	4.将当前输入保存到列表中。
​	5.使用图像增强网络 EnhanceNetwork 处理当前输入，得到增强后的图像。
​	6.计算增强前后的比例，并将比例值限制在 [0, 1] 范围内。
​	7.使用校准网络 CalibrateNetwork 对比例进行校准，得到校准值。
​	8.将原始输入与校准值相加，得到下一阶段的输入。
​	9.将当前阶段的增强图像、比例、输入和校准值的绝对值保存到对应的列表中。
​	10.返回四个列表，分别包含不同阶段的增强图像、比例、输入和校准值。
​	11.最后，计算损失。

​	下面是两张对比图

​	<img src=".\asset\trainB\第二次训练\train_B_1535.jpg" alt="P_curve" style="zoom:25%;" /><img src=".\asset\trainB\第二次训练\train_B_1535.png" alt="P_curve" style="zoom:25%;" />  

​	可以看到有了较为明显的改进。

​	得到的F1,P,R,PR

​						<img src=".\asset\trainB\第二次训练\F1_curve.png" alt="P_curve" style="zoom:25%;" />				

<img src=".\asset\trainB\第二次训练\P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainB\第二次训练\R_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainB\第二次训练\PR_curve.png" alt="P_curve" style="zoom:25%;" />

​	可以看到，F1的的值提高了5个点，提升效果不好可能是因为图片过于曝光。

​	参考文献：

```
@inproceedings{ma2022toward,
  title={Toward Fast, Flexible, and Robust Low-Light Image Enhancement},
  author={Ma, Long and Ma, Tengyu and Liu, Risheng and Fan, Xin and Luo, Zhongxuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5637--5646},
  year={2022}
}
```

