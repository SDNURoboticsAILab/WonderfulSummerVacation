## **第一次测试(8:2划分数据集)**

​	按照视频教程将train:val=8:2进行数据集的划分，然后利用yolov8n.pt进行train

​	得到的F1，P，R，PR曲线分别为：

<img src=".\asset\trainA\第一次训练\F1_curve.png" alt="A pic" style="zoom:25%;" />

<img src=".\asset\trainA\第一次训练\P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第一次训练\R_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第一次训练\PR_curve.png" alt="P_curve" style="zoom:25%;" />

​	从F1曲线种看出训练效果不佳

## **第二次尝试(9:1)划分数据集**

​	重新划分数据集，按照train:val=9:1划分

​	得到的F1，P，R，PR曲线分别为：

<img src=".\asset\trainA\第二次训练\F1_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第二次训练\P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第二次训练\R_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第二次训练\PR_curve.png" alt="P_curve" style="zoom:25%;" />

​	发现数据集划分为9:1不如8:2效果好

## **第三次尝试(soft_nms)**

​	改进soft_nms非极大值抑制算法

​	nms流程如下：

​		假设有6个矩形框，根据分类器的类别分类概率做排序，假设从小到大属于人物的概率 分别为A、B、C、D、E、F。
​		(1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;
​		(2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。
​		(3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E		是我们保留下来的第二个矩形框。
​		一直重复，找到所有被保留下来的矩形框。

​	但是nms存在一些问题：

​		NMS在密集遮挡场景会出现很大的问题，漏检很多，例如下图，红色框和绿色框是当前的检测结果，二者的得分分别是0.95和	0.80。如果按照传统的NMS进行处理，首先选中得分最高的红色框，然后绿色框就会因为与之重叠面积过大而被删掉。另外，NMS	的阈值也不太容易确定，设小了会出现下图的情况（绿色框因为和红色框重叠面积较大而被删掉），设置过高又容易增大误检。

​	soft_nms对nms的改进

​		在该算法中，我们基于重叠部分的大小为相邻检测框设置一个衰减函数而非彻底将其分数置为零。而是不要粗鲁地删除所有IOU          	大于阈值的框，而是降低其置信度。M为当前得分最高框，bi 为待处理框，bi 和M的IOU越大，bi 的得分si 就下降的越厉害。

​	改进实现，在修改ultralytics/yolo/utils/ops.py中添加以下代码：

```
def box_iou_for_nms(box1, box2, GIoU=False, DIoU=False, CIoU=False, SIoU=False, EIou=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)
 
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
    w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)
 
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)
 
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
 
    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or EIou:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIou:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIou:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    elif SIoU:
        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        return iou - 0.5 * (distance_cost + shape_cost)
    return iou  # IoU
 
 
def soft_nms(bboxes, scores, iou_thresh=0.5, sigma=0.5, score_threshold=0.25):
    order = torch.arange(0, scores.size(0)).to(bboxes.device)
    keep = []
 
    while order.numel() > 1:
        if order.numel() == 1:
            keep.append(order[0])
            break
        else:
            i = order[0]
            keep.append(i)
 
        iou = box_iou_for_nms(bboxes[i], bboxes[order[1:]]).squeeze()
 
        idx = (iou > iou_thresh).nonzero().squeeze()
        if idx.numel() > 0:
            iou = iou[idx]
            newScores = torch.exp(-torch.pow(iou, 2) / sigma)
            scores[order[idx + 1]] *= newScores
 
        newOrder = (scores[order[1:]] > score_threshold).nonzero().squeeze()
        if newOrder.numel() == 0:
            break
        else:
            maxScoreIndex = torch.argmax(scores[order[newOrder + 1]])
            if maxScoreIndex != 0:
                newOrder[[0, maxScoreIndex],] = newOrder[[maxScoreIndex, 0],]
            order = order[newOrder + 1]
 
    return torch.LongTensor(keep)
```

​	再修改def non_max_suppression函数：

```
i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
#改为
i = soft_nms(boxes, scores, iou_thres)  # NMS
```

```
iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#改为
iou =box_iou_for_nms(boxes[i], boxes, GIoU=True, DIoU=False, CIoU=False, SIoU=False, EIou=False, eps=1e-7) > iou_thres  # iou matrix
```

​	参考文献：

​	Yi-Fan Zhang, Weiqiang Ren, Zhang Zhang, Zhen Jia, Liang Wang, ‘Tieniu Tan,Focal and efficient IOU loss for accurate bounding box regression’,Neurocomputing,Volume 506,2022

​	Zhora Gevorgyan，‘ SIoU Loss: More Powerful Learning for Bounding Box Regression ’

​	得到的F1,P,R,PR曲线如图所示

​						<img src=".\asset\trainA\第三次训练\F1_curve.png" alt="P_curve" style="zoom:25%;" />

​                

<img src=".\asset\trainA\第三次训练\PR_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第三次训练\P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第三次训练\R_curve.png" alt="P_curve" style="zoom:25%;" />

​	结果分析，F1曲线效果并不好，意味着此改进并不能对结果进行优化。

## **第四次训练(Bifpn网络结构)**

​	使用Bifpn进行改进

​	**BiFPN（Bi-directional Feature Pyramid Network）是一种用于目标检测和语义分割任务的神经网络架构，旨在改善特征金字塔网络（Feature Pyramid Network, FPN）的性能。FPN是一种用于处理多尺度信息的网络结构，通常与骨干网络结合使用，以生成不同分辨率的特征金字塔，从而提高对象检测和分割的性能。BiFPN在此基础上进行了改进，以更好地捕获多尺度信息和提高模型性能。**

​	以下是BiFPN的关键特点和工作原理：

​	**双向连接： BiFPN引入了双向连接，允许信息在不同分辨率级别之间双向传播。这有助于更好地融合低级别和高级别特征，并促进了特征的上下文传播，从而提高了对象检测和分割的准确性。**

​	**自适应特征调整： BiFPN采用自适应的特征调整机制，可以学习权重，以调整不同层级的特征以更好地匹配不同任务的需求。这有助于改进特征融合的效果。**

​	**模块化设计： BiFPN的模块化设计使其易于嵌入到各种深度神经网络架构中，例如单发射点（Single Shot MultiBox Detector）、	YOLO（You Only Look Once）、以及Mask R-CNN等。**

​	**高效性： BiFPN被设计为高效的模型，具有较少的参数和计算复杂度，使其适用于嵌入式设备和实际部署。**

​	**提高性能： BiFPN的引入通常能够显著提高对象检测和分割任务的性能，特别是对于小目标或复杂场景，其性能改进尤为显著。**

​	**总的来说，BiFPN是一种改进的特征金字塔网络结构，通过双向连接、自适应特征调整和模块化设计，提高了对象检测和语义分割任	务的性能，使得神经网络能够更好地理解和解释多尺度信息，从而在计算机视觉任务中发挥更大的作用。**

​	以下是BiFPN的代码：

```

# BiFPN
# 两个特征图add操作
import torch.nn as nn
import torch

class BiFPN_Add2(nn.Module):
	def __init__(self, c1, c2):
		super(BiFPN_Add2, self).__init__()
		# 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
		# 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
		# 从而在参数优化的时候可以自动一起优化
		self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
		self.epsilon = 0.0001
		self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
		self.silu = nn.SiLU()

	def forward(self, x):
		x0, x1 = x
		w = self.w
		weight = w / (torch.sum(w, dim=0) + self.epsilon)
		return self.conv(self.silu(weight[0] * x0 + weight[1] * x1))


# 三个特征图add操作
class BiFPN_Add3(nn.Module):
	def __init__(self, c1, c2):
		super(BiFPN_Add3, self).__init__()
		self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
		self.epsilon = 0.0001
		self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
		self.silu = nn.SiLU()

	def forward(self, x):
		w = self.w
		weight = w / (torch.sum(w, dim=0) + self.epsilon)
		# Fast normalized fusion
		return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
```

​	在修改yaml文件即可进行训练

​	得到F1,P,R,PR曲线

​						<img src=".\asset\trainA\第四次训练\F1_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第四次训练\P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第四次训练\R_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第四次训练\PR_curve.png" alt="P_curve" style="zoom:25%;" />

​	从F1曲线中发现，虽然第四次训练的效果最好，但是F1的指数依旧比较低，效果依旧不好。

​	参考文献

```
@article{li2019scale,
  title={Scale-Aware Trident Networks for Object Detection},
  author={Li, Yanghao and Chen, Yuntao and Wang, Naiyan and Zhang, Zhaoxiang},
  journal={ICCV 2019},
  year={2019}
}
```



**第五次训练(剔除质量不好的照片)**

​	质量不好的照片通常指那些由于各种原因导致图像无法为模型提供有效信息的照片。以下是一些常见的质量问题及其剔除方法：

**质量不好的照片的特征**

1. **模糊**：由于运动、对焦不准等原因导致图像模糊，细节不清。
2. **曝光不良**：
   - **过曝**：图像过亮，细节丢失。
   - **欠曝**：图像过暗，细节丢失。
3. **噪声过多**：图像中有大量噪声，干扰了图像的清晰度。
4. **分辨率过低**：图像分辨率太低，无法提供足够的细节。
5. **遮挡**：目标物体被部分或完全遮挡，无法有效标注。
6. **不相关内容**：图像中没有目标类别的对象，或图像内容与目标类别无关。
7. **不完整的标注**：标注文件缺失、错误或不完整。

**剔除质量不好的照片的方法**

1. **人工检查**：
   - 手动浏览数据集，删除明显质量不好的照片。这虽然费时，但精确度高。
   
2. **自动化工具和算法**：
   - **模糊检测**：
     - 使用图像处理库（如 OpenCV）计算图像的拉普拉斯变换（Laplacian）并计算其方差。方差低于某个阈值时，可以认为图像是模糊的。
     ```python
     import cv2
     import numpy as np
     
     def is_blurry(image_path, threshold=100):
         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
         laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
         return laplacian_var < threshold
     ```

   - **曝光检测**：
     - 使用直方图分析图像的亮度分布，检测过曝或欠曝的图像。
     ```python
     def is_overexposed(image_path, threshold=200):
         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
         hist = cv2.calcHist([image], [0], None, [256], [0, 256])
         return np.argmax(hist) > threshold
     
     def is_underexposed(image_path, threshold=50):
         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
         hist = cv2.calcHist([image], [0], None, [256], [0, 256])
         return np.argmax(hist) < threshold
     ```

   - **噪声检测**：
     - 使用滤波器（如中值滤波）检测图像中的噪声水平。
   
   - **分辨率检测**：
     - 检查图像的尺寸，剔除分辨率低于某个阈值的图像。
     ```python
     def is_low_resolution(image_path, min_width=640, min_height=480):
         image = cv2.imread(image_path)
         height, width = image.shape[:2]
         return width < min_width or height < min_height
     ```

   - **遮挡和不相关内容检测**：
     - 使用预训练的目标检测模型（如 YOLOv8）对图像进行初步检测，剔除无法检测到目标或检测结果不可靠的图像。

3. **标注检查**：
   - 自动化脚本检查标注文件的完整性和正确性，剔除没有对应标注或标注文件错误的图像。

​	通过这些方法，可以有效地剔除质量不好的照片，从而提高数据集的整体质量，进而提升 YOLOv8 模型的训练效果和精度。并且得	到了剔除后的照片，然后进行train。

​	得到的F1，P，R，PR曲线如图所示

<img src=".\asset\trainA\第五次训练\F1_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第五次训练\P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第五次训练\R_curve.png" alt="P_curve" style="zoom:25%;" />

<img src=".\asset\trainA\第五次训练\PR_curve.png" alt="P_curve" style="zoom:25%;" />

可以看到，在未修改的F1中，指数为0.233-0.53，而在剔除照片之后指数为0.273-0.49，所以说明保留照片的质量性较好。

接下来将剔除后的图片进行数据增强。

**第六次训练**

​	对上述剔除后的照片进行数据增强。

​	我采取的的数据增强的方法有：

- `resize_keep_ratio`：保持长宽比缩放图像。

- `resizeDown_keep_ratio`：只缩小图像尺寸，不放大。

- `resize`：将图像缩放到指定尺寸。

- `random_flip_horizon`：随机水平翻转图像。

- `random_flip_vertical`：随机垂直翻转图像。

- `center_crop`：中心裁剪图像。

- `random_bright`：随机调整图像亮度。

- `random_contrast`：随机调整图像对比度。

- `random_saturation`：随机调整图像饱和度。

- `add_gasuss_noise`：添加高斯噪声。

- `add_salt_noise`：添加盐噪声。

- `add_pepper_noise`：添加胡椒噪声。

  以下是完整代码

  ```
  # -*- coding: utf-8 -*-
  """
  Created on 2023-04-01 9:08
  @author: Fan yi ming
  Func: 对于目标检测的数据增强[YOLO]（特点是数据增强后标签也要更改）
  review：常用的数据增强方式；
          1.翻转：左右和上下翻转，随机翻转
          2.随机裁剪，图像缩放
          3.改变色调
          4.添加噪声
  注意： boxes的标签和坐标一个是int，一个是float，存放的时候要注意处理方式。
  参考：https://github.com/REN-HT/Data-Augmentation/blob/main/data_augmentation.py
  """
  import torch
  from PIL import Image
  from PIL import ImageDraw
  from PIL import ImageFile
  
  ImageFile.LOAD_TRUNCATED_IMAGES = True
  from torchvision import transforms
  import numpy as np
  import matplotlib.pyplot as plt
  import os
  import random
  
  from tqdm import tqdm
  from concurrent.futures import ThreadPoolExecutor
  
  NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
  
  
  def run(func, this_iter, desc="Processing"):
      with ThreadPoolExecutor(max_workers=NUM_THREADS, thread_name_prefix='MyThread') as executor:
          results = list(
              tqdm(executor.map(func, this_iter), total=len(this_iter), desc=desc)
          )
      return results
  
  random.seed(0)
  
  
  class DataAugmentationOnDetection:
      def __init__(self):
          super(DataAugmentationOnDetection, self).__init__()
  
      # 以下的几个参数类型中，image的类型全部如下类型
      # 参数类型： image：Image.open(path)
      def resize_keep_ratio(self, image, boxes, target_size):
          """
              参数类型： image：Image.open(path)， boxes:Tensor， target_size:int
              功能：将图像缩放到size尺寸，调整相应的boxes,同时保持长宽比（最长的边是target size
          """
          old_size = image.size[0:2]  # 原始图像大小
          # 取最小的缩放比例
          ratio = min(float(target_size) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
          new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
          # boxes 不用变化，因为是等比例变化
          return image.resize(new_size, Image.BILINEAR), boxes
  
      def resizeDown_keep_ratio(self, image, boxes, target_size):
          """ 与上面的函数功能类似，但它只降低图片的尺寸，不会扩大图片尺寸"""
          old_size = image.size[0:2]  # 原始图像大小
          # 取最小的缩放比例
          ratio = min(float(target_size) / (old_size[i]) for i in range(len(old_size)))  # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
          ratio = min(ratio, 1)
          new_size = tuple([int(i * ratio) for i in old_size])  # 根据上边求得的比例计算在保持比例前提下得到的图像大小
  
          # boxes 不用变化，因为是等比例变化
          return image.resize(new_size, Image.BILINEAR), boxes
  
      def resize(self, img, boxes, size):
          # ---------------------------------------------------------
          # 类型为 img=Image.open(path)，boxes:Tensor，size:int
          # 功能为：将图像长和宽缩放到指定值size，并且相应调整boxes
          # ---------------------------------------------------------
          return img.resize((size, size), Image.BILINEAR), boxes
  
      def random_flip_horizon(self, img, boxes, h_rate=1):
          # -------------------------------------
          # 随机水平翻转
          # -------------------------------------
          if np.random.random() < h_rate:
              transform = transforms.RandomHorizontalFlip(p=1)
              img = transform(img)
              if len(boxes) > 0:
                  x = 1 - boxes[:, 1]
                  boxes[:, 1] = x
          return img, boxes
  
      def random_flip_vertical(self, img, boxes, v_rate=1):
          # 随机垂直翻转
          if np.random.random() < v_rate:
              transform = transforms.RandomVerticalFlip(p=1)
              img = transform(img)
              if len(boxes) > 0:
                  y = 1 - boxes[:, 2]
                  boxes[:, 2] = y
          return img, boxes
  
      def center_crop(self, img, boxes, target_size=None):
          # -------------------------------------
          # 中心裁剪 ，裁剪成 (size, size) 的正方形, 仅限图形，w,h
          # 这里用比例是很难算的，转成x1,y1, x2, y2格式来计算
          # -------------------------------------
          w, h = img.size
          size = min(w, h)
          if len(boxes) > 0:
              # 转换到xyxy格式
              label = boxes[:, 0].reshape([-1, 1])
              x_, y_, w_, h_ = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
              x1 = (w * x_ - 0.5 * w * w_).reshape([-1, 1])
              y1 = (h * y_ - 0.5 * h * h_).reshape([-1, 1])
              x2 = (w * x_ + 0.5 * w * w_).reshape([-1, 1])
              y2 = (h * y_ + 0.5 * h * h_).reshape([-1, 1])
              boxes_xyxy = torch.cat([x1, y1, x2, y2], dim=1)
              # 边框转换
              if w > h:
                  boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] - (w - h) / 2
              else:
                  boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] - (h - w) / 2
              in_boundary = [i for i in range(boxes_xyxy.shape[0])]
              for i in range(boxes_xyxy.shape[0]):
                  # 判断x是否超出界限
                  if (boxes_xyxy[i, 0] < 0 and boxes_xyxy[i, 2] < 0) or (
                          boxes_xyxy[i, 0] > size and boxes_xyxy[i, 2] > size):
                      in_boundary.remove(i)
                  # 判断y是否超出界限
                  elif (boxes_xyxy[i, 1] < 0 and boxes_xyxy[i, 3] < 0) or (
                          boxes_xyxy[i, 1] > size and boxes_xyxy[i, 3] > size):
                      in_boundary.append(i)
              boxes_xyxy = boxes_xyxy[in_boundary]
              boxes = boxes_xyxy.clamp(min=0, max=size).reshape([-1, 4])  # 压缩到固定范围
              label = label[in_boundary]
              # 转换到YOLO格式
              x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
              xc = ((x1 + x2) / (2 * size)).reshape([-1, 1])
              yc = ((y1 + y2) / (2 * size)).reshape([-1, 1])
              wc = ((x2 - x1) / size).reshape([-1, 1])
              hc = ((y2 - y1) / size).reshape([-1, 1])
              boxes = torch.cat([xc, yc, wc, hc], dim=1)
          # 图像转换
          transform = transforms.CenterCrop(size)
          img = transform(img)
          if target_size:
              img = img.resize((target_size, target_size), Image.BILINEAR)
          if len(boxes) > 0:
              return img, torch.cat([label.reshape([-1, 1]), boxes], dim=1)
          else:
              return img, boxes
  
      # ------------------------------------------------------
      # 以下img皆为Tensor类型
      # ------------------------------------------------------
  
      def random_bright(self, img, u=120, p=1):
          # -------------------------------------
          # 随机亮度变换
          # -------------------------------------
          if np.random.random() < p:
              alpha = np.random.uniform(-u, u) / 255
              img += alpha
              img = img.clamp(min=0.0, max=1.0)
          return img
  
      def random_contrast(self, img, lower=0.5, upper=1.5, p=1):
          # -------------------------------------
          # 随机增强对比度
          # -------------------------------------
          if np.random.random() < p:
              alpha = np.random.uniform(lower, upper)
              img *= alpha
              img = img.clamp(min=0, max=1.0)
          return img
  
      def random_saturation(self, img, lower=0.5, upper=1.5, p=1):
          # 随机饱和度变换，针对彩色三通道图像，中间通道乘以一个值
          if np.random.random() < p:
              alpha = np.random.uniform(lower, upper)
              img[1] = img[1] * alpha
              img[1] = img[1].clamp(min=0, max=1.0)
          return img
  
      def add_gasuss_noise(self, img, mean=0, std=0.1):
          noise = torch.normal(mean, std, img.shape)
          img += noise
          img = img.clamp(min=0, max=1.0)
          return img
  
      def add_salt_noise(self, img):
          noise = torch.rand(img.shape)
          alpha = np.random.random() / 5 + 0.7
          img[noise[:, :, :] > alpha] = 1.0
          return img
  
      def add_pepper_noise(self, img):
          noise = torch.rand(img.shape)
          alpha = np.random.random() / 5 + 0.7
          img[noise[:, :, :] > alpha] = 0
          return img
  
  
  def plot_pics(img, boxes):
      # 显示图像和候选框，img是Image.Open()类型, boxes是Tensor类型
      plt.imshow(img)
      label_colors = [(213, 110, 89)]
      w, h = img.size
      for i in range(boxes.shape[0]):
          box = boxes[i, 1:]
          xc, yc, wc, hc = box
          x = w * xc - 0.5 * w * wc
          y = h * yc - 0.5 * h * hc
          box_w, box_h = w * wc, h * hc
          plt.gca().add_patch(plt.Rectangle(xy=(x, y), width=box_w, height=box_h,
                                            edgecolor=[c / 255 for c in label_colors[0]],
                                            fill=False, linewidth=2))
      plt.show()
  
  
  def get_image_list(image_path):
      # 根据图片文件，查找所有图片并返回列表
      files_list = []
      for root, sub_dirs, files in os.walk(image_path):
          for special_file in files:
              special_file = special_file[0: len(special_file)]
              files_list.append(special_file)
      return files_list
  
  
  def get_label_file(label_path, image_name):
      # 根据图片信息，查找对应的label
      fname = os.path.join(label_path, image_name[0: len(image_name) - 4] + ".txt")
      data2 = []
      if not os.path.exists(fname):
          return data2
      if os.path.getsize(fname) == 0:
          return data2
      else:
          with open(fname, 'r', encoding='utf-8') as infile:
              # 读取并转换标签
              for line in infile:
                  data_line = line.strip("\n").split()
                  data2.append([float(i) for i in data_line])
      return data2
  
  
  def save_Yolo(img, boxes, save_path, prefix, image_name):
      # img: 需要时Image类型的数据， prefix 前缀
      # 将结果保存到save path指示的路径中
      if not os.path.exists(save_path) or \
              not os.path.exists(os.path.join(save_path, "images")):
          os.makedirs(os.path.join(save_path, "images"))
          os.makedirs(os.path.join(save_path, "labels"))
      try:
          img.save(os.path.join(save_path, "images", prefix + image_name))
          with open(os.path.join(save_path, "labels", prefix + image_name[0:len(image_name) - 4] + ".txt"), 'w',
                    encoding="utf-8") as f:
              if len(boxes) > 0:  # 判断是否为空
                  # 写入新的label到文件中
                  for data in boxes:
                      str_in = ""
                      for i, a in enumerate(data):
                          if i == 0:
                              str_in += str(int(a))
                          else:
                              str_in += " " + str(float(a))
                      f.write(str_in + '\n')
      except:
          print("ERROR: ", image_name, " is bad.")
  
  
  def runAugumentation(image_path, label_path, save_path):
      image_list = get_image_list(image_path)
      for image_name in image_list:
          print("dealing: " + image_name)
          img = Image.open(os.path.join(image_path, image_name))
          boxes = get_label_file(label_path, image_name)
          boxes = torch.tensor(boxes)
          # 下面是执行的数据增强功能，可自行选择
          # Image类型的参数
          DAD = DataAugmentationOnDetection()
  
          """ 尺寸变换   """
          # 缩小尺寸
          # t_img, t_boxes = DAD.resizeDown_keep_ratio(img, boxes, 1024)
          # save_Yolo(t_img, boxes, save_path, prefix="rs_", image_name=image_name)
          # 水平旋转
          t_img, t_boxes = DAD.random_flip_horizon(img, boxes.clone())
          save_Yolo(t_img, t_boxes, save_path, prefix="fh_", image_name=image_name)
          # 竖直旋转
          t_img, t_boxes = DAD.random_flip_vertical(img, boxes.clone())
          save_Yolo(t_img, t_boxes, save_path, prefix="fv_", image_name=image_name)
          # center_crop
          t_img, t_boxes = DAD.center_crop(img, boxes.clone(), 1024)
          save_Yolo(t_img, t_boxes, save_path, prefix="cc_", image_name=image_name)
  
          """ 图像变换，用tensor类型"""
          to_tensor = transforms.ToTensor()
          to_image = transforms.ToPILImage()
          img = to_tensor(img)
  
          # random_bright
          t_img, t_boxes = DAD.random_bright(img.clone()), boxes
          save_Yolo(to_image(t_img), boxes, save_path, prefix="rb_", image_name=image_name)
          # random_contrast 对比度变化
          t_img, t_boxes = DAD.random_contrast(img.clone()), boxes
          save_Yolo(to_image(t_img), boxes, save_path, prefix="rc_", image_name=image_name)
          # random_saturation 饱和度变化
          t_img, t_boxes = DAD.random_saturation(img.clone()), boxes
          save_Yolo(to_image(t_img), boxes, save_path, prefix="rs_", image_name=image_name)
          # 高斯噪声
          t_img, t_boxes = DAD.add_gasuss_noise(img.clone()), boxes
          save_Yolo(to_image(t_img), boxes, save_path, prefix="gn_", image_name=image_name)
          # add_salt_noise
          t_img, t_boxes = DAD.add_salt_noise(img.clone()), boxes
          save_Yolo(to_image(t_img), boxes, save_path, prefix="sn_", image_name=image_name)
          # add_pepper_noise
          t_img, t_boxes = DAD.add_pepper_noise(img.clone()), boxes
          save_Yolo(to_image(t_img), boxes, save_path, prefix="pn_", image_name=image_name)
  
  
  
  if __name__ == '__main__':
      # 图像和标签文件夹
      image_path = r"C:\Users\19916\Desktop\1\datasets\mydata3\images\val"
      label_path = r"C:\Users\19916\Desktop\1\datasets\mydata3\labels\val"
      save_path = r"C:\Users\19916\Desktop\a"  # 结果保存位置路径，可以是一个不存在的文件夹
      # 运行
      runAugumentation(image_path, label_path, save_path)
  ```

  再次运行，得到的F1,P,R,PR曲线

  <img src=".\asset\trainA\第六次训练\F1_curve.png" alt="P_curve" style="zoom:25%;" />

  <img src=".\asset\trainA\第五次训练\P_curve.png" alt="P_curve" style="zoom:25%;" />

  <img src=".\asset\trainA\第五次训练\R_curve.png" alt="P_curve" style="zoom:25%;" />

  <img src=".\asset\trainA\第五次训练\PR_curve.png" alt="P_curve" style="zoom:25%;" />

  可以看出，F1曲线增长了0.1，所以说明数据增强效果较好。

**从以上六次中，发现motor,bike,rider,person的检测效果不好，接下来着重加强对小目标的检测**
