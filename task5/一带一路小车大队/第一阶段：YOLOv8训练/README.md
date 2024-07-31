# 第一阶段：YOLOv8 训练

## BDD100K 数据集

https://arxiv.org/pdf/1805.04687

1. 数据集规模大:包含100,000个多样化的驾驶视频剪辑,是目前最大的驾驶场景数据集。
2. 数据多样性强:覆盖了多种天气条件、时间、城市和场景类型,有助于训练鲁棒性强的模型。
3. 标注丰富:提供了10种不同的视觉感知任务标注,包括图像分类、车道线检测、可驾驶区域分割、目标检测、语义分割、实例分割、多目标跟踪、多目标分割跟踪等。
4. 支持异构多任务学习:相比于同构的多任务学习设置,BDD100K的多任务设置涉及不同复杂度和输出结构的任务,可以促进异构多任务学习的发展。
5. 数据来源广泛:通过众包的方式从大量驾驶员那里收集视频数据,覆盖地域分布广泛。

## 常规训练流程

### 标注

数据集已经完成了标注。

数据量较大，人工重新标注十分繁琐。

因此后续操作不涉及对数据标注的操作。



### 环境配置

#### 创建 conda环境

```bash
conda create -n yolov8 python==3.9
conda init
conda activate yolov8
```

#### 安装 Torch

```bash
pip install torchvision==0.16.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 安装 Ultralytics

```bash
pip install ultralytics
```

#### 检测 Torch 是否安装成功

```python
import torch

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
print(f"os.environ['CUDA_VISIBLE_DEVICES']: {os.environ['CUDA_VISIBLE_DEVICES']}")
```

### 标签转换

[赛项一指导文件](./赛项一指导文件/行人车辆检测与计数.md#标签转换) 相关部分为我们提供了 BDD100K 数据集标签归一化处理为 YOLO 标签的转换脚本

```python
import os

# 假设所有图片的大小是固定的，根据你的实际情况调整
img_width = 1280
img_height = 720

def convert_bbox_to_yolo_format(x_min, y_min, x_max, y_max, img_w, img_h):
    """
    将边界框从 <x_min, y_min, x_max, y_max> 转换为 YOLO 格式 <x_center, y_center, width, height>，
    并归一化坐标。
    """
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

def process_label_files(label_dir):
    for label_file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, label_file)
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 准备新的标签内容
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, x_min, y_min, x_max, y_max = map(float, parts)
                x_center, y_center, width, height = convert_bbox_to_yolo_format(
                    x_min, y_min, x_max, y_max, img_width, img_height)
                new_line = f"{int(cls)}\t{x_center:.6f}\t{y_center:.6f}\t{width:.6f}\t{height:.6f}\n"
                new_lines.append(new_line)

        # 将转换后的标签写回文件
        with open(file_path, 'w') as file:
            file.writelines(new_lines)

if __name__ == "__main__":
    label_directory = 'trainA/labels/label-trainA'  # 修正路径中的反斜杠
    process_label_files(label_directory)

print("convert over")
```

### 数据集划分

BDD100K 数据集结构为：

```
├── images
├── labels
```

而 YOLO 的数据集结构为：

```
├── images
│   ├── train
│   ├── val
│   └── test
├── labels
│   ├── train
│   ├── val
│   └── test
```

[赛项一指导文件](./赛项一指导文件/行人车辆检测与计数.md#创建split_data.py) 相关部分为我们提供了训练集：校验集：测试集=8:1:1 的比例进行划分的脚本

```python
import os
import random
from shutil import copyfile

def split_dataset(image_folder, txt_folder, output_folder, split_ratio=(0.8, 0.1, 0.1)):
    # Ensure output folders exist
    for phase in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_folder, 'images', phase), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'labels', phase), exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    num_images = len(image_files)
    num_train = int(split_ratio[0] * num_images)
    num_val = int(split_ratio[1] * num_images)

    train_images = image_files[:num_train]
    val_images = image_files[num_train:num_train + num_val]
    test_images = image_files[num_train + num_val:]

    # Copy images and labels to respective folders
    for phase, images_list in zip(['train', 'val', 'test'], [train_images, val_images, test_images]):
        for image_file in images_list:
            # Copy image
            image_path = os.path.join(image_folder, image_file)
            target_image_path = os.path.join(output_folder, 'images', phase, image_file)
            copyfile(image_path, target_image_path)
            
            # Copy corresponding txt file if exists
            txt_file = os.path.splitext(image_file)[0] + '.txt'
            txt_path = os.path.join(txt_folder, txt_file)
            target_txt_path = os.path.join(output_folder, 'labels', phase, txt_file)
            if os.path.exists(txt_path):
                copyfile(txt_path, target_txt_path)

if __name__ == "__main__":
    image_folder_path = "trainA/images"
    txt_folder_path = "trainA/labels"
    output_dataset_path = "datasets"

    split_dataset(image_folder_path, txt_folder_path, output_dataset_path)

print("Split complete.")
```

### 创建 YAML 文件
```yaml
path: path/to/datasets  # dataset root dir
train: images/train # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 9  # number of classes
names: ['bus','traffic light', 'traffic sign', 'person', 'bike', 'truck', 'motor', 'car', 'rider', 
        ]  # class names
```

### 开始训练

创建 `train.py` 文件并运行

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')  # 从YAML构建并转移权重

if __name__ == '__main__':
    # 训练模型
    results = model.train(data='path/to/yaml', epochs=10, imgsz=512, device=1)
```

## 数据集优化

### 图片剔除

质量不好的照片通常指那些由于各种原因导致图像无法为模型提供有效信息的照片。以下是一些常见的质量问题及其剔除方法：

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

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第五次训练/F1_curve.png" alt="P_curve" style="zoom:25%;" />

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第五次训练/P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第五次训练/R_curve.png" alt="P_curve" style="zoom:25%;" />

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第五次训练/PR_curve.png" alt="P_curve" style="zoom:25%;" />

可以看到，在未修改的F1中，指数为0.233-0.53，而在剔除照片之后指数为0.273-0.49，所以说明保留照片的质量性较好。

接下来将剔除后的图片进行数据增强。

完整脚本

```python
import cv2
import numpy as np
import os
import glob



from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

def run(func, this_iter, desc="Processing"):
    with ThreadPoolExecutor(max_workers=NUM_THREADS, thread_name_prefix='MyThread') as executor:
        results = list(
            tqdm(executor.map(func, this_iter), total=len(this_iter), desc=desc)
        )
    return results

def is_blurry(image_path, threshold=100):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold


def is_overexposed(image_path, threshold=200):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return np.argmax(hist) > threshold


def is_underexposed(image_path, threshold=50):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return True
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return np.argmax(hist) < threshold


def is_low_resolution(image_path, min_width=640, min_height=480):
    image = cv2.imread(image_path)
    if image is None:
        return True
    height, width = image.shape[:2]
    return width < min_width or height < min_height


def has_invalid_annotation(label_path):
    if not os.path.exists(label_path):
        return True
    with open(label_path, 'r') as file:
        lines = file.readlines()
    return len(lines) == 0


def process_images(image_dir, label_dir, min_width=640, min_height=480, blur_threshold=100, exposure_threshold_high=200,
                   exposure_threshold_low=50):
    image_paths = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))

    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        label_path = os.path.join(label_dir, os.path.splitext(base_name)[0] + '.txt')

        if (is_blurry(image_path, blur_threshold) or
                is_overexposed(image_path, exposure_threshold_high) or
                is_underexposed(image_path, exposure_threshold_low) or
                is_low_resolution(image_path, min_width, min_height) or
                has_invalid_annotation(label_path)):
            print(f'Removing {image_path}')
            os.remove(image_path)
            if os.path.exists(label_path):
                os.remove(label_path)


if __name__ == "__main__":
    IMAGE_DIR = r'C:\Users\19916\Desktop\1\datasets\trainB-SCI+Filter\images\val'
    LABEL_DIR = r'C:\Users\19916\Desktop\1\datasets\trainB-SCI+Filter\labels\val'
    process_images(IMAGE_DIR, LABEL_DIR)
```

### 图片增强

对上述剔除后的照片进行数据增强。

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

```python
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



## 更换框架

尝试更换 efficientnetv2，未果

## 训练算法优化

### 	改进soft_nms非极大值抑制算法（trainA）

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

​						<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第三次训练/F1_curve.png" alt="P_curve" style="zoom:25%;" />

​                

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第三次训练/PR_curve.png" alt="P_curve" style="zoom:25%;" />

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第三次训练/P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第三次训练/R_curve.png" alt="P_curve" style="zoom:25%;" />

​	结果分析，F1曲线效果并不好，意味着此改进并不能对结果进行优化。

### Bifpn网络结构（trainA）

使用Bifpn进行改进

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

​						<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第四次训练/F1_curve.png" alt="P_curve" style="zoom:25%;" />

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第四次训练/P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第四次训练/R_curve.png" alt="P_curve" style="zoom:25%;" />

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainA/第四次训练/PR_curve.png" alt="P_curve" style="zoom:25%;" />

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

### SCI夜间增强技术处理照片（trainB）

## **

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

```python
def forward(self, input):
	# 将输入数据通过输入卷积层 self.in_conv，得到中间特征 fea。
	fea = self.in_conv(input)
	# 通过多个中间卷积层 self.blocks 进行迭代，
	for conv in self.blocks:
		# 每次迭代中将当前特征 fea 与中间卷积层的输出相加。
		fea = fea + conv(fea)
	fea = self.out_conv(fea)
	# 将输出特征与输入数据相加，得到增强后的图像。
	illu = fea + input
	# 通过 torch.clamp 函数将图像的像素值限制在 0.0001 到 1 之间，以确保输出在有效范围内。
	illu = torch.clamp(illu, 0.0001, 1)
	return illu
```


​	 	通过多次堆叠卷积块，来学习图像的特征。增强后的图像通过与输入相加并进行截断，以确保像素值在合理范围内。

​	首先，通过__init__初始化超参数和网络层，参数设置看注释。

​	然后，将输入图像通过3*3的卷积层，得到特征 fea，对特征 fea 多次应用相同的卷积块进行叠加，通过输出卷积层获得最终的特征 	fea。

​	接着，将生成的特征与输入图像相加，得到增强后的图像，通过 clamp 函数将图像像素值限制在 0.0001 和 1 之间。

​	最后，返回增强后的图像 illu。

```python
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

```python
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

​	<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainB/第二次训练/train_B_1535.jpg" alt="P_curve" style="zoom:25%;" />
<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainB/第二次训练/train_B_1535.png" alt="P_curve" style="zoom:25%;" />  

​	可以看到有了较为明显的改进。

​	得到的F1,P,R,PR

​						<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainB/第二次训练/F1_curve.png" alt="P_curve" style="zoom:25%;" />				

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainB/第二次训练/P_curve.png" alt="P_curve" style="zoom:25%;" />

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainB/第二次训练/R_curve.png" alt="P_curve" style="zoom:25%;" />

<img src="D:/LERNAS/COMPETITON/BRICS/2024/option1/第一阶段：YOLOv8训练/yolov8-zht/asset/trainB/第二次训练/PR_curve.png" alt="P_curve" style="zoom:25%;" />

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



## 非常规报错解决

经过一系列操作，torch 总算可以调用 GPU 运行了，但是出现以下报错

```bash
Traceback (most recent call last): File "<string>", line 1, in <module> File "C:\Users\excnies\.conda\envs\yolov8c123\lib\multiprocessing\[spawn.py](http://spawn.py/)", line 116, in spawn_main exitcode = _main(fd, parent_sentinel) File "C:\Users\excnies\.conda\envs\yolov8c123\lib\multiprocessing\[spawn.py](http://spawn.py/)", line 126, in _main self = reduction.pickle.load(from_parent) File "C:\Users\excnies\.conda\envs\yolov8c123\lib\multiprocessing\[connection.py](http://connection.py/)", line 962, in rebuild_pipe_connection handle = dh.detach() File "C:\Users\excnies\.conda\envs\yolov8c123\lib\multiprocessing\[reduction.py](http://reduction.py/)", line 131, in detach return _winapi.DuplicateHandle( PermissionError: [WinError 5] 拒绝访问。
```

根据报错直接修改权限并没有解决问题。经过搜索，看到 YOLO v5 的 Isuue 有类似的问题

https://github.com/ultralytics/yolov5/issues/8626

试图修改所有的 [train.py](http://train.py)，问题依然没有解决

将 device 设置为 CPU，看到如下日志

```bash
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
```

由于 v8 取消了主程序的 [train.py](http://train.py)，所以直接使用 Python 脚本运行

https://github.com/pytorch/csprng/issues/115

https://community.anaconda.cloud/t/omp-error-15-initializing-libiomp5md-dll-but-found-libiomp5-already-initialized/48993

为运行的脚本添加环境变量`["KMP_DUPLICATE_LIB_OK"] = "TRUE"`即可

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

if __name__ == '__main__':
    # 训练模型
    results = model.train(data='sloppy.yaml', epochs=10, imgsz=512)
```

