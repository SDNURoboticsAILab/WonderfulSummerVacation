### Stage2

**电脑配置：**
操作系统：Windows
GPU：NVIDIA GeForce RTX 3060 Laptop GPU
GPU计算能力：8.6
GPU内存：5.99951171875GB
多处理器数量：30个
CPU：12th Gen Intel(R) Core(TM) i9-12900H
CPU核心数：20个



```
import torch
import time
from transformers import BertTokenizer, BertModel

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

初始化模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)
model.eval()

# 准备输入数据
text = "Transformer is a deep learning model based on self attention mechanism."
inputs = tokenizer(text, return_tensors="pt").to(device)

# 推理速度测试
def measure_inference_speed(model, inputs, num_iterations=1000):
    torch.cuda.synchronize()  # 确保所有CUDA操作完成后再测量时间，以避免计时不准确。
    start_time = time.time()
    with torch.no_grad():   #  在推理阶段避免计算梯度，提高速度。
        for _ in range(num_iterations):
            model(**inputs)
    torch.cuda.synchronize()  # 确保所有计算完成
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_inference = total_time / num_iterations  # 修正了平均时间的计算
    return avg_time_per_inference, total_time

# 测试推理速度
avg_time, total_time = measure_inference_speed(model, inputs)
inferences_per_second = 1 / avg_time

print(f"Total time: {total_time:.6f} seconds")
print(f"Average time per inference: {avg_time:.6f} seconds")
print(f"Inferences per second: {inferences_per_second:.2f}")
```
用的模型为"bert-base-uncased",吞吐量为100左右，又将模型改为该模型的微调模型"distilbert-base-uncased",但吞吐量提升不大


**优化推理引擎**
```

import torch
import time
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型和tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name).to(device)
model.eval()

# 自定义Dataset
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return encoding

# 准备输入数据
texts = ["Transformer is a deep learning model based on self attention mechanism."] * 128  # 增加文本数量以适应批处理
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)  # 调整批量大小

# 推理速度测试
def measure_inference_speed(model, dataloader, num_iterations=1000):
    torch.cuda.synchronize()
    start_time = time.time()
    total_samples = 0
    with torch.no_grad():
        for _ in range(num_iterations):
            for batch in dataloader:
                inputs = {key: value.squeeze(1).to(device) for key, value in batch.items()}
                model(**inputs)
                total_samples += len(inputs['input_ids'])  # 计算处理的样本数量
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_inference = total_time / num_iterations
    inferences_per_second = total_samples / total_time
    return avg_time_per_inference, inferences_per_second, total_time

# 测试推理速度
avg_time, inferences_per_second, total_time = measure_inference_speed(model, dataloader)
print(f"Total time: {total_time:.6f} seconds")
print(f"Average time per iteration: {avg_time:.6f} seconds")
print(f"Inferences per second: {inferences_per_second:.2f}")
```

对模型进行了批量处理，使用`DataLoader`和较大的批量大小来减少处理次数并优化GPU利用率，吞吐量有了较大的提升,达到230左右

同时尝试通过实现异步数据加载以优化吞吐量，可以使用PyTorch的`DataLoader`的多线程/进程功能（`num_workers`参数）。这样，数据加载可以在多个线程/进程中异步进行，减少数据准备的时间浪费。

**`num_workers`**：设置为4，表示使用4个子进程进行数据加载。可以根据硬件性能调整此值来优化性能。

**pin_memory**：设置为`True`，使数据在加载到GPU之前被锁定在内存中，从而加快数据转移速度。

但在运行中出现报错：
``` 
raise RuntimeError(''' RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.

    This probably means that you are not using fork to start your
    child processes and you have forgotten to use the proper idiom
    in the main module:

        if __name__ == '__main__':
            freeze_support()
            ...

    The "freeze_support()" line can be omitted if the program
    is not going to be frozen to produce an executable.

```
已了解报错原因，代码仍在修改中,同时进行将BERT 模型转换为 ONNX 并使用 ONNX Runtime 加速推理，来优化，下载必要的库 `pip install transformers onnx onnxruntime`,目前仍在优化中。

