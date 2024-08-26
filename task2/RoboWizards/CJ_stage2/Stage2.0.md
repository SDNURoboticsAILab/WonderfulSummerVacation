

### Stage1

***

### 构建推理引擎

### 问题：

通过前期对Transformer、Tensor-RT的构建，发现有以下问题：

* 建立的环境(比如：cuda-toolkit、pytorch、cuda与其他包......)不匹配
* wsl下Ubuntu22.04的网络有问题（）,后来代理问题改为google的DNS才好点
* 在运行官方给的库时，始终存在运行到一部分，有一部分官方给的代码我的环境无法运行，报错，修改的结果就是越改越错。后来不得已，将miniconda创建的环境删了，又重新开始，又会有新的错误，真的很糟心......
* 官方的教程可能不全，有部分包得等运行报错时，才晓得还差哪些包
* 下载报错有时候分不清是网络问题还是环境问题
* 有时下载包时，总会卡在某个地方，卡几个小时都不报错

## 目前主要是对Transformers推理引擎构建

#### 环境配置：

**Transformers版本**：4.45.0dev

**cuda版本**:12.1

**cuda--toolkit版本**：cuda--toolkit-12-6

**部署环境**：wsl2-ubuntu22.04

**GPU**:nvidia-3050

**模型**：openai-community/gpt2

```import time
from transformers import pipeline

# 创建一个文本生成管道
pipe = pipeline(model="gpt2", device=0)

# 定义数据生成器
def data():
    for i in range(1000):
        yield f"My example {i}"

# 初始化token计数器
generated_tokens = 0

# 开始计时
start_time = time.time()

# 处理数据
for out in pipe(data(), max_length=50):  # 假设每个输入的最大长度是50 tokens
    generated_tokens += len(out[0]["generated_text"].split())  # 假设每个词对应一个token

# 结束计时
end_time = time.time()

# 计算总时间
total_time = end_time - start_time

# 计算每秒推理的token数量
inference_speed_tokens_per_second = generated_tokens / total_time

print(f"Total time for inference: {total_time:.2f} seconds")
print(f"Generated tokens: {generated_tokens}")
print(f"Inference speed: {inference_speed_tokens_per_second:.2f} tokens/second")
```

推理引擎的吞吐量（每秒推理数）：85.1 tokens/second

<img src=".\asset\2.0\f79ab02a297b41c157535309136d12e6.png" alt="f79ab02a297b41c157535309136d12e6" style="zoom: 50%;" />

<img src=".\asset\2.0\image-20240814143738017.png" alt="image-20240814143738017" style="zoom: 25%;" />

## stage2

***

### 优化推理引擎

```from transformers import pipeline
import torch
from transformers import pipeline, set_seed, GPT2Tokenizer, GPT2LMHeadModel
import time
import multiprocessing

def worker_init():
    global pipe
    # 使用 GPT2Tokenizer 和 GPT2LMHeadModel 自定义 pipeline
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # 设置模型的数据类型为 bfloat16
    model = model.to(torch.bfloat16)
    
    # 设置注意力实现（如果模型支持）
    # 注意：这个参数可能需要根据模型的具体实现进行调整
    # 在这里，我们假设模型有一个可以设置的属性 attn_implementation
    # 如果这个属性不存在，可能需要修改模型代码或使用不同的方法
    if hasattr(model.config, 'attn_implementation'):
        model.config.attn_implementation = "flash_attention_2"
    
    # 使用自定义模型和 tokenizer 创建 pipeline
    pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer, device=0, batch_size=1000)

def process_text(text):
    result = pipe(text)
    return result

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    def data():
        for i in range(1000):
            yield f"My example {i}"

    generated_tokens = 0
    start_time = time.time()

    with multiprocessing.Pool(processes=6, initializer=worker_init) as pool:
        results = pool.imap_unordered(process_text, data())
        for out in results:
            for item in out:
                generated_tokens += len(item["generated_text"].split())

    end_time = time.time()
    total_time = end_time - start_time
    inference_speed_tokens_per_second = generated_tokens / total_time

    print(f"Total time for inference: {total_time:.2f} seconds")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Inference speed: {inference_speed_tokens_per_second:.2f} tokens/second")
```

batch_size=5,processes=4:

<img src=".\asset\2.0\image-20240814183823489.png" alt="image-20240814183823489" style="zoom: 50%;" />

batch_size=16,processes=1:

<img src=".\asset\2.0\image-20240814185546473.png" alt="image-20240814185546473" style="zoom:50%;" />

batch_size=10000,processes=1:

<img src=".\asset\2.0\image-20240814190758726.png" alt="image-20240814190758726" style="zoom:50%;" />

batch_size=10000,processes=6:

<img src=".\asset\2.0\image-20240814191437256.png" alt="image-20240814191437256" style="zoom:50%;" />

batch_size=1000,processes=6,flash_attention2:

<img src=".\asset\2.0\image-20240815100955444.png" alt="image-20240815100955444" style="zoom:50%;" />

## Stage3

### 部署OpenVLA并开发机器人API接口

下载open-vla模型，下载好dataset数据集（124GB）：

<img src=".\asset\2.0\ff43b4f90290dba08a3adda3730f1de2.png" alt="ff43b4f90290dba08a3adda3730f1de2" style="zoom: 50%;" />

通过lora进行微调(原实验采用的是一块带有 80 GB VRAM 的单个 A100 GPU，如果采用全面微调的话，)：

* 最初用了一块NVIDIA GeForce RTX 4060 Ti：发现即使是将batch_size调至1，grad_accumulation_steps 调至1024，还是爆显存。

<img src=".\asset\2.0\733e28f1c3e95a997d8144a1e5957516.png" alt="733e28f1c3e95a997d8144a1e5957516" style="zoom:50%;" />

#### 实践发现：

后面即使采用4块NVIDIA GeForce RTX 4060 Ti，依然爆显存，虽然采用了4块，但是依旧爆显存。

既有多个 GPU ，也只能通过通过 PyTorch 分布式数据并行 (DDP) 进行训练。

可见对单个GPU的显存要求至少为27G。

## 通过 LoRA 微调 OpenVLA的代码：

```torchrun --standalone --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "/hy-tmp/openvla-7b" \
  --data_root_dir "/hy-tmp/dataset" \
  --dataset_name bridge_orig \
  --run_root_dir /hy-tmp/1/experiment_logs/ \
  --adapter_tmp_dir /hy-tmp/1/adapter_weights \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 10240 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project tes1 \
  --wandb_entity galaxry-shandong-normal-university \
  --save_steps 1000
```

结果：

![029bc7373b33e81f19fb8dc12d0a0cab](D:\211\WonderfulSummerVacation\task2\RoboWizards\CJ_stage2\asset\2.0\029bc7373b33e81f19fb8dc12d0a0cab.png)

![image-20240817213956569](D:\211\WonderfulSummerVacation\task2\RoboWizards\CJ_stage2\asset\2.0\image-20240817213956569.png)
