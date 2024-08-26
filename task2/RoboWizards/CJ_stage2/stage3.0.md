#### 环境配置：

**Transformers版本**：4.39.3

**cuda版本**:12.1

**cuda--toolkit版本**：cuda--toolkit-12-1

**部署环境**：Linux-Ubuntu20.04

**GPU**:nvidia-3090(显存：24GB)

**模型**：Qwen2-7B-instruct

### 第一阶段：构建推理引擎

#### 推理

使用**Transformers**实现Chat

实现代码：

以下是一个如何与 **Qwen2-7B-Instruct** 进行对话的示例：

```
from transformers import pipeline

pipe = pipeline("text-generation", "/hy-tmp/Qwen2-7B-Instruct", torch_dtype="auto", device_map="auto")

# the default system message will be used
messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]

response_message = pipe(messages, max_new_tokens=512)[0]["generated_text"][-1]
messages.append(response_message)

prompt = "Tell me more."
messages.append({"role": "user", "content": prompt})

response_message = pipe(messages, max_new_tokens=512)[0]["generated_text"][-1]
```

如要继续对话，只需将回复内容以 assistant 为 role 加入 messages ，然后重复以上流程即可。下面为示例：

```
messages.append(response_message)

prompt = "Tell me more."
messages.append({"role": "user", "content": prompt})

response_message = pipe(messages, max_new_tokens=512)[0]["generated_text"][-1]
```

#### 批处理

```
from transformers import pipeline

pipe = pipeline("text-generation", "/hy-tmp/Qwen2-7B-Instruct", torch_dtype="auto", device_map="auto")
pipe.tokenizer.padding_side="left"

message_batch = [
    [{"role": "user", "content": "Give me a detailed introduction to large language model."}],
    [{"role": "user", "content": "Hello!"}],
]

result_batch = pipe(messages, max_new_tokens=512, batch_size=2)
response_message_batch = [result[0]["generated_text"][-1] for result in result_batch]
```

#### 流式输出

```
from transformers import pipeline, TextStreamer

pipe = pipeline(
    "text-generation", 
    "/hy-tmp/Qwen2-7B-Instruct", 
    torch_dtype="auto", 
    device_map="auto", 
    model_kwargs=dict(attn_implementation="flash_attention_2"),
)
messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]
streamer = TextStreamer(pipe.tokenizer, skip_prompt=True, skip_special_tokens=True)

response_message = pipe(messages, max_new_tokens=512, streamer=streamer)[0]["generated_text"][-1]
```

回答：

<img src=".\asset\3.0\image-20240818115502705.png" alt="image-20240818115502705" style="zoom:50%;" />

##### 运行Qwen的GGUF文件 

##### llama.cpp

```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
```

```
make
```

然后你就能在 `llama.cpp`目录下运行GGUF文件。

```
 ./llama-cli -m /hy-tmp/qwen2-7b-instruct-q5_k_m.gguf \
-n 512 -co -i -if -f prompts/chat-with-qwen.txt \
--in-prefix "user:\n" \
--in-suffix ".\n<|im_start|>assistant\n" \
-ngl 80 -fa
```

交互结果：

<img src=".\asset\3.0\image-20240818130944418.png" alt="image-20240818130944418" style="zoom:50%;" />

1. **加载时间**： 模型加载所需的时间是4665.05毫秒。
2. **采样时间**： 每次采样（生成响应）所需的时间是66.68毫秒，总共进行了153次采样。这表示每次采样平均需要66.68毫秒，并且模型可以以每秒2294.44个token的速度进行采样。
3. **提示评估时间**： 对提示进行评估所需的时间是54358.16毫秒，总共评估了51个token。这表示每个token的平均评估时间是1065.85毫秒，并且模型可以以每秒0.94个token的速度进行评估。
4. **评估时间**： 每次评估（包括采样和提示评估）所需的总时间是63204.42毫秒，总共进行了151次评估。这表示每次评估的平均时间是418.57毫秒，并且模型可以以每秒2.39个token的速度进行评估。
5. **总时间**： 进行202个token的交互所需的总时间是137643.13毫秒。

### 第 2 阶段：优化推理引擎

## 部署

### **采用vllm部署：**

#### 离线推理：

Qwen2代码支持的模型都被vLLM所支持。 vLLM最简单的使用方式是通过以下演示进行离线批量推理。

```
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/hy-tmp/Qwen2-7B-Instruct")

# Pass the default decoding hyperparameters of Qwen2-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=256)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model="/hy-tmp/Qwen2-7B-Instruct", dtype='float16')

# Prepare your prompts
prompt = "Tell me something about large language models."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

**推理结果：**

<img src=".\asset\3.0\image-20240820160627333.png" alt="image-20240820160627333" style="zoom:50%;" />

```# vllm_model.py
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json

# 自动下载模型时，指定使用modelscope。不设置的话，会从 huggingface 下载
os.environ['VLLM_USE_MODELSCOPE']='True'

def get_completion(prompts, model, tokenizer=None, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":    
    # 初始化 vLLM 推理引擎
    model='/root/autodl-tmp/qwen/Qwen2-7B-Instruct' # 指定模型路径
    # model="qwen/Qwen2-7B-Instruct" # 指定模型名称，自动下载模型
    tokenizer = None
    # 加载分词器后传入vLLM 模型，但不是必要的。
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False) 
    
    text = ["你好，帮我介绍一下什么时大语言模型。",
            "可以给我将一个有趣的童话故事吗？"]
    # messages = [
    #     {"role": "system", "content": "你是一个有用的助手。"},
    #     {"role": "user", "content": prompt}
    # ]
    # 作为聊天模板的消息，不是必要的。
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )

    outputs = get_completion(text, model, tokenizer=tokenizer, max_tokens=512, temperature=1, top_p=1, max_model_len=2048)

    # 输出是一个包含 prompt、生成文本和其他信息的 RequestOutput 对象列表。
    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}"
```

结果如下：

```Prompt: '你好，帮我介绍一下什么时大语言模型。', Generated text: ' 当然！大语言模型是人工智能中的一种模型，特别擅长生成高质量的文本。它们从大量的文本数据中学习，并可以生成类似真实 文本的文本片段。例如，让它们写故事、文章、诗歌，或者在对话中生成连贯的回答。这类模型也被用于许多其他自然语言处理任务，如文本摘要、翻译和代码生成。这是因为它们能够理解和生成复杂的 语法和语义结构，以及捕捉到上下文中的微小细节。大语言模型的核心是采用深度学习技术，尤其是基于Transformer架构的模型，这种架构很好地处理了大量的序列数据，并在最近几年取得了显著的进展，这得益于大规模的训练数据集和计算资源。如今，许多大型语言模型是开源的，并且应用于各种开发和研究环境中。'

Prompt: '可以给我将一个有趣的童话故事吗？', Generated text: ' 当然可以。这是一个关于勇敢的小猫头鹰的主题的童话故事：\n\n从前，在一片宁静的森林深处，住着一个聪明而勇敢的小猫头鹰。 它的名字叫迈克。每天，它都会在夜色中穿梭，寻找食物和学习更多的生存之道。它的家是一个它自己制作的巨大鸟巢，挂在一片松树的高枝上。\n\n一天夜里，森林受到了威胁，因为一只贪婪的老母狼 叛领了一队强盗在他的领地打劫。所有动物都陷入了恐慌，胆小的们躲在家里不敢出来，而胆大的动物们则是四处逃难。但是，没有一只动物敢于挑战母狼。\n\n作为勇敢和智慧的象征，小猫头鹰迈克决 定挺身而出。它认识到单靠野兽的力量是无法对抗母狼及其随从的，但是凭借智慧与策略，它或许可以找到一条解决方案。\n\n不日，迈克带着一个大胆的计划回到了森林。它宣布，所有的生物都将暂时 放下彼此之间的争斗，携手合作对抗这场危机。为了做到这一点，迈克将动物们聚集在一起，让迷人的动物学者白鹤教授教授所有生物如何彼此沟通、理解，并动员各具专业能力的动物，如挖掘专家老鼠 、电子设备专家松鼠制作无线电来秘密向森林里的其他动物发送求助信息。\n\n计划逐渐展开，动物们开始有了防范意识，并在夜晚骚动的女狼群不知道任何人计划的时候做出了各种有效的防御。动物中 个个都贡献了他们的力量。兔子与貘堵住了几个重要的入口，灵巧的松鼠们则收集了大量的浆果和营养物质，以供整个森林的动物们补充能量。\n\n最后，在一场夜里的明智逮捕行动之后，迈克的小猫头 鹰巧妙地通过其较好的夜视和听力，联合瞳熊和狮子成功的将贪婪的老母狼及其共犯赶出了森林。\n\n消息遍传，所有动物都对小猫头鹰的智慧，勇敢以及作为团队领袖的力量表示了敬意。他们现在紧紧 团结在了一起，建立了和谐而有尊严的社群。\n\n从此，森林中充满了欢声笑语，动物们和小猫头鹰迈克一起快乐地生活在和平与和谐中，展现出团结与智慧的伟大力量。这则故事教会我们，当我们团结 一致，敢于面对困难，发挥创造力和共同努力时，没有什么不可能克服的。'
```

#### 速度测试

既然 `vLLM` 是一个高效的大型语言模型推理和部署服务系统，那么我们不妨就测试一下模型的回复生成速度。看看和原始的速度相比有多大的提升。这里直接使用 `vLLM` 自带的 `benchmark_throughput.py` 脚本进行测试。

下面是一些 `benchmark_throughput.py` 脚本的参数说明：

- `--model` 参数指定模型路径或名称。
- `--backend` 推理后端，可以是 `vllm`、`hf` 和 `mii`。分布对应 `vLLM`、`HuggingFace` 和 `Mii` 推理后端。
- `--input-len` 输入长度
- `--output-len` 输出长度
- `--num-prompts` 生成的 prompt 数量
- `--seed` 随机种子
- `--dtype` 数据类型
- `--max-model-len` 模型最大长度
- `--hf_max_batch_size` `transformers` 库的最大批处理大小（仅仅对于 `hf` 推理后端有效且为必填字段）
- `--dataset` 数据集路径。（未设置会自动生成数据）

```python benchmark_throughput.py \
	--model /root/autodl-tmp/qwen/Qwen2-7B-Instruct \
	--backend vllm \  # 
	--input-len 64 \
	--output-len 128 \
	--num-prompts 25 \
	--seed 2024 \
    --dtype float16 \
    --max-model-len 512
```

<img src=".\asset\3.0\e4821d2a0fce0a5c82198bdf732a5a8a.png" alt="e4821d2a0fce0a5c82198bdf732a5a8a" style="zoom:50%;" />

**Throughput: 7.71 requests/s, 1479.95 tokens/s**

* **测试原始方式（即使用 `HuggingFace` 的 `Transformers` 库）**推理速度的命令和参数设置

```python benchmark_throughput.py \
	--model /root/autodl-tmp/qwen/Qwen2-7B-Instruct \
	--backend hf \  # 
	--input-len 64 \
	--output-len 128 \
	--num-prompts 25 \
	--seed 2024 \
	--dtype float16 \
    --hf-max-batch-size 25
```

**Throughput: 4.17 requests/s, 800.48 tokens/s**

对比两者的推理速度，在本次测试中 `vLLM` 的速度要比原始的速度快 **84%** 左右 

| 推理框架           | requests/s | tokens/s |
| ------------------ | ---------- | -------- |
| `vllm`             | 7.68       | 1479.95  |
| `hf(Transformers)` | 5.73       | 800.48   |

参考文献：

[llama.cpp - Qwen](https://qwen.readthedocs.io/zh-cn/latest/run_locally/llama.cpp.html)

