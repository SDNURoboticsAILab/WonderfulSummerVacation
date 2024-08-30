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

cuda:  12.1



使用管道函数实现初级功能

```
from transformers import pipeline

# 创建一个用于文本情感分类的pipeline，并指定模型和设备
classifier = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased',
    device=0  # 指定使用GPU，如果您的环境有GPU的话
)

# 使用pipeline进行文本分类
result = classifier('It is my favourite food,but I can not eat it.So if you realy can give it to me, I will be very grateful.')

print(result)

#阅读理解
question_answerer=pipeline("question-answering")

context=r"""
Extractive Question Answering is the task of extracting an answer from question answering
dataset is the SQuAD dataset,which is entirely build a model on a SQuAD task,you may 
leverage the examples/pytorch/question"""

result=question_answerer(question="What is extractive question answering",
                         context=context)
print(result)

result=question_answerer(
    question="What is a good example of a question answering dataset",
    context=context
)
print(result )

# 完形填空
unmasker=pipeline("fill-mask")
from pprint import pprint
sentence='HuggingFace if creating a <mask> that the community uses.'
print(unmasker(sentence))


#文本生成
text_generator=pipeline("text-generation")
text_generator("As far as I am concerned, I will",
               max_length=50,
               do_sample=False)


#命名实体识别
ner_pipe=pipeline("ner")
sequence="""Hugging Face Inc. is a company based in New York City.
therefore very close to the Manhattan Btidge which is visible from t"""

for entity in ner_pipe(sequence):
    print(entity)
```





**吞吐量**


```

import torch
from transformers import AutoModel, AutoTokenizer
import time

# 准备环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和分词器
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name).to(device, dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name,clean_up_tokenization_spaces=False)


# 准备数据
input_text = "Transformer is a deep learning model based on self attention mechanism."
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 禁用梯度计算
torch.no_grad()

# 模型评估模式
model.eval()

# 测量时间
start_time = time.time()
for _ in range(1000):  # 假设我们重复推理100次以获得更稳定的测量结果
    outputs = model(**inputs)
end_time = time.time()

# 计算吞吐量
total_time = end_time - start_time
throughput = 1000 / total_time  # 每秒钟处理的样本数

print(f"Total time: {total_time} seconds")
print(f"Throughput: {throughput} samples/second")

```



本人运用 bert-base-uncased 模型，对 Transformer 吞吐量进行测试，吞吐量一般 100 samples/second 左右，优化还在测试中。