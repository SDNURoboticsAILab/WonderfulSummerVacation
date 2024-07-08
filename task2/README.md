# Code reasoner based on large language model

## Introduction

When SDNU freshman class of 2023 arrived in 2023,Huang_Shige wrote a intelligent dialogue software for the robot which is based on LLM.But with the development of embodied intelligence, it looks less intelligent. So, your goal is to develop an LLM-based behavior generation engine.

[RoboCodeX](https://arxiv.org/abs/2402.16117) is a framework which generate multimodal code for robotic behavior synthesis. You need to develop an intelligent engine based on this paper which let the robot understand what you're saying and react accordingly.

Let's say we want to build this engine on the nao robot. Here's an example:

```txt
User: Can you help me find the red ball on the ground?
Nao: Ok, just a moment.
// The robot receives the code provided by the engine and automatically runs the code to complete the relevant instructions
```

Please refer to the following "**Procedure**" to complete the relevant work.

## Procedure

### stage1

1. Read the paper summarizing robocodeX and check out related papers.
2. Based on these papers, Based on these papers, please design such a platform architecture and identify the needs to be addressed.
3. Please write the design document and draw the system architecture diagram.

**Note:** You need to verify the feasibility of these designs

### stage2

1. Based on your design, Implement your architecture.
2. After the design is complete, test the corresponding functionality

**Note:** It is recommended to build your local LLM module(Llama,Qwen,Phi,OPT.....) instead of directly calling others engine interface. If you want to complete stage 3, choose a local large model which can fine-tune。

### stage3 (optional)

Fine-tuning large models: On the basis of selecting relevant datasets and pre-trained models, the model is trained with task-specific data to optimize its performance by setting appropriate hyperparameters and making necessary adjustments to the model.

1. Build your own dataset.
2. Fine-tune your model.

## Submit

Any details and learning about the deployment process are worth putting in your folder.
