# Raspberry PI ASR Reasoner using Cross-compilation

## Introduction

Recently, Ma_shijie wanted to make an Streaming ASR reasoner based on Raspberry PI 4B, and she needed to deploy this software on plenty of Raspberry PIs, and she needed to modify the code. Please refer to the following "**Procedure**" to complete the relevant work.And here are some introduction may help to you.

[RaspberryPi](https://www.raspberrypi.com/) is a series of small single-board computers (SBCs) developed in the United Kingdom by the Raspberry Pi Foundation in association with Broadcom.  The Raspberry Pi project originally leaned toward the promotion of teaching basic computer science in schools.The original model became more popular than anticipated, selling outside its target market for diverse uses such as robotics, home and industrial automation, and by computer and electronic hobbyists, because of its low cost, modularity, open design, and its adoption of the HDMI and USB standards.

[FastASR](https://github.com/chenkui164/FastASR) is a C++ implementation of ASR reasoning project, it relies on few, installation is also very simple, reasoning speed is very fast, Raspberry PI 4B and other ARM platform can also run smoothly. The supported model is optimized from Google's Transformer model, and the data set is open source wenetspeech(10,000 + hours) or Ali's private data set (60,000 + hours), so the recognition effect is also very good, comparable to many commercial ASR software.

## Procedure

### Stage1

1. Compile FastASR with Source code on your computer.

2. Select a model and test.

### Stage2

1. Building a Cross-compilation chain on Raspberry PI 4B.

2. Compile fftw3 and OpenBLAS based on the cross-compile chain you built.

3. Compile FastASR with Source code on your Cross-compilation chain.

**Notes:** When you build a Cross-compilation chain , you need to make the compiler or optimizer of the Cross-compilation chain correspond to the Raspberry PI CPU model.For the Raspberry PI 4B version uses the bcm2711 architecture, where the processor is [ARM_Cortex-A72(ARM v8)](https://en.wikipedia.org/wiki/ARM_Cortex-A72) architecture

### Stage3 (optional)

1. Write a real-time speech-to-text conversion using a Raspberry PI microphone based on the FastASR you built.

## Submit

Any details and learning about the deployment process are worth putting in your folder.
