### Stage1

### 构建推理引擎

### 问题：

通过前期对Transformer、Tensor-RT的构建，发现有以下问题：

* 建立的环境(比如：cuda-toolkit、pytorch、cuda与其他包......)不匹配
* wsl下Ubuntu22.04的网络有问题（）,后来代理问题改为goole的DNS才好点
* 在运行官方给的库时，始终存在运行到一部分，有一部分官方给的代码我的环境无法运行，报错，修改的结果就是越改越错。后来不得已，将miniconda创建的环境删了，又重新开始，又会有新的错误，真的很糟心......
* 官方的教程可能不全，有部分包得等运行报错时，才晓得还差哪些包
* 下载报错有时候分不清是网络问题还是环境问题
* 有时下载包时，总会卡在某个地方，卡几个小时都不报错

## 目前主要是对vllm推理引擎构建

#### 环境配置：

VLLM：0.5.3

cuda:12.1

cuda--toolkit-12-6

wsl2-ubuntu22.04

GPU:nvidia-3050

模型：Qwen/Qwen2-1.5B-Instruct

#### 以下通过查阅文献比较 Transformer、VLLM、Tensor-RT、DeepSpeed 和 Text Generation Inference 等几种主流推理引擎：

| 推理引擎                  | 主要特点                         | 性能优势                   | 优化技术                     | 支持平台/语言               | 使用场景                 |
| ------------------------- | -------------------------------- | -------------------------- | ---------------------------- | --------------------------- | ------------------------ |
| Transformer               | 基于自注意力机制的序列到序列模型 | 并行计算能力强             | 多头注意力，位置编码         | 多平台，支持多种编程语言等  | 自然语言处理，机器翻译等 |
| VLLM                      | 针对大语言模型的低延迟推理       | 高吞吐量，低延迟           | 批量调度，动态解码           | GPU加速，Python等           | 大规模语言模型推理       |
| TensorRT                  | NVIDIA的深度学习优化框架         | 高效推理速度               | 动态形状支持，张量重编译     | NVIDIA GPU，CUDA等          | 深度学习模型的高性能推理 |
| DeepSpeed                 | 微软开发的深度学习优化工具       | 训练和推理加速             | 分布式训练优化，自动混合精度 | 多种硬件平台，Python等      | 大型模型训练与推理       |
| Text Generation Inference | 专为文本生成任务设计的优化工具   | 快速响应时间，高效内存管理 | 量化，批处理优化             | CPU/GPU，支持多种编程语言等 | 文本生成任务             |

#### 使用 Docker 进行部署（已完成）：

##### BUG:

```(moon) angel@angle:/mnt/c/Users/evil angle/vllm$ DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai
(moon) angel@angle:/mnt/c/Users/evil angle/vllm$ DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai
[+] Building 244.6s (44/50)                                                                              docker:default
 => [internal] load build definition from Dockerfile                                                               0.2s
 => => transferring dockerfile: 8.71kB                                                                             0.2s
 => WARN: FromAsCasing: 'as' and 'FROM' keywords' casing do not match (line 130)                                   0.2s
 => WARN: FromAsCasing: 'as' and 'FROM' keywords' casing do not match (line 139)                                   0.2s
 => [internal] load metadata for docker.io/nvidia/cuda:12.4.1-base-ubuntu20.04                                     4.8s
 => [internal] load metadata for docker.io/nvidia/cuda:12.4.1-devel-ubuntu20.04                                    5.1s
 => [internal] load .dockerignore                                                                                  0.1s
 => => transferring context: 50B                                                                                   0.1s
 => [internal] load build context                                                                                  3.4s
 => => transferring context: 58.53kB                                                                               3.3s
 => [vllm-base  1/10] FROM docker.io/nvidia/cuda:12.4.1-base-ubuntu20.04@sha256:6fdb33fd81a5e214cfff44685aa32e3ab  0.0s
 => CACHED [vllm-base  2/10] WORKDIR /vllm-workspace                                                               0.0s
 => CACHED [vllm-base  3/10] RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections     && echo '  0.0s
 => CACHED [vllm-base  4/10] RUN apt-get update -y     && apt-get install -y python3-pip git vim curl libibverbs-  0.0s
 => CACHED [vllm-base  5/10] RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10                        0.0s
 => CACHED [vllm-base  6/10] RUN python3 -m pip --version                                                          0.0s
 => CACHED [vllm-base  7/10] RUN ldconfig /usr/local/cuda-$(echo 12.4.1 | cut -d. -f1,2)/compat/                   0.0s
 => [base  1/13] FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu20.04@sha256:8d577fd078ae56c37493af4454a5b700c72a7  0.0s
 => CACHED [dev 3/3] COPY requirements-dev.txt requirements-dev.txt                                                0.0s
 => CACHED [base 11/13] COPY requirements-mamba.txt requirements-mamba.txt                                         0.0s
 => CACHED [base  4/13] RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10                             0.0s
 => CACHED [base  5/13] RUN python3 -m pip --version                                                               0.0s
 => CACHED [base  6/13] RUN ldconfig /usr/local/cuda-$(echo 12.4.1 | cut -d. -f1,2)/compat/                        0.0s
 => CACHED [base  7/13] WORKDIR /workspace                                                                         0.0s
 => CACHED [base  8/13] COPY requirements-common.txt requirements-common.txt                                       0.0s
 => CACHED [base  9/13] COPY requirements-cuda.txt requirements-cuda.txt                                           0.0s
 => CACHED [base 10/13] RUN --mount=type=cache,target=/root/.cache/pip     python3 -m pip install -r requirements  0.0s
 => CACHED [base  2/13] RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections     && echo 'tzdat  0.0s
 => CACHED [base 12/13] RUN python3 -m pip install packaging                                                       0.0s
 => CACHED [base 13/13] RUN python3 -m pip install -r requirements-mamba.txt                                       0.0s
 => CACHED [dev 1/3] COPY requirements-lint.txt requirements-lint.txt                                              0.0s
 => CACHED [base  3/13] RUN apt-get update -y     && apt-get install -y git curl sudo                              0.0s
 => CACHED [dev 2/3] COPY requirements-test.txt requirements-test.txt                                              0.0s
 => CACHED [mamba-builder 1/3] WORKDIR /usr/src/mamba                                                              0.0s
 => CACHED [mamba-builder 2/3] COPY requirements-mamba.txt requirements-mamba.txt                                  0.0s
 => CACHED [build  8/15] COPY requirements-common.txt requirements-common.txt                                      0.0s
 => CACHED [build  2/15] RUN --mount=type=cache,target=/root/.cache/pip     python3 -m pip install -r requirement  0.0s
 => CACHED [build  3/15] RUN apt-get update -y && apt-get install -y ccache                                        0.0s
 => CACHED [build  4/15] COPY csrc csrc                                                                            0.0s
 => CACHED [build  5/15] COPY setup.py setup.py                                                                    0.0s
 => CACHED [build  6/15] COPY cmake cmake                                                                          0.0s
 => CACHED [build  7/15] COPY CMakeLists.txt CMakeLists.txt                                                        0.0s
 => CACHED [build  1/15] COPY requirements-build.txt requirements-build.txt                                        0.0s
 => CACHED [build  9/15] COPY requirements-cuda.txt requirements-cuda.txt                                          0.0s
 => CACHED [build 10/15] COPY pyproject.toml pyproject.toml                                                        0.0s
 => ERROR [mamba-builder 3/3] RUN pip wheel -r requirements-mamba.txt                                            234.3s
 => [build 11/15] COPY vllm vllm                                                                                   0.4s
 => [build 12/15] RUN --mount=type=cache,target=/root/.cache/pip     if [ "$USE_SCCACHE" = "1" ]; then         ec  0.5s
 => CANCELED [build 13/15] RUN if [ "0" != "1" ]; then         python3 setup.py bdist_wheel --dist-dir=dist --p  234.8s
------
 > [mamba-builder 3/3] RUN pip wheel -r requirements-mamba.txt:
2.599 Collecting mamba-ssm>=1.2.2 (from -r requirements-mamba.txt (line 2))
2.600   Using cached mamba_ssm-2.2.2-cp310-cp310-linux_x86_64.whl
3.066 Collecting causal-conv1d>=1.2.0 (from -r requirements-mamba.txt (line 3))
3.066   Using cached causal_conv1d-1.4.0-cp310-cp310-linux_x86_64.whl
3.375 Collecting torch (from mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
3.919   Downloading torch-2.4.0-cp310-cp310-manylinux1_x86_64.whl.metadata (26 kB)
4.190 Collecting packaging (from mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
4.328   Downloading packaging-24.1-py3-none-any.whl.metadata (3.2 kB)
4.523 Collecting ninja (from mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
4.650   Downloading ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl.metadata (5.3 kB)
4.822 Collecting einops (from mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
4.826   Using cached einops-0.8.0-py3-none-any.whl.metadata (12 kB)
4.954 Collecting triton (from mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
5.103   Downloading triton-3.0.0-1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.3 kB)
5.275 Collecting transformers (from mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
5.409   Downloading transformers-4.43.3-py3-none-any.whl.metadata (43 kB)
5.790 Collecting filelock (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
5.913   Downloading filelock-3.15.4-py3-none-any.whl.metadata (2.9 kB)
6.060 Collecting typing-extensions>=4.8.0 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
6.185   Downloading typing_extensions-4.12.2-py3-none-any.whl.metadata (3.0 kB)
6.433 Collecting sympy (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
6.557   Downloading sympy-1.13.1-py3-none-any.whl.metadata (12 kB)
6.753 Collecting networkx (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
6.889   Downloading networkx-3.3-py3-none-any.whl.metadata (5.1 kB)
7.062 Collecting jinja2 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
7.212   Downloading jinja2-3.1.4-py3-none-any.whl.metadata (2.6 kB)
7.386 Collecting fsspec (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
7.519   Downloading fsspec-2024.6.1-py3-none-any.whl.metadata (11 kB)
7.680 Collecting nvidia-cuda-nvrtc-cu12==12.1.105 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
7.807   Downloading nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
7.967 Collecting nvidia-cuda-runtime-cu12==12.1.105 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
8.089   Downloading nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
8.237 Collecting nvidia-cuda-cupti-cu12==12.1.105 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
8.356   Downloading nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
8.483 Collecting nvidia-cudnn-cu12==9.1.0.70 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
8.606   Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
8.740 Collecting nvidia-cublas-cu12==12.1.3.1 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
8.875   Downloading nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
8.998 Collecting nvidia-cufft-cu12==11.0.2.54 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
9.116   Downloading nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
9.284 Collecting nvidia-curand-cu12==10.3.2.106 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
9.412   Downloading nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl.metadata (1.5 kB)
9.549 Collecting nvidia-cusolver-cu12==11.4.5.107 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
9.663   Downloading nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
9.822 Collecting nvidia-cusparse-cu12==12.1.0.106 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
9.945   Downloading nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl.metadata (1.6 kB)
10.08 Collecting nvidia-nccl-cu12==2.20.5 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
10.21   Downloading nvidia_nccl_cu12-2.20.5-py3-none-manylinux2014_x86_64.whl.metadata (1.8 kB)
10.35 Collecting nvidia-nvtx-cu12==12.1.105 (from torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
10.47   Downloading nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl.metadata (1.7 kB)
10.61 Collecting nvidia-nvjitlink-cu12 (from nvidia-cusolver-cu12==11.4.5.107->torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
10.73   Downloading nvidia_nvjitlink_cu12-12.6.20-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
10.89 Collecting huggingface-hub<1.0,>=0.23.2 (from transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
11.03   Downloading huggingface_hub-0.24.5-py3-none-any.whl.metadata (13 kB)
11.47 Collecting numpy>=1.17 (from transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
11.60   Downloading numpy-2.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (60 kB)
11.84 Collecting pyyaml>=5.1 (from transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
11.96   Downloading PyYAML-6.0.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)
12.46 Collecting regex!=2019.12.17 (from transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
12.59   Downloading regex-2024.7.24-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (40 kB)
12.79 Collecting requests (from transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
12.91   Downloading requests-2.32.3-py3-none-any.whl.metadata (4.6 kB)
13.16 Collecting safetensors>=0.4.1 (from transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
13.29   Downloading safetensors-0.4.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.8 kB)
13.59 Collecting tokenizers<0.20,>=0.19 (from transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
13.71   Downloading tokenizers-0.19.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
13.98 Collecting tqdm>=4.27 (from transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
14.12   Downloading tqdm-4.66.4-py3-none-any.whl.metadata (57 kB)
14.35 Collecting MarkupSafe>=2.0 (from jinja2->torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
14.49   Downloading MarkupSafe-2.1.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.0 kB)
14.70 Collecting charset-normalizer<4,>=2 (from requests->transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
14.83   Downloading charset_normalizer-3.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (33 kB)
14.99 Collecting idna<4,>=2.5 (from requests->transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
15.12   Downloading idna-3.7-py3-none-any.whl.metadata (9.9 kB)
15.30 Collecting urllib3<3,>=1.21.1 (from requests->transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
15.44   Downloading urllib3-2.2.2-py3-none-any.whl.metadata (6.4 kB)
15.58 Collecting certifi>=2017.4.17 (from requests->transformers->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
15.71   Downloading certifi-2024.7.4-py3-none-any.whl.metadata (2.2 kB)
15.88 Collecting mpmath<1.4,>=1.1.0 (from sympy->torch->mamba-ssm>=1.2.2->-r requirements-mamba.txt (line 2))
16.00   Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
16.02 Using cached einops-0.8.0-py3-none-any.whl (43 kB)
16.15 Downloading ninja-1.11.1.1-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl (307 kB)
16.46 Downloading packaging-24.1-py3-none-any.whl (53 kB)
16.63 Downloading torch-2.4.0-cp310-cp310-manylinux1_x86_64.whl (797.2 MB)
233.1    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━        658.2/797.2 MB 9.3 MB/s eta 0:00:15
233.2 ERROR: Exception:
233.2 Traceback (most recent call last):
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_vendor/urllib3/response.py", line 438, in _error_catcher
233.2     yield
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_vendor/urllib3/response.py", line 561, in read
233.2     data = self._fp_read(amt) if not fp_closed else b""
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_vendor/urllib3/response.py", line 527, in _fp_read
233.2     return self._fp.read(amt) if amt is not None else self._fp.read()
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_vendor/cachecontrol/filewrapper.py", line 98, in read
233.2     data: bytes = self.__fp.read(amt)
233.2   File "/usr/lib/python3.10/http/client.py", line 466, in read
233.2     s = self.fp.read(amt)
233.2   File "/usr/lib/python3.10/socket.py", line 705, in readinto
233.2     return self._sock.recv_into(b)
233.2   File "/usr/lib/python3.10/ssl.py", line 1307, in recv_into
233.2     return self.read(nbytes, buffer)
233.2   File "/usr/lib/python3.10/ssl.py", line 1163, in read
233.2     return self._sslobj.read(len, buffer)
233.2 TimeoutError: The read operation timed out
233.2
233.2 During handling of the above exception, another exception occurred:
233.2
233.2 Traceback (most recent call last):
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_internal/cli/base_command.py", line 105, in _run_wrapper
233.2     status = _inner_run()
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_internal/cli/base_command.py", line 96, in _inner_run
233.2     return self.run(options, args)
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_internal/cli/req_command.py", line 67, in wrapper
233.2     return func(self, options, args)
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_internal/commands/wheel.py", line 147, in run
233.2     requirement_set = resolver.resolve(reqs, check_supported_wheels=True)
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_internal/resolution/resolvelib/resolver.py", line 179, in resolve
233.2     self.factory.preparer.prepare_linked_requirements_more(reqs)
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_internal/operations/prepare.py", line 554, in prepare_linked_requirements_more
233.2     self._complete_partial_requirements(
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_internal/operations/prepare.py", line 469, in _complete_partial_requirements
233.2     for link, (filepath, _) in batch_download:
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_internal/network/download.py", line 184, in __call__
233.2     for chunk in chunks:
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_internal/cli/progress_bars.py", line 55, in _rich_progress_bar
233.2     for chunk in iterable:
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_internal/network/utils.py", line 65, in response_chunks
233.2     for chunk in response.raw.stream(
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_vendor/urllib3/response.py", line 622, in stream
233.2     data = self.read(amt=amt, decode_content=decode_content)
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_vendor/urllib3/response.py", line 560, in read
233.2     with self._error_catcher():
233.2   File "/usr/lib/python3.10/contextlib.py", line 153, in __exit__
233.2     self.gen.throw(typ, value, traceback)
233.2   File "/usr/local/lib/python3.10/dist-packages/pip/_vendor/urllib3/response.py", line 443, in _error_catcher
233.2     raise ReadTimeoutError(self._pool, None, "Read timed out.")
233.2 pip._vendor.urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.
------

 3 warnings found (use --debug to expand):
 - FromAsCasing: 'as' and 'FROM' keywords' casing do not match (line 130)
 - FromAsCasing: 'as' and 'FROM' keywords' casing do not match (line 139)
 - LegacyKeyValueFormat: "ENV key=value" should be used instead of legacy "ENV key value" format (line 225)
Dockerfile:149
--------------------
 147 |
 148 |     # Download the wheel or build it if a pre-compiled release doesn't exist
 149 | >>> RUN pip wheel -r requirements-mamba.txt
 150 |
 151 |     #################### MAMBA Build IMAGE ####################
--------------------
ERROR: failed to solve: process "/bin/sh -c pip wheel -r requirements-mamba.txt" did not complete successfully: exit code: 2
```



参考文献：

[字节跳动提出高性能 transformer 推理库，获 IPDPS 2023 最佳论文奖-CSDN博客](https://blog.csdn.net/ByteDanceTech/article/details/131238326)

[解锁 vLLM：大语言模型推理的速度与效率双提升-腾讯云开发者社区-腾讯云 (tencent.com)](https://cloud.tencent.com/developer/article/2351458)

[【综述论文】UC Berkeley：Transformer推理全栈优化研究进展综述 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/663879289)

[DeepSpeed-Inference: 大规模Transformer模型高效推理 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzU2ODU3Mzc4Nw==&mid=2247500735&idx=2&sn=bc6104b44547701bff8014aa22e39fc2)

[[2205.09579\] TRT-ViT: TensorRT-oriented Vision Transformer (arxiv.org)](https://arxiv.org/abs/2205.09579)

[Dynamic and Efficient Inference for Text Generation via BERT Family - ACL Anthology](https://aclanthology.org/2023.acl-long.162/)
