# GPU 支持

TensorFlow GPU 支持需要一系列的驱动和库。为了简化安装和避免依赖库的冲突，我们推荐使用[支持 GPU 的 TensorFlow Docker 镜像](./docker.md)（仅限 Linux 操作系统），此设置仅需安装 [NVIDIA®GPU 驱动程序](https://www.nvidia.com/drivers){:.external}。


## 硬件需求

TensorFlow 支持以下具有 GPU 的设备：

- CUDA 计算能力大于等于 3.5 的 NVIDIA 的 GPU，CUDA 计算能力可以参考[支持 CUDA 的 GPU](https://developer.nvidia.com/cuda-gpus){:.external}。


## 软件需求

您需要在系统上安装下列的 NVIDIA 软件：

- [NVIDIA® GPU 驱动](https://www.nvidia.com/drivers){:.external} —CUDA 9.0 需要 384.x 或更高的版本
- [CUDA® Toolkit](https://developer.nvidia.com/cuda-zone){:.external} —TensorFlow 支持 CUDA 9.0
- [CUPTI](http://docs.nvidia.com/cuda/cupti/){:.external} 随 CUDA® Toolkit 附带
- [cuDNN SDK](https://developer.nvidia.com/cudnn){:.external} 版本大于等于 7.2
- （可选项）[NCCL 2.2](https://developer.nvidia.com/nccl){:.external} 支持多块 GPU
- （可选项）[TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html){:.external} 改善一些模型推理时的延迟和吞吐量


## Linux 安装教程

下面的 `apt` 说明是在 Ubuntu 操作系统上安装所需 NVIDIA 软件的最简单方法。如果你是从[源码](./source.md)构建 Tensorflow，你需要手动安装之前提到的软件依赖，并且使用一个 `-devel` 的 [TensorFlow Docker 镜像](./docker.md)作为基础。

安装随 CUDA® Toolkit 附带的 [CUPTI](http://docs.nvidia.com/cuda/cupti/){:.external}，并通过下列命令将它的安装目录添加到 `$LD_LIBRARY_PATH` 环境变量中：

<pre class="devsite-click-to-copy">
<code class="devsite-terminal">export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64</code>
</pre>
对于具有 CUDA Compute Capability 3.0 的 GPU 或不同版本的 NVIDIA 库，请参阅 [Linux 源代码构建指南](./source.md)。

### 使用 apt 安装 CUDA 

对于使用 Ubuntu 16.04 或者其他的基于 Debian 的 Linux 发行版，添加 NVIDIA 的仓库并且使用 `apt` 命令来安装 CUDA。

注意：通过 `apt` 方式会将 NVIDIA 库和头文件安装到难以配置和 debug 的位置。

<pre class="prettyprint lang-bsh">

# 添加 NVIDIA 仓库
<code class="devsite-terminal">sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub</code>
<code class="devsite-terminal">wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb</code>
<code class="devsite-terminal">sudo apt install ./cuda-repo-ubuntu1604_9.1.85-1_amd64.deb</code>
<code class="devsite-terminal">wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb</code>
<code class="devsite-terminal">sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb</code>
<code class="devsite-terminal">sudo apt update</code>

# 安装 CUDA 和相关工具，包括可选项 NCCL 2.x
<code class="devsite-terminal">sudo apt install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \ 
    cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.2.1.38-1+cuda9.0 \ 
    libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0 </code>

# 可选项: 安装 TensorRT runtime (必须在 CUDA 安装之后进行)
<code class="devsite-terminal">sudo apt update</code>
<code class="devsite-terminal">sudo apt install libnvinfer4=4.1.2-1+cuda9.0</code>
</pre>

## Windows 安装教程

请查看上面的[硬件需求](#硬件需求)以及[软件需求](#软件需求)，并阅读 [Windows 操作系统的 CUDA 的安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/){:.external}。

请确保安装了上面列出的对应版本的 NVIDIA 软件包，特别是，如果缺少了 `cuDNN64_7.dll` 这个文件，TensorFlow 将无法正确加载。要使用不同版本的软件包，请参阅 [Windows 源码构建指南](./source_windows.md)。

将 CUDA、CUPTI 以及 cuDNN 的安装目录添加到 `%PATH%` 环境变量中。例如，如果 CUDA Toolkit 和 cuDNN 分别安装在 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0` 和 `C:\tools\cuda` 目录下，记得更新你的 `%PATH%` 以和下面的设置相符合：

<pre class="devsite-click-to-copy">
<code class="devsite-terminal tfo-terminal-windows">SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;%PATH%</code>
<code class="devsite-terminal tfo-terminal-windows">SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64;%PATH%</code>
<code class="devsite-terminal tfo-terminal-windows">SET PATH=C:\tools\cuda\bin;%PATH%</code>
</pre>
