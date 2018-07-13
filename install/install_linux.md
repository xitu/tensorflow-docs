# 在 Ubuntu 上安装 TensorFlow

本指南将介绍如何在 Ubuntu 上安装 TensorFlow。虽然这些说明可能也适用于其他 Linux 版本，但我们只在满足以下要求的计算机上验证过这些说明（而且我们只支持在此类计算机上按这些说明操作）：

  * 64 位台式机或笔记本电脑
  * Ubuntu 16.04 或更高版本

## 安装哪一个 TensorFlow

你必须在下列几种 TensorFlow 中选择其一来安装：

  * **仅有 CPU 支持的 TensorFlow**。 如果你的系统中没有 NVIDIA® GPU，你必须安装这个版本。 需要注意的是，这个版本的 TensorFlow 通常要更易于安装（往往仅需 5 至 10 分钟），所以即使你有英伟达（NVIDIA）的 GPU 显卡，我们仍然推荐你首先尝试安装这一版本。  
  * **含有 GPU 支持的 TensorFlow**。 TensorFlow 在 GPU上的运行速度要远大于相同程序在 CPU 上的运行速度。因此，如果你的系统中有符合以下要求的 NVIDIA® GPU 显卡，并且你的应用对性能有着严格的要求，你应该安装这一版本。

<a name="英伟达要求标准"></a>
### NVIDIA 对于使用 GPU 运行 TensorFlow 的要求

如果你正在利用本指南中描述的方法之一来安装支持 GPU 的 TensorFlow，那么你的系统中必须要有如下的 NVIDIA 软件：

  * [CUDA Toolkit 9.0](http://nvidia.com/cuda)。详见 [NVIDIA 的文档](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/)。请保证你将 CUDA 相关的路径像 NVIDIA 文档中所描述的那样添加在 `LD_LIBRARY_PATH` 环境变量中。
  * [cuDNN SDK v7](http://developer.nvidia.com/cudnn). 详见 [NVIDIA 的文档](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/)，请保证你像 NVIDIA 文档中描述的那样创建了 `CUDA_HOME` 环境变量。
  * GPU 显卡拥有 CUDA 3.0 或者更高版本的计算性能，用于构建源码以及 3.5 或者更高版本的二进制文件。详见 [NVIDIA 英伟达的文档](https://developer.nvidia.com/cuda-gpus) 中支持的 GPU 显卡列表。
  * [GPU 驱动](http://nvidia.com/driver) 支持你的 CUDA Toolkit 版本。
  * NVIDIA CUDA 解析工具的接口，libcupti-dev 库。该库提供了更高级的分析工具支持。要安装这个库，对 CUDA Toolkit 9.0 以上的版本运行如下命令即可：

    <pre>
    $ <b>sudo apt-get install cuda-command-line-tools</b>
    </pre>

    并且将其路径加在你的环境变量 `LD_LIBRARY_PATH` 中：

	<pre>
    $ <b>export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64</b>
    </pre>

    对于 CUDA Toolkit 7.5及以下版本，运行：

    <pre>
    $ <b>sudo apt-get install libcupti-dev</b>
    </pre>

  * **[可选]：** 为了优化推论性能，你也可以安装 **NVIDIA TensorRT 3.0**。The minimal set of TensorRT runtime components needed for use with the pre-built `tensorflow-gpu` package can be installed as follows:

   <pre>
   $ <b>wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1404/x86_64/nvinfer-runtime-trt-repo-ubuntu1404-3.0.4-ga-cuda9.0_1.0-1_amd64.deb</b>
   $ <b>sudo dpkg -i nvinfer-runtime-trt-repo-ubuntu1404-3.0.4-ga-cuda9.0_1.0-1_amd64.deb</b>
   $ <b>sudo apt-get update</b>
   $ <b>sudo apt-get install -y --allow-downgrades libnvinfer-dev libcudnn7-dev=7.0.5.15-1+cuda9.0 libcudnn7=7.0.5.15-1+cuda9.0</b>
   </pre>

  * **[重要]：** 为了与预构建包 `tensorflow-gpu` 进行兼容，please use the Ubuntu **14.04** package of TensorRT as shown above, even when installing onto an Ubuntu 16.04 system.<br/><br/>
  To build the TensorFlow-TensorRT integration module from source rather than using pre-built binaries, see the [module documentation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tensorrt#using-tensorrt-in-tensorflow).
  For detailed TensorRT installation instructions, see [NVIDIA's TensorRT documentation](http://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html).<br/><br/>
  To avoid cuDNN version conflicts during later system upgrades, you can hold the cuDNN version at 7.0.5:

   <pre>
   $ <b> sudo apt-mark hold libcudnn7 libcudnn7-dev</b>
   </pre>

   To later allow upgrades, you can remove the hold:

   <pre>
   $ <b> sudo apt-mark unhold libcudnn7 libcudnn7-dev</b>
   </pre>

如果您已安装前述软件包的旧版本，请升级到指定版本。如果升级失败，那么你可以使用 @{$install_sources$install TensorFlow from Sources}，此时你仍然可以运行支持 GPU 的 TensorFlow。

## 决定如何安装 TensorFlow

你必须决定使用哪一种方法来安装 TensorFlow。支持的方法有如下几种：

  * [Virtualenv](#InstallingVirtualenv)
  * ["native" pip](#InstallingNativePip)
  * [Docker](#InstallingDocker)
  * [Anaconda](#InstallingAnaconda)
  * 使用文档中的资源安装[另一个帮助文档](https://www.tensorflow.org/install/install_sources)。

**我们推荐使用 Virtualenv 安装方法** [Virtualenv](https://virtualenv.pypa.io/en/stable/) 是一个 Python 的虚拟环境，独立于其他的 Python 部署，不会与同一台机器上的其他 Python 程序互相影响。在安装 Virtualenv 的过程中，你需要安装 TensorFlow 及其依赖的所有包（实际上这很简单）。要开始使用 TensorFlow 工作的时候，你只需要激活("activate")虚拟环境。总而言之，Virtualenv 提供了一种安全可靠的方法来安装并运行 TensorFlow。

使用原生 pip 直接在你的系统上安装 TensorFlow 而不使用任何容器系统。**对于希望使每一个用户都能够使用 TensorFlow 的多用户系统的管理员，我们推荐使用原生 pip 直接安装**。由于原生 pip 安装不是在一个独立容器中的进行的隔离安装，因此，pip 可能会影响到同台机器上其他基于 Python 的程序。然而如果你了解 pip 和你当前的 Python 环境，原生 pip 安装会更加简单，往往只需要一条命令即可。

Docker 完全地将 TensorFlow 的安装与其他之前安装于你机器上的库隔离开。Docker 容器中包含 TensorFlow 和其他所有的依赖包。请注意 Docker 镜像可能会比较大（几百 MB 大小）。若你已经在一个很大的应用项目中使用了 Docker，你应该也用它来安装你的 TensorFlow。

在 Anaconda 中，你可以使用 conda 来创建一个虚拟环境。然而，我们推荐你使用 `pip install` 命令在 Anaconda 中安装 TensorFlow，而不是 `conda install`。

**注意：** conda 中的包是社区而非官方支持的。也就是说，TensorFlow 的团队既不负责测试也不负责维护 conda 中的包。这可能给你的使用带来隐患，需要你自行负责。


<a name="InstallingVirtualenv"></a>
## 使用 Virtualenv 安装

按照如下步骤来使用 virtualenv 安装 TensorFlow：

  1. 选择下面的一条命令来安装 pip 和 Virtualenv：

     <pre>
     $ <b>sudo apt-get install python-pip python-dev python-virtualenv</b> # for Python 2.7
     $ <b>sudo apt-get install python3-pip python3-dev python-virtualenv</b> # for Python 3.n </pre>

  2. 挑选其中的一条命令来创建一个 Virtualenv 的虚拟环境:

     <pre>
     $ <b>virtualenv --system-site-packages</b> <i>targetDirectory</i> # for Python 2.7
     $ <b>virtualenv --system-site-packages -p python3</b> <i>targetDirectory</i> # for Python 3.n </pre>

     其中 <code><em>targetDirectory</em></code> 指明了 Virtualenv 树中根部位置。我们的命令中假设了 <code><em>targetDirectory</em></code> 是 `~/tensorflow`，但你也可以指定任意目录。

  3. 通过以下任意一条命令激活 Virtualenv 的虚拟环境:

     <pre>
     $ <b>source ~/tensorflow/bin/activate</b> # bash, sh, ksh, or zsh 
     $ <b>source ~/tensorflow/bin/activate.csh</b>  # csh or tcsh
     $ <b>. ~/tensorflow/bin/activate.fish</b>  # fish </pre>

     第一条 <tt>source</tt> 命令会将你的前缀变为

     <pre>(tensorflow)$ </pre>

  4. 确保安装了 pip 8.1 或更高版本：
  
     <pre>(tensorflow)$ <b>easy_install -U pip</b></pre>

  5. 运行下列其中的一条命令来在激活的 Virtualenv 环境中安装 TensorFlow:
	
     <pre>(tensorflow)$ <b>pip install --upgrade tensorflow</b>      # for Python 2.7
     (tensorflow)$ <b>pip3 install --upgrade tensorflow</b>     # for Python 3.n
     (tensorflow)$ <b>pip install --upgrade tensorflow-gpu</b>  # for Python 2.7 and GPU
     (tensorflow)$ <b>pip3 install --upgrade tensorflow-gpu</b> # for Python 3.n and GPU </pre>

     如果前面几步成功了，则可以跳过步骤 6，否则需要继续执行步骤 6。

  6. （可选）如果步骤 5 失败了（通常是由于你运行了一个低于 8.1 的 pip 版本），通过以下命令来在激活的 Virtualenv 环境中安装 TensorFlow：
  
     <pre>
     (tensorflow)$ <b>pip install --upgrade</b> <i>tfBinaryURL</i>   # Python 2.7
     (tensorflow)$ <b>pip3 install --upgrade</b> <i>tfBinaryURL</i>  # Python 3.n </pre>

     其中 <code><em>tfBinaryURL</em></code> 指明了 TensorFlow 的 Python 包的 URL 路径。 <code><em>tfBinaryURL</em></code> 的值取决于操作系统，Python 版本和 GPU 支持。在这里找到时候你的系统的 <code><em>tfBinaryURL</em></code> [值](#the_url_of_the_tensorflow_python_package)。例如，如果你要在 Linux 中安装 Python 3.4 和仅支持 CPU 环境的 TensorFlow，在激活的 virtualenv 环境中运行如下命令即可：
     
     <pre>
     (tensorflow)$ <b>pip3 install --upgrade \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp34-cp34m-linux_x86_64.whl</b> </pre>

如果你遇见了安装问题，请见：[常见安装问题](#common_installation_problems)。

### 下一步

在安装了 TensorFlow 之后，需要[验证你的安装](#ValidateYourInstallation)。

请注意你必须在每次运行 TensorFlow 之前都要激活你的 Virtualenv 环境。如果 Virtualenv 环境当前并没有激活，运行以下其中一条命令：

<pre>
$ <b>source ~/tensorflow/bin/activate</b>      # bash, sh, ksh, or zsh
$ <b>source ~/tensorflow/bin/activate.csh</b>  # csh or tcsh </pre>

当 Virtualenv 环境激活后，你可以使用 shell 来运行相关程序。出现如下提示时，代表着那你的虚拟环境已经激活了：

<pre>(tensorflow)$ </pre>

当你使用完 TensorFlow 之后，你可以通过 `deactivate` 命令来休眠该环境:

<pre>(tensorflow)$ <b>deactivate</b> </pre>

前缀提示会变回原来默认的样式（由 `PS1` 环境变量定义）。

### 卸载 TensorFlow

要卸载 TensorFlow，只需要简单地移除你所创建的整个目录树
例如：

<pre>$ <b>rm -r</b> <i>targetDirectory</i> </pre>

<a name="InstallingNativePip"></a>
## 使用原生 pip 安装

你也可以使用 pip 安装 TensorFlow，在简单的安装过程和复杂的安装过程之间进行选择。你可能会需要通过 pip 来安装 TensorFlow，在一个简单的安装过程和一个更复杂的中选择其一：

**注意：**[setup.py 中的 REQUIRED_PACKAGES 部分](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)
列出了 TensorFlow 安装时 pip 会安装或升级的所有的包

### 安装前提：Python 和 Pip

Python 是自动安装于 Ubuntu 中的。花一秒的时间来确认一下系统中安装的 Python 版本(命令 `python -V`)：

  * Python 2.7
  * Python 3.4+

pip 或 pip3 包管理**通常**会安装在 Ubuntu 系统上。通过 `pip -V` 或 `pip3 -V` 命令来确认下是否有安装。我们强烈建议安装 8.1 或更高版本的 pip 或 pip3。如果没有安装，运行如下命令来安装或更新到最新的 pip 版本：

<pre>$ <b>sudo apt-get install python-pip python-dev</b>   # for Python 2.7
$ <b>sudo apt-get install python3-pip python3-dev</b> # for Python 3.n
</pre>

### 安装 TensorFlow

假设之前所需的软件已经安装在了你的 Linux 主机上，那么进行如下几步：

  1. 通过其中的**一条命令**安装 TensorFlow：

     <pre>
     $ <b>pip install tensorflow</b>      # Python 2.7; CPU support (no GPU support) 
     $ <b>pip3 install tensorflow</b>     # Python 3.n; CPU support (no GPU support)
     $ <b>pip install tensorflow-gpu</b>  # Python 2.7;  GPU support
     $ <b>pip3 install tensorflow-gpu</b> # Python 3.n; GPU support </pre>

     如果命令完成了安装，你现在应该[对你的安装进行验证](#ValidateYourInstallation)。

  2. (可选) 如果步骤 1 失败了，安装如下格式执行命令进行安装:

     <pre>
     $ <b>sudo pip  install --upgrade</b> <i>tfBinaryURL</i>   # Python 2.7
     $ <b>sudo pip3 install --upgrade</b> <i>tfBinaryURL</i>   # Python 3.n </pre>

     其中 <code><em>tfBinaryURL</em></code> 指明了 TensorFlow 的 Python 包的 URL 路径。<code><em>tfBinaryURL</em></code> 的值取决于操作系统，Python 版本和 GPU 支持。在[这里](#the_url_of_the_tensorflow_python_package)找到时候你的系统的 <code><em>tfBinaryURL</em></code> 值。例如，如果你要在 Linux 中安装 Python 3.4 和仅支持 CPU 环境的 TensorFlow，在激活的 Virtualenv 环境中运行如下命令即可：

     <pre>
     (tensorflow)$ <b>pip3 install --upgrade \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp34-cp34m-linux_x86_64.whl</b> </pre>

     如果该步骤失败了，见这里：[常见安装问题](#common_installation_problems).

### 下一步

安装完毕 TensorFlow 之后，[验证你的安装](#ValidateYourInstallation).

### 卸载

要卸载 TensorFlow，运行如下命令：

<pre>
$ <b>sudo pip uninstall tensorflow</b>  # for Python 2.7
$ <b>sudo pip3 uninstall tensorflow</b> # for Python 3.n </pre>

<a name="InstallingDocker"></a>
## 使用 Docker 安装

通过以下几步来使用 Docker 安装 TensorFlow：

  1. 如 [Docker 文档](http://docs.docker.com/engine/installation/)中所描述的那样来安装 Docker。
  2. 或者，创建一个 Linux group 叫做 <code>docker</code>，如 [Docker 文档](https://docs.docker.com/engine/installation/linux/linux-postinstall/)中所描述的那样，这样无需 sudo 命令即可运行容器。（如果你不做这一步，你需要在每次使用 Docker 时都使用 sudo 命令。)
  3. 要安装支持 GPU 的 TensorFlow，你必须先安装位于 GitHub 中的[nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
  4. 运行包含[TensorFlow 二进制镜像](https://hub.docker.com/r/tensorflow/tensorflow/tags/)的 Docker。

剩下的部分解释了如何运行一个 Docker 容器。

### 仅 CPU 支持

要运行一个仅支持 CPU 的 Docker 容器（即不带 GPU 支持），运行如下格式的命令：

<pre>
$ docker run -it <i>-p hostPort:containerPort TensorFlowCPUImage</i>
</pre>

其中:

  * <tt><i>-p hostPort:containerPort</i></tt> 是可选的如果你准备从 shell 命令行中运行 TensorFlow 程序，那么忽略该选项。如果你准备在如 Jupyter notebooks 中运行 TensorFlow，把 <tt><i>hostPort</i></tt> 和 <tt><i>containerPort</i></tt> 都设置为 <tt>8888</tt>。如果你想在容器中运行 TensorBoard，加一个 `-p`，将<i>hostPort</i> 和 <i>containerPort</i> 都设置为 6006。
  * <tt><i>TensorFlowCPUImage</i></tt> 是必需的。它指定了 Docker。选择声明其中的一个值：
    * <tt>tensorflow/tensorflow</tt>，这是 TensorFlow CPU 二进制镜像的值。
    * <tt>tensorflow/tensorflow:latest-devel</tt>，这是最新的 TensorFlow CPU 二进制镜像加上源码，
    * <tt>tensorflow/tensorflow:<i>version</i></tt>，是某一特定的版本（比如，1.1.0rc1）的 TensorFlow CPU 二进制镜像。
    * <tt>tensorflow/tensorflow:<i>version</i>-devel</tt>，是某一特定的版本（比如，1.1.0rc1）的 TensorFlow CPU 二进制镜像加源码。

    <tt>gcr.io</tt> 是 Google 容器注册（Google Container Registry）。注意一些 TensorFlow 的镜像也可以在 [dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/) 中找到。

例如，如下命令在 Docker 容器中运行 TensorFlow CPU 二进制镜像，可以从 shell 命令行中运行 TensorFlow：

<pre>
$ <b>docker run -it tensorflow/tensorflow bash</b>
</pre>

如下命令也可以在 Docker 中运行最新的 TensorFlow CPU 二进制镜像。不同的是，在这个 Docker 镜像中，你可以在 Jupyter notebook 中运行 TensorFlow：

<pre>
$ <b>docker run -it -p 8888:8888 tensorflow/tensorflow</b>
</pre>

Docker 将会在你第一次运行的时候下载 TensorFlow 二进制镜像。

### GPU 支持

在安装 GPU 支持的 TensorFlow 之前，确保你的系统符合
[NVIDIA 软件要求](#NVIDIARequirements).  要运行一个带有 NVIDIA GPU 支持的 Docker 容器运行如下格式的命令：

<pre>
$ <b>nvidia-docker run -it</b> <i>-p hostPort:containerPort TensorFlowGPUImage</i>
</pre>

其中:

  * <tt><i>-p hostPort:containerPort</i></tt> 是可选的如果你准备从 shell 命令行中运行 TensorFlow 程序，那么忽略该选项。如果你准备在如 Jupyter notebooks 中运行 TensorFlow，把 <tt><i>hostPort</i></tt> 和 <tt><i>containerPort</i></tt> 都设置为 <tt>8888</tt>。如果你想在容器中运行 TensorBoard，加一个 `-p`，将<i>hostPort</i> 和 <i>containerPort</i> 都设置为 6006。
  * <tt><i>TensorFlowCPUImage</i></tt> 是必需的。 它指定了 Docker。 选择声明其中的一个值：
    * <tt>tensorflow/tensorflow</tt>， 这是 TensorFlow CPU 二进制镜像的值。
    * <tt>tensorflow/tensorflow:latest-devel</tt>，这是最新的 TensorFlow CPU 二进制镜像加上源码，
    * <tt>tensorflow/tensorflow:<i>version</i></tt>，是某一特定的版本（比如，1.1.0rc1）的 TensorFlow CPU 二进制镜像。
    * <tt>tensorflow/tensorflow:<i>version</i>-devel</tt>，是某一特定的版本（比如，1.1.0rc1）的 TensorFlow CPU 二进制镜像加源码。

    <tt>gcr.io</tt> 是 Google 容器注册（Google Container Registry）。注意一些 TensorFlow 的镜像也可以在
    [dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/) 中找到.

例如，如下命令在 Docker 容器中运行 TensorFlow CPU 二进制镜像，可以从 shell 命令行中运行 TensorFlow：

<pre>
$ <b>nvidia-docker run -it tensorflow/tensorflow:latest-gpu bash</b>
</pre>

如下命令也在 Docker 容器中运行了最新的 TensorFlow GPU 二进制镜像。在这个 Docker 容器中，你可以在 Jupyter notebook 中运行程序：

<pre>
$ <b>nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu</b>
</pre>

如下的命令可以安装一个较早的 TensorFlow 版本（0.12.1）：

<pre>
$ <b>nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:0.12.1-gpu</b>
</pre>

Docker 会在你第一次运行的时候下载 TensorFlow 二进制镜像。更多信息见 [TensorFlow docker readme](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker)。

### 下一步

你应该[验证你的安装](#ValidateYourInstallation).

<a name="InstallingAnaconda"></a>
## 使用 Anaconda 安装

按照如下步骤在 Anaconda 环境中按照 TensorFlow：

  1. 按照 [Anaconda 下载网站](https://www.continuum.io/downloads)中的指导来下载并安装 Anaconda。
 
  2. 通过以下命令建立一个叫做<tt>tensorflow</tt> 的 conda 环境来运行某一版本的 Python:

     <pre>$ <b>conda create -n tensorflow pip python=2.7 # or python=3.3, etc.</b></pre>

  3. 使用如下命令来激活 conda 环境：

     <pre>$ <b>source activate tensorflow</b>
     (tensorflow)$  # 这时你的前缀应该变成这样 </pre>

  4. 运行如下格式的命令来在你的 conda 环境中安装 TensorFlow：

     <pre>
     (tensorflow)$ <b>pip install --ignore-installed --upgrade</b> <i>tfBinaryURL</i> </pre>

     其中 <code><em>tfBinaryURL</em></code> 是 [TensorFlow Python 包的 URL](#the_url_of_the_tensorflow_python_package)。例如，如下命令安装了仅支持 CPU 的 Python 3.4 版本下的 TensorFlow：

     <pre>
     (tensorflow)$ <b>pip install --ignore-installed --upgrade \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp34-cp34m-linux_x86_64.whl</b> </pre>

<a name="ValidateYourInstallation"></a>
## 验证你的安装

按照如下步骤验证你的 TensorFlow 安装：

  1. 确保你的环境可以运行 TensorFlow （即：若有虚拟环境应激活它）
  2. 执行一个简短的 TensorFlow 程序

### 准备你的环境

如果你是使用原生 pip，Virtualenv 或者 Anaconda 安装的，那么进行如下步骤：

  1. 开启一个终端。
  2. 如果是使用 Virtualenv 或 Anaconda 安装，激活你的容器。 
  3. 如果使用的 TensorFlow 源码安装，跳转至任意路径，**除了**有 TensorFlow 源码的地方。

如果你是通过 Docker 安装的，开启一个你可以使用 bush 的 Docker 容器，如：

<pre>
$ <b>docker run -it tensorflow/tensorflow bash</b>
</pre>


### 执行一个简短的 TensorFlow 程序

在你的 shell 命令行中开启 Python：

<pre>$ <b>python</b></pre>

在 Python 的交互式 shell 命令行中运行如下程序：

``` python
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

如果系统输出一下数据，那么代表着你已经准备好编写 TensorFlow 程序了：

<pre>Hello, TensorFlow!</pre>

如果系统输出了一个错误信息，见[常见安装错误](#common_installation_problems).

如果你是机器学习的新手，我们推荐以下内容：

*  [机器学习速成课程](https://developers.google.com/machine-learning/crash-course)
*  @{$get_started/get_started_for_beginners$Getting Started for ML Beginners}

如果你有机器学习的经验，但刚接触 TensorFlow 请看 @{$get_started/premade_estimators$Getting Started with TensorFlow}。

## 常见安装错误

我们依赖于 Stack Overflow 来编写 TensorFlow 的安装问题和它们的解决方案。下面的表格包含了 Stack Overflow 关于常见安装问题的回答。如果你遇见了其他的错误信息或者没有在表格中列出的安装问题，请在 Stack Overflow 上搜索。如果 Stack Overflow 中没有显示相关的错误信息，创建一个新的问题并加上 `tensorflow` 标签。

<table>
<tr> <th>Stack Overflow 链接</th> <th>错误信息 Error Message</th> </tr>

<tr>
  <td><a href="https://stackoverflow.com/q/36159194">36159194</a></td>
  <td><pre>ImportError: libcudart.so.<i>Version</i>: cannot open shared object file:
  No such file or directory</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/41991101">41991101</a></td>
  <td><pre>ImportError: libcudnn.<i>Version</i>: cannot open shared object file:
  No such file or directory</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/36371137">36371137</a> and
  <a href="#Protobuf31">here</a></td>
  <td><pre>libprotobuf ERROR google/protobuf/src/google/protobuf/io/coded_stream.cc:207] A
  protocol message was rejected because it was too big (more than 67108864 bytes).
  To increase the limit (or to disable these warnings), see
  CodedInputStream::SetTotalBytesLimit() in google/protobuf/io/coded_stream.h.</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/35252888">35252888</a></td>
  <td><pre>Error importing tensorflow. Unless you are using bazel, you should
  not try to import tensorflow from its source directory; please exit the
  tensorflow source tree, and relaunch your python interpreter from
  there.</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/33623453">33623453</a></td>
  <td><pre>IOError: [Errno 2] No such file or directory:
  '/tmp/pip-o6Tpui-build/setup.py'</tt></pre>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><pre>ImportError: Traceback (most recent call last):
  File ".../tensorflow/core/framework/graph_pb2.py", line 6, in <module>
  from google.protobuf import descriptor as _descriptor
  ImportError: cannot import name 'descriptor'</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/questions/35190574">35190574</a> </td>
  <td><pre>SSLError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
  failed</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42009190">42009190</a></td>
  <td><pre>
  Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
  Found existing installation: setuptools 1.1.6
  Uninstalling setuptools-1.1.6:
  Exception:
  ...
  [Errno 1] Operation not permitted:
  '/tmp/pip-a1DXRT-uninstall/.../lib/python/_markerlib' </pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/questions/36933958">36933958</a></td>
  <td><pre>
  ...
  Installing collected packages: setuptools, protobuf, wheel, numpy, tensorflow
  Found existing installation: setuptools 1.1.6
  Uninstalling setuptools-1.1.6:
  Exception:
  ...
  [Errno 1] Operation not permitted:
  '/tmp/pip-a1DXRT-uninstall/System/Library/Frameworks/Python.framework/
   Versions/2.7/Extras/lib/python/_markerlib'</pre>
  </td>
</tr>

</table>


<a name="TF_PYTHON_URL"></a>
## TensorFlow Python 软件包的网址

一些安装方法中需要 TensorFlow Python 包的 URL，你所声明的值取决下面三个因素：

  * 操作系统
  * Python 版本
  * CPU 还是 GPU 支持

这个部分记录了 Linux 相关安装的 URL 值

### Python 2.7

仅支持 CPU:

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp27-none-linux_x86_64.whl
</pre>

支持 GPU:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp27-none-linux_x86_64.whl
</pre>

注意 GPU 支持需要符合[NVIDIA 对运行 GPU 支持版本的 TensorFlow 的要求](#NVIDIARequirements)的软硬件要求。

### Python 3.4

仅支持 CPU：

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp34-cp34m-linux_x86_64.whl
</pre>

支持 GPU:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp34-cp34m-linux_x86_64.whl
</pre>

注意 GPU 支持需要符合[NVIDIA 对运行 GPU 支持版本的 TensorFlow 的要求](#NVIDIARequirements)的软硬件要求。

### Python 3.5

支持 CPU：

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp35-cp35m-linux_x86_64.whl
</pre>

GPU 支持：

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp35-cp35m-linux_x86_64.whl
</pre>

注意 GPU 支持需要符合[NVIDIA 对运行 GPU 支持版本的 TensorFlow 的要求](#NVIDIARequirements)的软硬件要求。

### Python 3.6

仅支持 CPU：

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
</pre>

GPU 支持：

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
</pre>

注意 GPU 支持需要符合[NVIDIA 对运行 GPU 支持版本的 TensorFlow 的要求](#NVIDIARequirements)的软硬件要求。
