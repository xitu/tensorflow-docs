# 在 Ubuntu 上安装 TensorFlow

本指南将介绍如何在 Linux 的 Ubuntu 上安装 TensorFlow。虽然这些说明可能也适用于其他 Linux 版本，我们已经在满足以下条款的系统上进行了测试和支持：

* 64 位台式机或笔记本电脑
* Ubuntu 16.04 或更高版本

## 选择要安装的 TensorFlow

有以下的 TensorFlow 版本可供安装：

  * **仅有 CPU 支持的 TensorFlow**。如果你的系统中没有 NVIDIA®&nbsp;GPU，你必须安装这个版本。此版本的 TensorFlow 通常更易于安装，因此即使你拥有NVIDIA GPU，我们也建议你先安装此版本。
  * **含有 GPU 支持的 TensorFlow**。TensorFlow 在 GPU 上运行通常要比在 CPU 上快。如果是在运行存在性能瓶颈应用而正好系统中有 NVIDIA GPU 符合要求，你应安装这个版本。详情请查看 [TensorFlow GPU 支持](#NVIDIARequirements)。

## 如何安装 TensorFlow

安装 TensorFlow 有如下几个可选方式：

*   [在虚拟环境中使用 pip 安装](#InstallingVirtualenv) **(推荐)**
*   [在系统环境中使用 pip 安装](#InstallingNativePip)
*   [配置一个 Docker 容器](#InstallingDocker)
*   [在 Anacoda 下使用 pip](#InstallingAnaconda)
*   [由源码安装 TensorFlow](/install/install_sources)

<a name="InstallingVirtualenv"></a>

### 在虚拟环境中使用 `pip`

要点：使用虚拟环境是推荐安装方式。

[Virtualenv](https://virtualenv.pypa.io/en/stable/) 工具是用于创建虚拟 Python 环境以便与同台物理机中的其他 Python 隔离开。在这个情景中，会在虚拟环境中安装 TensorFlow 及其依赖并只在虚拟环境 **activated（激活）**时可用。Virtualenv 提供了一种避免与系统其他部分产生冲突来安装运行 TensorFlow 的可靠方式。

##### 1. 安装 Python, `pip`，以及 `virtualenv`。

在 Ubuntu 上，Python 是自动安装的并且 `pip` **通常** 也已安装。
确认 `python` 和 `pip` 版本：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">python -V  # or: python3 -V</code>
  <code class="devsite-terminal">pip -V     # or: pip3 -V</code>
</pre>

安装以下包到 Ubuntu 中：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo apt-get install python-pip python-dev python-virtualenv   # for Python 2.7</code>
  <code class="devsite-terminal">sudo apt-get install python3-pip python3-dev python-virtualenv # for Python 3.n</code>
</pre>

我们**推荐**使用 8.1 及以上版本的 `pip`。如果当前使用版本低于 8.1，请升级 `pip`：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo pip install -U pip</code>
</pre>

如果不是 Ubuntu 系统但安装有 [setuptools](https://pypi.org/project/setuptools/)，使用 `easy_install` 来安装 `pip`：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">easy_install -U pip</code>
</pre>

##### 2. 为虚拟环境创建一个目录并选择一个 Python 解释器。

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">mkdir ~/tensorflow  # 工作目录</code>
  <code class="devsite-terminal">cd ~/tensorflow</code>
  <code># Choose one of the following Python environments for the ./venv directory:</code>
  <code class="devsite-terminal">virtualenv --system-site-packages <var>venv</var>            # 使用默认 Python (Python 2.7)</code>
  <code class="devsite-terminal">virtualenv --system-site-packages -p python3 <var>venv</var> # 使用 Python 3.n</code>
</pre>

##### 3. 激活虚拟环境。

按照以下不同 shell 使用特定命令激活虚拟环境：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">source ~/tensorflow/<var>venv</var>/bin/activate      # bash、sh、ksh 或 zsh</code>
  <code class="devsite-terminal">source ~/tensorflow/<var>venv</var>/bin/activate.csh  # csh 或 tcsh</code>
  <code class="devsite-terminal">. ~/tensorflow/<var>venv</var>/bin/activate.fish      # fish</code>
</pre>

当虚拟环境激活后，shell 提示会显示为 `(venv) $`。

##### 4. 在虚拟环境中升级 `pip`。

在激活虚拟环境后，升级 `pip`：

<pre class="prettyprint lang-bsh">
(venv)$ pip install -U pip
</pre>

在虚拟环境中安装其他 Python 包并不会影响 `vurtualenv` 外的包。

##### 5. 在虚拟环境中安装 TensorFlow。

选择一个可用的 TensorFlow 包进行安装：

*   `tensorflow` — 适用于 CPU 的当前发布版本
*   `tensorflow-gpu` — 支持 GPU 的当前发布版本
*   `tf-nightly` — 适用于 CPU 的最新构建版本
*   `tf-nightly-gpu` — 支持 GPU 的最新构建版本

在激活虚拟环境后，使用 `pip` 安装包：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">pip install -U tensorflow</code>
</pre>

使用 `pip list` 查看虚拟环境中所有已安装包。
[验证安装](#ValidateYourInstallation)并查看版本：

<pre class="prettyprint lang-bsh">
(venv)$ python -c "import tensorflow as tf; print(tf.__version__)"
</pre>

成功：TensorFlow 现已安装完成。

使用 `deactivate` 命令停用 Python 虚拟环境。

#### 问题

如果以上步骤失败，可以尝试通过使用远程 `pip` 包的 URL 安装 TensorFlow 已编译文件：

<pre class="prettyprint lang-bsh">
(venv)$ pip install --upgrade <var>remote-pkg-URL</var>   # Python 2.7
(venv)$ pip3 install --upgrade <var>remote-pkg-URL</var>  # Python 3.n
</pre>

<var>remote-pkg-URL</var> 取决于操作系统、Python 版本以及是否有 GPU 支持。从[这里](#the_url_of_the_tensorflow_python_package)查看 URL 名称列表获取文件地址。

如果遇到问题，请查看[一般安装问题](#common_installation_problems)。

#### 卸载 TensorFlow

如要卸载 TensorFlow，移除步骤 2 中你创建的 Virtualenv 目录即可：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">deactivate  # stop the virtualenv</code>
  <code class="devsite-terminal">rm -r ~/tensorflow/<var>venv</var></code>
</pre>

<a name="InstallingNativePip"></a>

### 在系统环境下使用 `pip` 安装

使用 `pip` 在系统环境下安装 TensorFlow 包而不使用容器或虚拟环境隔离开。这种方式推荐在多系统环境中系统管理员想为其他用户提供可用的 TensorFlow 时使用。 

由于系统环境下安装，它可能会与其他以 Python 为基础的安装环境产生冲突。但如果你理解 `pip` 以及你的 Python 环境，系统环境下 `pip` 安装时最便捷的方式。
.
查看 [setup.py 必备包](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)了解 TensorFlow 安装必需的包。

##### 1. 安装 Python、`pip` 和 `virtualenv`。

在 Ubuntu 上，Python 是自动安装的而 `pip` **通常** 也已安装。
确认 `python` 和 `pip` 版本：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">python -V  # or: python3 -V</code>
  <code class="devsite-terminal">pip -V     # or: pip3 -V</code>
</pre>

安装以下包到 Ubuntu 中：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo apt-get install python-pip python-dev   # for Python 2.7</code>
  <code class="devsite-terminal">sudo apt-get install python3-pip python3-dev # for Python 3.n</code>
</pre>

我们**推荐**使用 8.1 及以上版本的 `pip`。如果当前使用版本低于 8.1，请升级 `pip`：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo pip install -U pip</code>
</pre>

如果不是 Ubuntu 系统但安装有 [setuptools](https://pypi.org/project/setuptools/)，使用 `easy_install` 来安装 `pip`：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">easy_install -U pip</code>
</pre>

##### 2. 在系统上安装 TensorFlow。

选择一个可用的 TensorFlow 包进行安装：

*   `tensorflow` — 适用于 CPU 的当前发布版本
*   `tensorflow-gpu` — 支持 GPU 的当前发布版本
*   `tf-nightly` — 适用于 CPU 的最新构建版本
*   `tf-nightly-gpu` — 支持 GPU 的最新构建版本

然后使用 `pip` 为 Python 2 或 3 安装以下包：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo pip install -U tensorflow   # Python 2.7</code>
  <code class="devsite-terminal">sudo pip3 install -U tensorflow  # Python 3.n</code>
</pre>

使用 `pip list` 查看虚拟环境中所有已安装包。
[验证安装](#ValidateYourInstallation)并查看版本：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">python -c "import tensorflow as tf; print(tf.__version__)"</code>
</pre>

成功：TensorFlow 现已安装完成。

#### 问题

如果以上步骤失败，可以尝试通过使用远程 `pip` 包的 URL 安装 TensorFlow 已编译文件：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo pip install --upgrade <var>remote-pkg-URL</var>   # Python 2.7</code>
  <code class="devsite-terminal">sudo pip3 install --upgrade <var>remote-pkg-URL</var>  # Python 3.n</code>
</pre>

由于系统环境下安装，它可能会与其他以 Python 为基础的安装环境产生冲突。但如果你理解 `pip` 以及你的 Python 环境，系统环境下 `pip` 安装时最便捷的方式。
.
查看 [setup.py 必备包](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)了解 TensorFlow 安装必需的包。

#### 卸载 TensorFlow

使用下面其中一条命令，将 TensorFlow 从你的系统中卸载：

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo pip uninstall tensorflow   # for Python 2.7</code>
  <code class="devsite-terminal">sudo pip3 uninstall tensorflow  # for Python 3.n</code>
</pre>

<a name="InstallingDocker"></a>

### 配置一个 Docker 容器

Docker 能将 TensorFlow 安装环境与宿主机中的已有包完全隔离开。这个 Docker 容器包含 TensorFlow 及其所有依赖。需要注意的是这个 Docker 镜像可能会相当大（数百 MB）。如果在使用 Docker 时需要将 TensorFlow 并入一个更大的应用架构中你可以选择使用 Docker 安装。

使用以下步骤来使用 Docker 安装 TensorFlow：

1.  按照 [Docker 文档](http://docs.docker.com/engine/installation/)的描述在你的机器上安装 Docker。
2.  可选项，按照 [Docker 文档](https://docs.docker.com/engine/installation/linux/linux-postinstall/)的描述创建一个 <code>docker</code> 用户组以便不使用 sudo 操作容器。（如果省略此步骤，你每次调用 Docker 命令时都需要使用 sudo。）
3.  要想安装能够支持 GPU 的 TensorFlow 版本，需要先安装 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)，这个存储在 github。
4.  启动一个包含以下 [已编译TensorFlow 镜像](https://hub.docker.com/r/tensorflow/tensorflow/tags/)的镜像。

本节的余下部分将讲解如何启动一个 Docker 容器。

#### CPU-only

要启动一个仅支持 CPU（表示没有 GPU 支持）的 Docker 容器。输入以下格式的一个命令：

<pre>
$ docker run -it <i>-p hostPort:containerPort TensorFlowCPUImage</i>
</pre>

详细解释：

*   <tt><i>-p 宿主机端口（hostPort）:容器端口（containerPort）</i></tt>可选项，如果你准备以 Jupyter notebook 的方式运行 TensorFlow，要将<tt><i>宿主机端口</i></tt>和<tt><i>容器端口</i></tt>都设为 <tt>8888</tt>。如果你想在容器中运行 TensorBoard，添加第二个 `-p` 参数，将<tt><i>宿主机端口</i></tt>和<tt><i>容器端口</i></tt>都设为 <tt>6066</tt>。
*   <tt><i>TensorFlowCPUImage</i></tt> 必须项。决定 Docker 容器。指定为以下选项之一：

    *   <tt>tensorflow/tensorflow</tt>，已编译 TensorFlow 支持 CPU 的镜像。
    *   <tt>tensorflow/tensorflow:latest-devel</tt>，最新的支持 CPU 的已编译 TensorFlow 镜像和源码。
    *   <tt>tensorflow/tensorflow:<i>version</i></tt>，特定版本（例如，1.1.0rc1）的支持 CPU 的已编译 TensorFlow 镜像。
    *   <tt>tensorflow/tensorflow:<i>version</i>-devel</tt>，特定版本（例如，1.1.0rc1）的支持 CPU 的已编译 TensorFlow 镜像和源码。

    TensorFlow 镜像可在 [dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/) 找到。

例如，下面的命令将最新的支持 CPU 的已编译 TensorFlow 镜像启动在一个能够在命令行中运行 TensorFlow 程序的 Docker 容器中：

<pre>
$ <b>docker run -it tensorflow/tensorflow bash</b>
</pre>

下面的命令也将最新的支持 CPU 的已编译 TensorFlow 镜像启动在 Docker 容器中。只不过，在这个 Docker 容器中，你可以在一个 Jupyter notebook 中运行 TensorFlow 程序：

<pre>
$ <b>docker run -it -p 8888:8888 tensorflow/tensorflow</b>
</pre>

Docker 会在你第一次启动这个容器时下载已编译 TensorFlow 镜像。

#### GPU 支持

想要启动一个带有 NVIDIA GPU 支持的 Docker 容器，输入以下格式的命令（这[不需要任何本地 CUDA 安装环境支持](https://github.com/nvidia/nvidia-docker/wiki/CUDA#requirements)）：

<pre>
$ <b>nvidia-docker run -it</b> <i>-p hostPort:containerPort TensorFlowGPUImage</i>
</pre>

详细解释:

*   <tt><i>-p 宿主机端口（hostPort）:容器端口（containerPort）</i></tt>可选项，如果你准备以 Jupyter notebook 的方式运行 TensorFlow，要将<tt><i>宿主机端口</i></tt>和<tt><i>容器端口</i></tt>都设为 <tt>8888</tt>。
*   <i>TensorFlowGPUImage</i> 决定 Docker 容器。指定为以下选项之一：
    *   <tt>tensorflow/tensorflow:latest-gpu</tt>，最新的支持 GPU 的已编译 TensorFlow 镜像。
    *   <tt>tensorflow/tensorflow:latest-devel-gpu</tt>，最新的支持 GPU 的已编译 TensorFlow 镜像和源码。
    *   <tt>tensorflow/tensorflow:<i>version</i>-gpu</tt>，特定版本（例如，0.12.1）的支持 GPU 的已编译 TensorFlow 镜像。
    *   <tt>tensorflow/tensorflow:<i>version</i>-devel-gpu</tt>，特定版本（例如，0.12.1）的支持 GPU 的已编译 TensorFlow 镜像和源码。

我们推荐安装 `latest` 版本。例如，下面的命令将最新的支持 GPU 的已编译 TensorFlow 镜像启动在一个能够在命令行中运行 TensorFlow 程序的 Docker 容器中：

<pre>
$ <b>nvidia-docker run -it tensorflow/tensorflow:latest-gpu bash</b>
</pre>

下面的命令也将最新的支持 GPU 的已编译 TensorFlow 镜像启动在 Docker 容器中。只不过，在这个 Docker 容器中，你可以在一个 Jupyter notebook 中运行 TensorFlow 程序：

<pre>
$ <b>nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu</b>
</pre>

下面的命令会安装更早版本的 TensorFlow（0.12.1）：

<pre>
$ <b>nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:0.12.1-gpu</b>
</pre>

Docker 会在你第一次启动容器时下载已编译 TensorFlow 镜像。更多细节请查看 [TensorFlow docker 自述](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker)。

#### 下一步

现在你需要[验证安装](#ValidateYourInstallation)。

<a name="InstallingAnaconda"></a>

### 在 Anaconda 中使用 `pip` 

Anaconda 提供了 `conda` 模块来创建一个虚拟环境。但是，在 Anaconda 中，我们推荐使用 `pip install` 而**不**是 `conda install` 命令来安装 TensorFlow。

注意：`conda` 是社区支持的包，它不是由 TensorFlow 团队官方维护的。由于它没有在最新的 TensorFlow 发布版本上测试，使用此包有一定未知风险。

按照如下步骤在 Anaconda 环境中安装 TensorFlow：

1. 按照 [Anaconda 下载网站](https://www.continuum.io/downloads) 中的指导下载并安装 Anaconda。
 
2. 通过以下命令建立一个叫做 <tt>tensorflow</tt> 的 conda 环境来运行某一版本的 Python:

     <pre>$ <b>conda create -n tensorflow pip python=2.7 # or python=3.3, etc.</b></pre>

3. 使用如下命令来激活 conda 环境：

     <pre>$ <b>source activate tensorflow</b>
     (tensorflow)$  # 这时你的前缀应该变成这样 </pre>

4. 运行如下格式的命令来在你的 conda 环境中安装 TensorFlow：

     <pre>(tensorflow)$ <b>pip install --ignore-installed --upgrade</b> <i>tfBinaryURL</i></pre>

   其中 <code><em>tfBinaryURL</em></code> 是 [TensorFlow Python 包的 URL](#the_url_of_the_tensorflow_python_package)。例如，如下命令安装了仅支持 CPU 的 Python 3.4 版本下的 TensorFlow：

     <pre>
     (tensorflow)$ <b>pip install --ignore-installed --upgrade \
     https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp34-cp34m-linux_x86_64.whl</b> </pre>

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

如果系统输出了一个错误信息，见[常见安装错误](#常见安装错误)。

如果你是机器学习的新手，我们推荐以下内容：

学习更多内容，请到 [TensorFlow 教程](../tutorials/)。

<a name="NVIDIARequirements"></a>

## TensorFlow GPU 支持

注意：由于需要安装大量的必备库，推荐使用 [Docker](#InstallingDocker) 安装而不是在宿主系统中直接安装。

以下 NVIDIA <i>硬件</i>必须安装在你的系统中：

*   GPU 卡拥有 CUDA Compute Capability 3.5 或以上。查看 [NVIDIA 文档](https://developer.nvidia.com/cuda-gpus)获取受支持 GPU 卡列表。

以下 NVIDIA <i>软件</i>必须安装在你的系统中：
*   [GPU drivers](http://nvidia.com/driver)。CUDA 9.0 需要 384.x 或更高。
*   [CUDA Toolkit 9.0](http://nvidia.com/cuda)。
*   [cuDNN SDK](http://developer.nvidia.com/cudnn) (>= 7.0)。推荐使用 7.1 版本 is recommended。
*   [CUPTI](http://docs.nvidia.com/cuda/cupti/) 随 CUDA Toolkit 一起，但你需要将它的路径添加到系统环境变量 `LD_LIBRARY_PATH` 中：`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64`。
*   **可选**：[NCCL 2.2](https://developer.nvidia.com/nccl) 在多块 GPU 中使用 TensorFlow。
*   **可选**：[TensorRT](http://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html) 理论上可以为一些模型降低延时和吞吐。

要使用带有 CUDA Compute Capability 3.0 的 GPU，或其他较早版本的 NVIDIA 库，请查看[通过源码安装 TensorFlow](./install_sources.md)。如果使用 Ubuntu 16.04 或其他基于 Debian 的发行版 Linux，可通过 NVIDIA 安装包仓库使用 `apt-get` 简便地安装。

```bash
# 添加 NVIDIA 安装包仓库。
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update
# 安装可选包 NCCL 2.x.
sudo apt-get install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
  cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.1.4.18-1+cuda9.0 \
   libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0
# 可选性安装 TensorRT 运行时环境，但必须在上面 cuda 安装之后进行。
sudo apt-get update
sudo apt-get install libnvinfer4=4.1.2-1+cuda9.0
```

## 常见安装错误

我们依赖于 Stack Overflow 来编写 TensorFlow 的安装问题和它们的解决方案。下面的表格包含了 Stack Overflow 关于常见安装问题的回答。如果你遇见了其他的错误信息或者没有在表格中列出的安装问题，请在 Stack Overflow 上搜索。如果 Stack Overflow 中没有显示相关的错误信息，创建一个新的问题并加上 `tensorflow` 标签。

<table>
<tr> <th>GitHub 或 Stack&nbsp;Overflow 链接</th> <th>报错信息</th> </tr>

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
  <a href="#Protobuf31">这里</a></td>
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

这个部分记录了 Linux 相关安装的 URL 值。

### Python 2.7

仅支持 CPU:

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp27-none-linux_x86_64.whl
</pre>

支持 GPU:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp27-none-linux_x86_64.whl
</pre>

注意 GPU 支持需要符合 [NVIDIA 对运行 GPU 支持版本的 TensorFlow 的要求](#NVIDIARequirements) 的软硬件要求。

### Python 3.4

仅支持 CPU：

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp34-cp34m-linux_x86_64.whl
</pre>

支持 GPU:

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp34-cp34m-linux_x86_64.whl
</pre>

注意 GPU 支持需要符合 [NVIDIA 对运行 GPU 支持版本的 TensorFlow 的要求](#NVIDIARequirements) 的软硬件要求。

### Python 3.5

支持 CPU：

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp35-cp35m-linux_x86_64.whl
</pre>

GPU 支持：

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp35-cp35m-linux_x86_64.whl
</pre>

注意 GPU 支持需要符合 [NVIDIA 对运行 GPU 支持版本的 TensorFlow 的要求](#NVIDIARequirements) 的软硬件要求。

### Python 3.6

仅支持 CPU：

<pre>
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl
</pre>

GPU 支持：

<pre>
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp36-cp36m-linux_x86_64.whl
</pre>

注意 GPU 支持需要符合 [NVIDIA 对运行支持 GPU 版本的 TensorFlow 的条件](#NVIDIARequirements)的软硬件要求。
