# 在 Ubuntu 上安装 TensorFlow

本指南将介绍如何在 Linux 的 Ubuntu 上安装 TensorFlow。虽然这些说明可能也适用于其他 Linux 版本，我们已经在满足以下条款的系统上进行了测试和支持：

* 64 位台式机或笔记本电脑
* Ubuntu 16.04 或更高版本

## 选择要安装的 TensorFlow

有以下的 TensorFlow 版本可供安装：

  * **仅有 CPU 支持的 TensorFlow**。如果你的系统中没有 NVIDIA®&nbsp;GPU，你必须安装这个版本。此版本的 TensorFlow 通常更易于安装，因此即使你拥有NVIDIA GPU，我们也建议你先安装此版本。
  * **含有 GPU 支持的 TensorFlow**。 TensorFlow 在 GPU 上运行通常要比在 CPU 上快。如果是在运行存在性能瓶颈应用而正好系统中有 NVIDIA GPU 符合要求，你应安装这个版本。详情请查看 [TensorFlow GPU 支持](#NVIDIARequirements)。
  
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

[Virtualenv](https://virtualenv.pypa.io/en/stable/) 工具是用于创建虚拟 Python 环境以与同台物理机中的其他 Python 隔离开。在这个情景中，会在虚拟环境中安装 TensorFlow 及其依赖并只在虚拟环境 **activated（激活）**时可用。Virtualenv 提供了一种避免与系统其他部分产生冲突来安装运行 TensorFlow 的可靠方式。

##### 1. 安装 Python, `pip`，以及 `virtualenv`。

在 Ubuntu 上，Python 是自动安装的而 `pip` **通常** 也已安装。
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

我们**推荐**使用 8.1 及以上版本的 `pip`。如果当前使用版本低于 8.1， 请升级 `pip`:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo pip install -U pip</code>
</pre>

如果不是 Ubuntu 系统但安装有 [setuptools](https://pypi.org/project/setuptools/) ，使用 `easy_install` 来安装 `pip`：

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
  <code class="devsite-terminal">source ~/tensorflow/<var>venv</var>/bin/activate      # bash，sh，ksh，或 zsh</code>
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

*   `tensorflow` —适用于 CPU 的当前发布版本
*   `tensorflow-gpu` —支持 GPU 的当前发布版本
*   `tf-nightly` —适用于 CPU 的最新构建版本
*   `tf-nightly-gpu` —支持 GPU 的最新构建版本

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

If the above steps failed, try installing the TensorFlow binary using the remote URL of the `pip` package:

<pre class="prettyprint lang-bsh">
(venv)$ pip install --upgrade <var>remote-pkg-URL</var>   # Python 2.7
(venv)$ pip3 install --upgrade <var>remote-pkg-URL</var>  # Python 3.n
</pre>

The <var>remote-pkg-URL</var> depends on the operating system, Python version, and GPU support. See [here](#the_url_of_the_tensorflow_python_package) for the URL naming scheme and location.

See [Common Installation Problems](#common_installation_problems) if you encounter problems.

#### Uninstall TensorFlow

To uninstall TensorFlow, remove the Virtualenv directory you created in step 2:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">deactivate  # stop the virtualenv</code>
  <code class="devsite-terminal">rm -r ~/tensorflow/<var>venv</var></code>
</pre>

<a name="InstallingNativePip"></a>

### Use `pip` in your system environment

Use `pip` to install the TensorFlow package directly on your system without using a container or virtual environment for isolation. This method is recommended for system administrators that want a TensorFlow installation that is available to everyone on a multi-user system.

Since a system install is not isolated, it could interfere with other Python-based installations. But if you understand `pip` and your Python environment, a system `pip` install is straightforward.

See the [REQUIRED_PACKAGES section of setup.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py) for a list of packages that TensorFlow installs.

##### 1. Install Python, `pip`, and `virtualenv`.

On Ubuntu, Python is automatically installed and `pip` is *usually* installed.
Confirm the `python` and `pip` versions:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">python -V  # or: python3 -V</code>
  <code class="devsite-terminal">pip -V     # or: pip3 -V</code>
</pre>

To install these packages on Ubuntu:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo apt-get install python-pip python-dev   # for Python 2.7</code>
  <code class="devsite-terminal">sudo apt-get install python3-pip python3-dev # for Python 3.n</code>
</pre>

We *recommend* using `pip` version 8.1 or higher. If using a release before version 8.1, upgrade `pip`:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo pip install -U pip</code>
</pre>

If not using Ubuntu and [setuptools](https://pypi.org/project/setuptools/) is installed, use `easy_install` to install `pip`:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">easy_install -U pip</code>
</pre>

##### 2. Install TensorFlow on system.

Choose one of the available TensorFlow packages for installation:

*   `tensorflow` —Current release for CPU
*   `tensorflow-gpu` —Current release with GPU support
*   `tf-nightly` —Nightly build for CPU
*   `tf-nightly-gpu` —Nightly build with GPU support

And use `pip` to install the package for Python 2 or 3:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo pip install -U tensorflow   # Python 2.7</code>
  <code class="devsite-terminal">sudo pip3 install -U tensorflow  # Python 3.n</code>
</pre>

Use `pip list` to show the packages installed on the system.
[Validate the install](#ValidateYourInstallation) and test the version:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">python -c "import tensorflow as tf; print(tf.__version__)"</code>
</pre>

Success: TensorFlow is now installed.

#### Problems

If the above steps failed, try installing the TensorFlow binary using the remote URL of the `pip` package:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo pip install --upgrade <var>remote-pkg-URL</var>   # Python 2.7</code>
  <code class="devsite-terminal">sudo pip3 install --upgrade <var>remote-pkg-URL</var>  # Python 3.n</code>
</pre>

The <var>remote-pkg-URL</var> depends on the operating system, Python version, and GPU support. See [here](#the_url_of_the_tensorflow_python_package) for the URL naming scheme and location.

See [Common Installation Problems](#common_installation_problems) if you encounter problems.

#### Uninstall TensorFlow

To uninstall TensorFlow on your system, use one of following commands:

<pre class="prettyprint lang-bsh">
  <code class="devsite-terminal">sudo pip uninstall tensorflow   # for Python 2.7</code>
  <code class="devsite-terminal">sudo pip3 uninstall tensorflow  # for Python 3.n</code>
</pre>

<a name="InstallingDocker"></a>

### Configure a Docker container

Docker completely isolates the TensorFlow installation from pre-existing packages on your machine. The Docker container contains TensorFlow and all its dependencies. Note that the Docker image can be quite large (hundreds of MBs). You might choose the Docker installation if you are incorporating TensorFlow into a larger application architecture that already uses Docker.

Take the following steps to install TensorFlow through Docker:

1.  Install Docker on your machine as described in the [Docker documentation](http://docs.docker.com/engine/installation/).
2.  Optionally, create a Linux group called <code>docker</code> to allow launching containers without sudo as described in the [Docker documentation](https://docs.docker.com/engine/installation/linux/linux-postinstall/). (If you don't do this step, you'll have to use sudo each time you invoke Docker.)
3.  To install a version of TensorFlow that supports GPUs, you must first install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), which is stored in github.
4.  Launch a Docker container that contains one of the [TensorFlow binary images](https://hub.docker.com/r/tensorflow/tensorflow/tags/).

The remainder of this section explains how to launch a Docker container.

#### CPU-only

To launch a Docker container with CPU-only support (that is, without GPU support), enter a command of the following format:

<pre>
$ docker run -it <i>-p hostPort:containerPort TensorFlowCPUImage</i>
</pre>

where:

*   <tt><i>-p hostPort:containerPort</i></tt> is optional. If you plan to run TensorFlow programs from the shell, omit this option. If you plan to run TensorFlow programs as Jupyter notebooks, set both <tt><i>hostPort</i></tt> and <tt><i>containerPort</i></tt> to <tt>8888</tt>. If you'd like to run TensorBoard inside the container, add a second `-p` flag, setting both <i>hostPort</i> and <i>containerPort</i> to 6006.
*   <tt><i>TensorFlowCPUImage</i></tt> is required. It identifies the Docker container. Specify one of the following values:

    *   <tt>tensorflow/tensorflow</tt>, which is the TensorFlow CPU binary image.
    *   <tt>tensorflow/tensorflow:latest-devel</tt>, which is the latest TensorFlow CPU Binary image plus source code.
    *   <tt>tensorflow/tensorflow:<i>version</i></tt>, which is the specified version (for example, 1.1.0rc1) of TensorFlow CPU binary image.
    *   <tt>tensorflow/tensorflow:<i>version</i>-devel</tt>, which is the specified version (for example, 1.1.0rc1) of the TensorFlow GPU binary image plus source code.

    TensorFlow images are available at [dockerhub](https://hub.docker.com/r/tensorflow/tensorflow/).

For example, the following command launches the latest TensorFlow CPU binary image in a Docker container from which you can run TensorFlow programs in a shell:

<pre>
$ <b>docker run -it tensorflow/tensorflow bash</b>
</pre>

The following command also launches the latest TensorFlow CPU binary image in a Docker container. However, in this Docker container, you can run TensorFlow programs in a Jupyter notebook:

<pre>
$ <b>docker run -it -p 8888:8888 tensorflow/tensorflow</b>
</pre>

Docker will download the TensorFlow binary image the first time you launch it.

#### GPU support

To launch a Docker container with NVidia GPU support, enter a command of the following format (this [does not require any local CUDA installation](https://github.com/nvidia/nvidia-docker/wiki/CUDA#requirements)):

<pre>
$ <b>nvidia-docker run -it</b> <i>-p hostPort:containerPort TensorFlowGPUImage</i>
</pre>

where:

*   <tt><i>-p hostPort:containerPort</i></tt> is optional. If you plan to run TensorFlow programs from the shell, omit this option. If you plan to run TensorFlow programs as Jupyter notebooks, set both <tt><i>hostPort</i></tt> and <code><em>containerPort</em></code> to `8888`.
*   <i>TensorFlowGPUImage</i> specifies the Docker container. You must specify one of the following values:
    *   <tt>tensorflow/tensorflow:latest-gpu</tt>, which is the latest TensorFlow GPU binary image.
    *   <tt>tensorflow/tensorflow:latest-devel-gpu</tt>, which is the latest TensorFlow GPU Binary image plus source code.
    *   <tt>tensorflow/tensorflow:<i>version</i>-gpu</tt>, which is the specified version (for example, 0.12.1) of the TensorFlow GPU binary image.
    *   <tt>tensorflow/tensorflow:<i>version</i>-devel-gpu</tt>, which is the specified version (for example, 0.12.1) of the TensorFlow GPU binary image plus source code.

We recommend installing one of the `latest` versions. For example, the following command launches the latest TensorFlow GPU binary image in a Docker container from which you can run TensorFlow programs in a shell:

<pre>
$ <b>nvidia-docker run -it tensorflow/tensorflow:latest-gpu bash</b>
</pre>

The following command also launches the latest TensorFlow GPU binary image in a Docker container. In this Docker container, you can run TensorFlow programs in a Jupyter notebook:

<pre>
$ <b>nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu</b>
</pre>

The following command installs an older TensorFlow version (0.12.1):

<pre>
$ <b>nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:0.12.1-gpu</b>
</pre>

Docker will download the TensorFlow binary image the first time you launch it. For more details see the [TensorFlow docker readme](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker).

#### Next Steps

You should now [validate your installation](#ValidateYourInstallation).

<a name="InstallingAnaconda"></a>

### Use `pip` in Anaconda

Anaconda provides the `conda` utility to create a virtual environment. However, within Anaconda, we recommend installing TensorFlow using the `pip install` command and *not* with the `conda install` command.

Caution: `conda` is a community supported package this is not officially maintained by the TensorFlow team. Use this package at your own risk since it is not tested on new TensorFlow releases.

Take the following steps to install TensorFlow in an Anaconda environment:

按照如下步骤在 Anaconda 环境中按照 TensorFlow：

1. 按照 [Anaconda 下载网站](https://www.continuum.io/downloads) 中的指导来下载并安装 Anaconda。
 
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

## TensorFlow GPU support

Note: Due to the number of libraries required, using [Docker](#InstallingDocker) is recommended over installing directly on the host system.

The following NVIDIA® <i>hardware</i> must be installed on your system:

*   GPU card with CUDA Compute Capability 3.5 or higher. See [NVIDIA documentation](https://developer.nvidia.com/cuda-gpus) for a list of supported GPU cards. The following NVIDIA® <i>software</i> must be installed on your system:
*   [GPU drivers](http://nvidia.com/driver). CUDA 9.0 requires 384.x or higher.
*   [CUDA Toolkit 9.0](http://nvidia.com/cuda).
*   [cuDNN SDK](http://developer.nvidia.com/cudnn) (>= 7.0). Version 7.1 is recommended.
*   [CUPTI](http://docs.nvidia.com/cuda/cupti/) ships with the CUDA Toolkit, but you also need to append its path to the `LD_LIBRARY_PATH` environment variable: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64`
*   *OPTIONAL*: [NCCL 2.2](https://developer.nvidia.com/nccl) to use TensorFlow with multiple GPUs.
*   *OPTIONAL*: [TensorRT](http://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html) which can improve latency and throughput for inference for some models.

To use a GPU with CUDA Compute Capability 3.0, or different versions of the preceding NVIDIA libraries see [通过源码安装 TensorFlow](./install_sources.md). If using Ubuntu 16.04 and possibly other Debian based linux distros, `apt-get` can be used with the NVIDIA repository to simplify installation.

```bash
# Adds NVIDIA package repository.
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update
# Includes optional NCCL 2.x.
sudo apt-get install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
  cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.1.4.18-1+cuda9.0 \
   libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0
# Optionally install TensorRT runtime, must be done after above cuda install.
sudo apt-get update
sudo apt-get install libnvinfer4=4.1.2-1+cuda9.0
```

## 常见安装错误

我们依赖于 Stack Overflow 来编写 TensorFlow 的安装问题和它们的解决方案。下面的表格包含了 Stack Overflow 关于常见安装问题的回答。如果你遇见了其他的错误信息或者没有在表格中列出的安装问题，请在 Stack Overflow 上搜索。如果 Stack Overflow 中没有显示相关的错误信息，创建一个新的问题并加上 `tensorflow` 标签。

<table>
<tr> <th>Link to GitHub or Stack&nbsp;Overflow</th> <th>Error Message</th> </tr>

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
