# 在 Raspbian 系统上安装 TensorFlow

这篇指南将讲解如何在安装 Raspbian 系统的树莓派（Raspberry Pi）上安装 TensorFlow。尽管这些方式可能也能在其他树莓派系统上成功安装，但我们只在以下规格的的机器上进行了测试（这是我们唯一支持）：

*   运行 Raspbian 9.0 或更高版本的树莓派设备。

## 决定如何安装 TensorFlow

你需要选择安装 TensorFlow 的方式。目前包括以下选项：

*   “原生” pip。
*   由源码交叉编译。

**我们推荐使用 pip 安装。**

## 使用原生 pip 安装

我们已将 TensorFlow 已编译文件上传到 piwheels.org。你可以通过 pip 安装 TensorFlow。

[setup.py 安装必需包](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py)罗列了所有使用 pip 安装或升级所需要的包。

### 必备项：Python

要想安装 TensorFlow，你的系统必须包含以下任一版本的 Python：

*   Python 2.7
*   Python 3.4+

如果你的系统之前没有安装任何版本的 Python。请先[安装](https://wiki.python.org/moin/BeginnersGuide/Download)它。它应当在 Raspbian 安装时已经包含在系统中。所以并不需要多余的步骤。 

### 必备项：pip

[pip](https://en.wikipedia.org/wiki/Pip_\(package_manager\)) 用于安装和管理 Python 包。如果你想用原生 pip 安装，那么以下一个版本的 pip 必须安装在你的系统中：

*   `pip3`，适用于 Python 3.n（推荐）。
*   `pip`，适用于 Python 2.7。

`pip` 或 `pip3` 很可能在你安装 Python 时已同时安装。检查系统中是否已安装 pip 或 pip3，使用以下命令之一检查：

<pre>$ <b>pip3 -V</b> # for Python 3.n
$ <b>pip -V</b>  # for Python 2.7</pre>

如果返回错误 "Command not found"。此包尚未安装。如要第一次安装它，请运行：

<pre>$ sudo apt-get install python3-pip # for Python 3.n
sudo apt-get install python-pip # for Python 2.7</pre>

如果在安装和升级 pip 的过程中需要更多帮助，请参阅 [Raspberry Pi 文档](https://www.raspberrypi.org/documentation/linux/software/python.md)

### 必备项：Atlas

[Atlas](http://math-atlas.sourceforge.net/) 是 numpy 依赖的一个线性代数库，所以需要在 TensorFlow 前安装。使用以下命令将其添加到系统中：

<pre>$ sudo apt install libatlas-base-dev</pre>

### 安装 TensorFlow

在你的树莓派上安装完这些必备软件后，调用以下其中**一**条命令安装 TensorFlow：

     <pre> $ <b>pip3 install tensorflow</b>     # Python 3.n
     $ <b>pip install tensorflow</b>      # Python 2.7</pre>

由于一些 TensorFlow 依赖的 Python 包（比如 scipy）需要在安装过程中编译，所以在某些平台（比如 树莓派 zero）上会花费些时间。因为 piwheels.org 已将这些依赖的包进行了预编译，所以 Python 3 版本通常安装速度较快，这也是我们推荐的方式。

### 下一步

在安装 TensorFlow 后，[验证安装](#ValidateYourInstallation)确认安装成功并运行正常。

### 卸载 TensorFlow

使用以下其中一条命令卸载 TensorFlow：

<pre>$ <b>pip uninstall tensorflow</b>
$ <b>pip3 uninstall tensorflow</b> </pre>

## 由源码交叉编译

交叉编译是指在与部署系统不同的系统环境下编译构建。由于树莓派只有有限的内存和相对较慢的处理器，而 TensorFlow 由非常多的源码需要编译，所以使用 MacOS 或 Linux 桌面环境或笔记本进行编译要快捷的多。因为在树莓派上构建将会超过 24 个小时，并且需要外部的交换空间处理内存存储。所以如果你需要从源码编译 TensorFlow 我们推荐使用交叉编译。为了简化依赖管理过程，我们同样推荐使用 Docker 来帮助简化构建过程。

注意：我们提供了经过测试的 Raspbian 系统下使用的预构建 TensorFlow 已编译包。所以除非你熟悉由源码构建复杂包并能处理文档中未能详细描述的各种问题，不要自行构建 TensorFlow 已编译包。

### 必备项：Docker

按照 [Docker 文档](https://docs.docker.com/engine/installation/#/on-macos-and-windows)描述在你的机器上安装 Docker。

### 克隆 TensorFlow 仓库

构建 TensorFlow 的第一步是克隆 TensorFlow 仓库。

通过以下命令克隆**最新**的 TensorFlow 仓库：

<pre>$ <b>git clone https://github.com/tensorflow/tensorflow</b> </pre>

这条 <code>git clone</code> 命令会创建一个 `tensorflow`的子目录。克隆完成后，你可以选择性地通过下面这条命令构建一个**特定分支**(比如 发布分支)：

<pre>
$ <b>cd tensorflow</b>
$ <b>git checkout</b> <i>Branch</i> # <i>Branch</i> 为你设定的分支
</pre>

例如，使用以下命令由 master 发布分支切换到 `r1.0` 发布分支：

<pre>$ <b>git checkout r1.0</b></pre>

### 由源码构建

To compile TensorFlow and produce a binary pip can install, do the following:
要编译 TensorFlow 并产生一个 pip 可以安装的二进制文件，按如下步骤操作：

1.  启动一个终端。
2.  进入存放 TensorFlow 源码的目录下。
3.  运行交叉编译此库的命令，例如：

<pre>$ CI_DOCKER_EXTRA_PARAMS="-e CI_BUILD_PYTHON=python3 -e CROSSTOOL_PYTHON_INCLUDE_PATH=/usr/include/python3.4" \
tensorflow/tools/ci_build/ci_build.sh PI-PYTHON3 tensorflow/tools/ci_build/pi/build_raspberry_pi.sh
 </pre>

这会构建一个用于 Python 3.4 的 pip .whl 文件，并使用 Arm v7 指令仅能运行在树莓派 2 或 3 上。NEON 指令集在这些设备上是加速操作所必须的，但是你可以通过在命令末尾添加 `PI_ONE` 参数来构建能运行在所有 Pi 设备上的库。同样可以通过省略其实的 docker 参数来设定目标环境为 Python 2.7。下面的样例可以构建运行在树莓派 Zero 或 1 上 Python 2.7 的包：

<pre>$ tensorflow/tools/ci_build/ci_build.sh PI tensorflow/tools/ci_build/pi/build_raspberry_pi.sh PI_ONE</pre>

这一过程会花费一段时间，通常是二十到三十分钟，并会最后会在源码目录下脚本生成的子目录中产生一个 .whl 文件。这个 wheel 文件可以使用 pip 或 pip3（取决于你的 Python版本）安装，将它拷贝到 树莓派上并在终端上运行如下命令（使用实际产生的文件名称）：

<pre>$ pip3 install tensorflow-1.9.0-cp34-none-linux_armv7l.whl</pre>

### 构建中可能会产生的问题

构建脚本使用 Docker 创建一个 Linux 虚拟机在内部处理构建过程。如果运行脚本出现问题，首先在系统上检查 Docker 是否可用（比如 `docker run hello-world` ）。

如果你由最新开发分支进行构建，请尝试同步到最近可用的版本，例如，release 1.9，使用如下命令：

<pre>$ <b>git checkout r1.0</b></pre>

<a name="ValidateYourInstallation"></a>

## 验证安装

按照以下步骤验证 TensorFlow 安装：

1.  确保系统环境能够运行 TensorFlow 程序。
2.  运行一个简短的 TensorFlow 程序。

### 准备系统环境

If you installed on native pip, Virtualenv, or Anaconda, then do the following:
如果你已安装原生 pip， Virtualenv 或 Anaconda，那么按照如下步骤：

1.  启动一个终端。
2.  如果你下载 TensorFlow 源码，切换到**除了** TensorFlow 源码目录的任一目录。

### 运行一个简短的 TensorFlow 程序

按如下方式在命令行中调用 Python：

<pre>$ <b>python</b></pre>

在 Python 交互命令行中输入如下简短程序：

```python
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

如果系统输出如下，那就可以开始编写 TensorFlow 程序了：

<pre>Hello, TensorFlow!</pre>

如果你使用 Python3.5，第一次导入 TensorFlow 可能会产生一条警告。这不是一个错误，TensorFlow 也能正常运行，可以忽视这个日志信息。

如果系统输出了报错信息，请查看 [一般安装问题](#common_installation_problems)。

想要学习更多，请查看 [TensorFlow 教程](../tutorials/)。

## 一般安装问题

我们在 Stack Overflow 记录 TensorFlow 安装问题和解决办法，下面的表格中罗列了一些安装过程中的一般问题在 Stack Overflow 上的回答。如果你遇到了下表中未出现的报错信息或安装错误，请在 Stack Overflow 上查找。如果 Stack Overflow 中没有此报错信息，请在 Stack Overflow 上提出一个关于它的新问题并将其特定地标记上 `tensorflow`。
<table>
<tr> <th>Stack Overflow 链接</th> <th>报错信息</th> </tr>


<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><pre>ImportError: Traceback (most recent call last):
File ".../tensorflow/core/framework/graph_pb2.py", line 6, in <module>
from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/33623453">33623453</a></td>
  <td><pre>IOError: [Errno 2] No such file or directory:
  '/tmp/pip-o6Tpui-build/setup.py'</tt></pre>
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
  <td><a href="https://stackoverflow.com/q/33622019">33622019</a></td>
  <td><pre>ImportError: No module named copyreg</pre></td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/37810228">37810228</a></td>
  <td>During a <tt>pip install</tt> operation, the system returns:
  <pre>OSError: [Errno 1] Operation not permitted</pre>
  </td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/33622842">33622842</a></td>
  <td>An <tt>import tensorflow</tt> statement triggers an error such as the
  following:<pre>Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib/python2.7/site-packages/tensorflow/__init__.py",
    line 4, in <module>
    from tensorflow.python import *
    ...
  File "/usr/local/lib/python2.7/site-packages/tensorflow/core/framework/tensor_shape_pb2.py",
    line 22, in <module>
    serialized_pb=_b('\n,tensorflow/core/framework/tensor_shape.proto\x12\ntensorflow\"d\n\x10TensorShapeProto\x12-\n\x03\x64im\x18\x02
      \x03(\x0b\x32
      .tensorflow.TensorShapeProto.Dim\x1a!\n\x03\x44im\x12\x0c\n\x04size\x18\x01
      \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\tb\x06proto3')
  TypeError: __init__() got an unexpected keyword argument 'syntax'</pre>
  </td>
</tr>


</table>
