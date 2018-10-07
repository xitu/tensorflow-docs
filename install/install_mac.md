# 在 macOS 上安装 TensorFlow

本指南介绍如何在 macOS 上安装 TensorFlow。虽然这些说明可能也适用于其他 macOS 版本，但我们只在满足以下要求的计算机上对这些说明中的内容进行过测试（并提供相关支持）：

  * macOS 10.12.6 (Sierra) 或更高

注意: 在 [GitHub #15933](https://github.com/tensorflow/tensorflow/issues/15933#issuecomment-366331383) 中已经描述了在 macOS 10.12.6 (Sierra) 之前已知会影响数值准确性的问题。

## 选择安装 TensorFlow 的方式

你必须选择使用哪种方式来安装 TensorFlow。支持的方式有如下几种：

  * Virtualenv
  * 原生 pip
  * Docker
  * 通过源码进行安装，详见[这篇文档](https://www.tensorflow.org/install/install_sources)。

**我们推荐使用 Virtualenv 进行安装。**[Virtualenv](https://virtualenv.pypa.io/en/stable) 是与其他 Python 开发隔离的虚拟 Python 环境，使得在同一台机器上不受其他 Python 程序的干扰。在 Virtualenv 安装过程中，你不仅要安装 TensorFlow，还需安装 TensorFlow 所需的软件包（其实很简单）。要开始使用 TensorFlow，只需要「激活」虚拟环境。总而言之，Virtualenv 为安装和运行 TensorFlow 提供了一个安全可靠的机制。

使用本地的 pip 会在系统里直接安装 TensorFlow，无需任何容器或虚拟环境系统。然而由于本地安装并不是完全封闭的，因此本地安装可能会受到系统上其他基于 Python 安装软件的干扰，或者影响到这类软件。此外，你可能还需要禁用系统完整性保护（SIP）才能进行本地安装。但是，如果你了解 SIP、pip 和你本地的 Python 环境，那么使用本地 pip 安装会相对容易一些。

[Docker](http://docker.com) 则会将 TensorFlow 安装与机器上的现有软件包完全隔离。Docker 容器包含 TensorFlow 及其所有依赖项。请注意，Docker 镜像可能非常大（几百 MB）。如果你将 TensorFlow 集成到已经使用 Docker 的较大应用程序体系结构中，则可以选择 Docker 安装。

在 Anaconda 中，你可以使用 conda 来创建虚拟环境。但是，在 Anaconda 中，我们建议使用 `pip install` 而不是 `conda install` 命令来安装 TensorFlow。

**注意：** conda 包由社区提供支持，而非正式支持。也就是说，TensorFlow 团队既不测试也不维护 conda 包。使用此包请自行承担相关风险。

## 通过 Virtualenv 安装

通过执行下面的步骤来使用 Virtualenv 安装 TensorFlow：

  1. 启动终端。你需要在命令行中执行下面所有的步骤。

  2. 通过下面的命令安装 pip 和 Virtualenv：

     <pre>
     $ <b>sudo easy_install pip</b>
     $ <b>pip install --upgrade virtualenv</b> </pre>

  3. 通过执行下面的命令来创建 Virtualenv 环境：

     <pre>
     $ <b>virtualenv --system-site-packages</b> <i>targetDirectory</i> # 对应 Python 2.7
     $ <b>virtualenv --system-site-packages -p python3</b> <i>targetDirectory</i> # 对应 Python 3.n
     </pre>

     其中 <i>targetDirectory</i> 表示 Virtualenv 目录树所在的顶层路径。我们假设 <i>targetDirectory</i> 为 `~/tensorflow`，但你也可以选择任何你喜欢的路径。

  4. 通过执行下面的命令来激活 Virtualenv 环境：

     <pre>
     $ <b>cd <i>targetDirectory</i></b>
     $ <b>source ./bin/activate</b> # 如果是使用 bash、sh、ksh、或 zsh
     $ <b>source ./bin/activate.csh</b> # 如果是使用 csh 或 tcsh </pre>

     前面的 `source` 命令会将你的命令行提示更改为以下内容：

     <pre>(<i>targetDirectory</i>)$</pre>

  5. 确保安装的 pip 版本大于或等于 8.1：

     <pre>(<i>targetDirectory</i>)$ <b>easy_install -U pip</b></pre>

  6. 执行下面的命令会将 TensorFlow 及其全部依赖安装至 Virtualenv 环境中：

     <pre>
     (<i>targetDirectory</i>)$ <b>pip install --upgrade tensorflow</b>      # 对应 Python 2.7
     (<i>targetDirectory</i>)$ <b>pip3 install --upgrade tensorflow</b>     # 对应 Python 3.n </pre>

  7. （可选）如果第 6 步失败了（通常可能是因为你使用的 pip 版本小于 8.1），你还可以在激活的 Virtualenv 环境下，通过下面的命令安装 TensorFlow：

     <pre>
     $ <b>pip install --upgrade</b> <i>tfBinaryURL</i> # Python 2.7
     $ <b>pip3 install --upgrade</b> <i>tfBinaryURL</i> # Python 3.n </pre>

     其中 <i>tfBinaryURL</i> 指向 TensorFlow Python 软件包所在的 URL。合适的 <i>tfBinaryURL</i> 取决于你的操作系统和 Python 版本。你可以在[这儿](#the_url_of_the_tensorflow_python_package)找到你系统所对应的 <i>tfBinaryURL</i>。例如，如果你要在安装了 Python 2.7 的 macOS 上安装 TensorFlow，那么可以执行下面的命令：

     <pre>
     $ <b>pip3 install --upgrade \
     https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py3-none-any.whl</b> </pre>

如果你遇到了任何安装问题，请查看[常见安装问题](#常见安装问题).

### 下一步

安装 TensorFlow 之后，你需要[验证安装](#ValidateYourInstallation)来确保 TensorFlow 能够正常工作。

注意，每当你使用一个新的 Shell 来使用 TensorFlow 时，你必须激活 Virtualenv 环境。如果 Virtualenv 环境没有被激活（即命令行提示符中没有 `(<i>targetDirectory</i>)`），使用下面的命令可以激活虚拟环境：

<pre>
$ <b>cd <i>targetDirectory</i></b>
$ <b>source ./bin/activate</b>      # bash、sh、ksh 或 zsh
$ <b>source ./bin/activate.csh</b>  # csh 或 tcsh </pre>

如果你的终端提示出现如下内容，证明 TensorFlow 的环境已经激活：

<pre>(<i>targetDirectory</i>)$ </pre>

当 Virtualenv 环境激活后，你就可以在 Shell 里运行 TensorFlow 程序了。

当你使用完 TensorFlow 后，你还可以解除虚拟环境：

<pre>(<i>targetDirectory</i>)$ <b>deactivate</b> </pre>

这时命令行提示将会变回你激活虚拟环境之前的样子。

### 卸载 TensorFlow

如果你希望卸载 TensorFlow，只需要简单的删除你创建的目录树即可。例如：

<pre>$ <b>rm -r ~/tensorflow</b> </pre>

## 通过本地 pip 安装

我们已经将 TensorFlow 的二进制编译版上传到了 PyPI 中。因此你可以直接使用 pip 进行安装。[setup.py 里要求的包](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py) 列出了 pip 需要安装或升级的包。

### 环境要求：Python

要安装 TensorFlow，你的系统必须至少包含一个以下版本的 Python：

  * Python 2.7
  * Python 3.3+

如果你的系统没有安装适当版本的 Python，那么赶紧[安装](https://wiki.python.org/moin/BeginnersGuide/Download)吧。

在安装 Python 时，你可能需要关闭系统完整性保护（SIP）来允许安装非 Mac App Store 的程序。

### 环境要求：pip

[Pip](https://en.wikipedia.org/wiki/Pip_(package_manager)) 能安装并管理 Python 编写的软件包。如果你想要通过本地 pip 进行安装，你的系统至少应该有包含下面命令中的一个：

  * `pip`, 对应 Python 2.7
  * `pip3`, 对应 Python 3.n.

如果你安装了 python，可能 `pip` 或 `pip3` 已经安装在你的系统上了。为了确定它们是否真的安装在系统里，
可以使用下面的命令：

<pre>$ <b>pip -V</b>  # 对应 Python 2.7
$ <b>pip3 -V</b> # 对应 Python 3.n </pre>

我们强烈推荐使用 8.1 或更高版本的 pip 或 pip3 来安装 TensorFlow。
如果你没有安装的话，可以通过下面的命令来升级当前的 pip：

<pre>$ <b>sudo easy_install --upgrade pip</b>
$ <b>sudo easy_install --upgrade six</b> </pre>

### 安装 TensorFlow

假设你的 Mac 上已经安装了依赖的软件，请遵循下面的步骤：

  1. 通过下面的一个命令来安装 TensorFlow：

     <pre>
     $ <b>pip install tensorflow</b>      # Python 2.7; CPU 支持
     $ <b>pip3 install tensorflow</b>     # Python 3.n; CPU 支持

     如果前面的命令执行完成了，那么接下来你应该
     [验证安装](#ValidateYourInstallation).

  2. （可选）如果第 1 步失败了，那么可以通过下面的命令安装最新版的 TensorFlow：

     <pre>
     $ <b>sudo pip  install --upgrade</b> <i>tfBinaryURL</i>   # Python 2.7
     $ <b>sudo pip3 install --upgrade</b> <i>tfBinaryURL</i>   # Python 3.n </pre>

     其中 <i>tfBinaryURL</i> 指向 TensorFlow Python 软件包的所在 URL。
     合适的 <i>tfBinaryURL</i> 取决于你的操作系统和 Python 版本。你可以在[here](#the_url_of_the_tensorflow_python_package)
     找到你系统所对应的 <i>tfBinaryURL</i>。
     例如，如果你要在安装了 Python 2.7 的 macOS 上安装 TensorFlow，那么可以执行下面的命令：

     <pre>
     $ <b>pip3 install --upgrade \
     https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py3-none-any.whl</b></pre>

### 下一步

TensorFlow 安装完成后，你应该[验证安装](#ValidateYourInstallation)是否能使 TensorFlow 正确工作。

### 卸载 TensorFlow

执行下面的命令可以卸载 TensorFlow：

<pre>$ <b>pip uninstall tensorflow</b>
$ <b>pip3 uninstall tensorflow</b> </pre>

## 通过 Docker 安装

遵循下面的步骤可通过 Docker 安装 TensorFlow：

  1. 安在你的机器上安装 Docker，请参考 [Docker 文档](https://docs.docker.com/engine/installation/#/on-macos-and-windows).
    
  2. 从包含 TensorFlow 的镜像中创建并启动 Docker 容器。

本节的其余部分将介绍如何启动 Docker 容器。

要启动包含 TensorFlow 镜像的 Docker 容器，请输入以下指令：

<pre>$ <b>docker run -it <i>-p hostPort:containerPort</i> TensorFlowImage</b> </pre>

其中：

  * <i>-p hostPort:containerPort</i> 可选。如果你想要在 shell 命令行中运行 TensorFlow，忽略这个选项。如果你想要在 Jupyter notebooks 中运行这个程序，将 <i>hostPort</i> 和 <i>containerPort</i> 都设置为 <code>8888</code>。

    如果你想要运行包含 TensorBoard 的容器，增加第二个 `-p` 参数来指定宿主端口和容器端口为 6006。

  * <i>TensorFlowImage</i> 是必须的。它指定了你的 Docker 容器。你必须指定下面其中一个值：
    * <code>tensorflow/tensorflow</code>: TensorFlow 二进制镜像。
    * <code>tensorflow/tensorflow:latest-devel</code>: TensorFlow
      二进制镜像和源代码。

例如，下面的命令从 TensorFlow CPU 镜像启动了一个 Docker 容器，从而你可以在这个命令行里执行 TensorFlow 程序：

<pre>$ <b>docker run -it tensorflow/tensorflow bash</b></pre>

下面的命令同样是从一个 TensorFlow CPU 的镜像启动的容器。然而在这个容器中，你还可以在 Jupyter notebook 里
运行 TensorFlow 程序：

<pre>$ <b>docker run -it -p 8888:8888 tensorflow/tensorflow</b></pre>

Docker 会在第一次启动容器时下载对应的镜像。

### 下一步

现在你应该[验证安装](#ValidateYourInstallation)。

## 通过 Anaconda 安装

**Anaconda 的安装由社区提供，而非官方支持。**

请按照下面的步骤在 Anaconda 环境中安装 TensorFlow：

  1. 按照[Anaconda 下载网站](https://www.continuum.io/downloads)中的指南来下载并安装 Anaconda。

  2. 调用以下命令新建一个名字叫 <tt>tensorflow</tt> 的 conda 环境来运行某一版本的 Python：

     <pre>$ <b>conda create -n tensorflow pip python=2.7 # or python=3.3, etc.</b></pre>

  3. 使用如下命令来激活 conda 环境：

     <pre>$ <b>source activate tensorflow</b>
     (<i>targetDirectory</i>)$  # Your prompt should change</pre>

  4. 运行如下格式的命令来在你的 conda 环境中安装 TensorFlow：

     <pre>(<i>targetDirectory</i>)<b>$ pip install --ignore-installed --upgrade</b> <i>TF_PYTHON_URL</i></pre>

     其中 <i>TF_PYTHON_URL</i>  是 [TensorFlow Python 包的 URL](#the_url_of_the_tensorflow_python_package)。例如，如下命令安装了仅支持 CPU 的 Python 2.7 版本下的 TensorFlow：

     <pre>(<i>targetDirectory</i>)$ <b>pip install --ignore-installed --upgrade \
     https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py2-none-any.whl</b></pre>

<a name="ValidateYourInstallation"></a>
## 验证安装

下面的步骤能够验证 TensorFlow 是否已正确安装：

    1. 确定你已经具备了运行 TensorFlow 程序的运行环境。
    2. 运行一个简短的 TensorFlow 程序。

### 准备环境

如果你已经安装了 pip、Virtualenv 或者 Anaconda，那么：

    1. 运行终端。
    2. 如果你使用 Virtualenv 或 Anaconda 进行的安装，请激活你的容器。
    3. 如果你是使用 TensorFlow 源码进行的安装，请切换到除了包含 TensorFlow 源码的任意目录下。

如果你使用 Docker 进行安装，启动一个运行 bash 的 Docker 容器，例如：

<pre>$ <b>docker run -it tensorflow/tensorflow bash</b></pre>

### 运行一个简短的 TensorFlow 程序

在命令行中输入下面的命令调用 Python：

<pre>$ <b>python</b></pre>

在 Python 交互命令行环境中输入下面的代码：

```python
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
```

如果你的系统正确的输出了下面的内容，那么说明你已经正确安装了 TensorFlow：

<pre>Hello, TensorFlow!</pre>

如果安装过程出现了错误，请看[常见安装问题](#常见安装问题)

学习更多内容，请到 [TensorFlow 教程](../tutorials/)。

## 常见安装问题

我们使用 Stack Overflow 来记录 TensorFlow 的安装问题及其解决方案。下表列出了一些常见安装问题的 Stack Overflow 答案的链接。如果你遇到下表中未列出的错误信息或其他安装问题，请在 Stack Overflow 中进行搜索。如果 Stack Overflow 没有相应的解决方案，请在 Stack Overflow 上询问一个有关它的新问题，并指定 `tensorflow` 标签。

<table>
<tr> <th>Stack Overflow 链接</th> <th>错误消息</th> </tr>

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

<tr>
  <td><a href="http://stackoverflow.com/q/42075397">42075397</a></td>
  <td>A <tt>pip install</tt> command triggers the following error:
<pre>...<lots of warnings and errors>
You have not agreed to the Xcode license agreements, please run
'xcodebuild -license' (for user-level acceptance) or
'sudo xcodebuild -license' (for system-wide acceptance) from within a
Terminal window to review and agree to the Xcode license agreements.
...<more stack trace output>
  File "numpy/core/setup.py", line 653, in get_mathlib_info

    raise RuntimeError("Broken toolchain: cannot link a simple C program")

RuntimeError: Broken toolchain: cannot link a simple C program</pre>
</td>

</table>

<a name="TF_PYTHON_URL"></a>

## TensorFlow 的 Python 包的 URL

一些安装方法中需要 TensorFlow Python 包的 URL，你所声明的量取决于你的 Python 版本。

### Python 2.7

<pre>
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py2-none-any.whl
</pre>

### Python 3.4、3.5 或 3.6

<pre>
https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.0-py3-none-any.whl
</pre>
