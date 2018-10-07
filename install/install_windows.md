# 在 Windows 上安装 TensorFlow

本指南将介绍如何在 Windows 上安装 TensorFlow。虽然这些说明可能也适用于其他 Windows 版本，但我们只在满足以下要求的计算机上验证过这些说明（而且我们只支持在此类计算机上按这些说明操作）：

  * 64 位、x86 台式机或笔记本电脑
  * Windows 7 或更高版本

## 选择准备安装的 TensorFlow 类型

从以下选项中选择您需要安装的 TensorFlow 类型：

-  **仅支持 CPU 的 TensorFlow。** 如果系统无 NVIDIA® GPU，则必须安装该版本。需要说明的是，该版本的 TensorFlow 相比另一版本更容易安装（通常 5 到 10 分钟即可完成安装），因此即使系统有 NVIDIA GPU，我们仍然推荐您优先安装该版本。预构建的二进制文件会使用 AVX 指令集。
- **支持 GPU 的 TensorFlow。** 一般而言，TensorFlow 程序在 GPU 上的运行速度要明显高于在 CPU 上的。因此，如果您的系统含符合以下先决条件的 NVIDIA ® GPU，且需要运行性能关键型应用程序，那么您最终需要安装此版本的 TensorFlow。

### 运行支持 GPU 版本 TensorFlow 的要求

若使用本指南中介绍的任一方式来安装支持 GPU 的 TensorFlow，那么您必须在系统中安装如下 NVIDIA 软件：

- CUDA® Toolkit 9.0。详细说明请查看[ NVIDIA 官方文档](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)。请确保您已按照 NVIDIA 官方文档描述将相关的 Cuda 路径名称添加到 %PATH% 环境变量中。
- 与 CUDA Toolkit 9.0 相关的 NVIDIA 驱动。
- cuDNN v7.0 版本。详细说明请查看[ NVIDIA 官方文档](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)。需要注意的是，一般而言，cuDNN 的安装地址和其他 CUDA DLL 是不同的。同时，请确保将安装 cuDNN DLL 的目录添加到 %PATH% 环境变量中。
- 支持 CUDA Compute Capability 3.0 或更高版本的 GPU 卡，可用从源码构建或 3.5 或更高版本的二进制文件。请在 [NVIDIA 官方文档](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) 中查询具备条件的 GPU 清单。

如果您使用的版本与上述要求不一致，请更新为规定的版本。特别说明的是，cuDNN 的版本必须与要求的一致：如果无法找到 cuDNN64_7.dll，那么 TensorFlow 将无法加载。如果您想使用其他版本的 cuDNN，您需要从源代码开始重新编译。

## 选择安装 TensorFlow 的方式

您需要选择安装 TensorFlow 的方式。当前的可选方式如下：
- 原生的 pip 
- 使用 Anaconda

原生 pip 无需安装虚拟环境，可以直接在系统安装 TensorFlow。由于原生的一个 pip 安装应用并没有被隔离在一个独立的应用中，使用 pip 安装方法可能会影响到系统里其他基于 Python 的安装。但是，如果您了解您系统里的 pip 和 Python 环境，那么使用原生 pip 安装仅仅只需要一条命令就够了。而且，如果您使用原生的 pip 安装方法，那么用户可以从系统的任何路径去运行 TensorFlow 程序。

在 Anaconda 中，你可以使用 conda 去创建一个虚拟环境（virtural environment）。但是，如果是使用 Anaconda 方式，我们依然推荐使用 pip 安装命令来安装 TensorFlow，而不是 conda 安装命令。

**注意：** conda 包是由社区提供的，而不是官方。也就是说，TensorFlow 团队并不会测试也不会维护 conda 包。使用 conda 包需要您自己承担风险。

## 使用原生pip安装

如果您的机器上没有安装以下版本的Python，请立刻安装：

- [Python 3.5.x 64-bit from python.org](https://www.python.org/downloads/release/python-352/)
- [Python 3.6.x 64-bit from python.org](https://www.python.org/downloads/release/python-362/)

在 Windows 上，TensorFlow 支持 Python3.5.x 版本和 Python 3.6.x 版本。需要注意的是， Python 3 使用的是 pip3 包管理， 这也是您用来安装 TensorFlow 的程序。

打开一个终端，开始安装 TensorFlow。然后在终端上运行正确的 pip3 安装命令。 安装仅支持 CPU 版本的 TensorFlow，请输入下面的命令：
`C:\> pip3 install --upgrade tensorflow`

安装 GPU 版本的 TensorFlow，请输入下面的命令：
`C:\> pip3 install --upgrade tensorflow-gpu`

## 使用 Anaconda 进行安装

**Anaconda 的安装包是由社区提供，非官方提供的。**

在 Anaconda 的环境下，按照以下步骤进行 TensorFlow 的安装：

1.根据网页 [Anaconda 下载站点](https://www.anaconda.com/download/)说明下载并安装 Anaconda。 
2.请通过使用以下命令来创建一个名为 tensorflow 的 conda 环境：

`C:\> conda create -n tensorflow pip python=3.5`

3.通过输入以下命令来激活一个 conda 环境：

`C:\> activate tensorflow`
`(tensorflow)C:\>  # Your prompt should change `

4.在 conda 环境里输入正确的命令来安装 TensorFlow。 安装仅支持 CPU 版本的 TensorFlow，请输入下面的命令：

`(tensorflow)C:\> pip install --ignore-installed --upgrade tensorflow `

如果是安装 GPU 版本的 TensorFlow，请输入下面的命令：

`(tensorflow)C:\> pip install --ignore-installed --upgrade tensorflow-gpu `

## 安装验证

打开一个终端。如果您采用 Anaconda 方式安装，则进入 Anaconda 环境。采用下列方式从你的 shell 激活 python：

`$python`

在 python 交互 shell 中输入下列代码：

```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> print(sess.run(hello))
```

如果系统的输出如下所示，那就说明您可以开始在上面撰写 TensorFlow 的程序了：

<pre>Hello, TensorFlow!</pre>

如果系统输出了一个错误信息而不是一个打招呼提示，请查看[常见安装问题](#常见安装问题)。

学习更多内容，请到 [TensorFlow 教程](../tutorials/)。

## 常见安装问题

我们使用 Stack Overflow 来记录 TensorFlow 的安装问题和修正方法。下表中包含有一些常见安装问题在 Stack Overflow 上的回答链接。如果您遇到的错误消息或安装问题不在下表中，请在 Stack Overflow 上搜索它的答案。如果 Stack Overflow 上并没有显示这个错误消息或者安装问题的答案，请在 Stack Overflow 上提一个关于这个错误消息或者安装问题的新问题，并给这个问题指定一个 `tensorflow` 的标签。

<table>
<tr> <th>Stack Overflow Link</th> <th>Error Message</th> </tr>

<tr>
  <td><a href="https://stackoverflow.com/q/41007279">41007279</a></td>
  <td>
  <pre>[...\stream_executor\dso_loader.cc] Couldn't open CUDA library nvcuda.dll</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/41007279">41007279</a></td>
  <td>
  <pre>[...\stream_executor\cuda\cuda_dnn.cc] Unable to load cuDNN DSO</pre>
  </td>
</tr>

<tr>
  <td><a href="http://stackoverflow.com/q/42006320">42006320</a></td>
  <td><pre>ImportError: Traceback (most recent call last):
File "...\tensorflow\core\framework\graph_pb2.py", line 6, in <module>
from google.protobuf import descriptor as _descriptor
ImportError: cannot import name 'descriptor'</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/42011070">42011070</a></td>
  <td><pre>No module named "pywrap_tensorflow"</pre></td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/42217532">42217532</a></td>
  <td>
  <pre>OpKernel ('op: "BestSplits" device_type: "CPU"') for unknown op: BestSplits</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/43134753">43134753</a></td>
  <td>
  <pre>The TensorFlow library wasn't compiled to use SSE instructions</pre>
  </td>
</tr>

<tr>
  <td><a href="https://stackoverflow.com/q/38896424">38896424</a></td>
  <td>
  <pre>Could not find a version that satisfies the requirement tensorflow</pre>
  </td>
</tr>
 
 </table>
