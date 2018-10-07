# 开发者指南

在你的移动 app 中使用 TensorFlow Lite 格式有如下诸多因素需要注意：你必须选择一个预训练或者自定义的模型，把这个模型转化为 TensorFlow Lite 格式，最后把模型整合进你的 app 中。

## 1. 模型选择

取决于使用案例，你可以选择任何一个流行的开源模型，例如 *InceptionV3* 或者 *MobileNets*，然后使用你自定义的数据集对（你选择的）这些模型进行重新训练，或者甚至构建你自定义的模型。

### 使用预训练模型

[MobileNets](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html) 是一系列专为 TensorFlow 设计的移动（场景）优先的计算机视觉模型，这类模型用于有效地最大化（提升）精确度，同时，模型还考虑到设备内置应用或嵌入式应用的资源限制问题。MobileNets 是小型化、低延迟、低能耗的模型，能够参数化地满足各种各样使用案例中资源限制的要求。这些模型可以被用于分类、检测、嵌入和分割 —— 原理和其他一些流行的大规模模型相似，例如 [Inception](https://arxiv.org/pdf/1602.07261.pdf)。谷歌提供了 16 个利用 [ImageNet](http://www.image-net.org/challenges/LSVRC/) 进行预训练的 MobileNets 模型分类检查点（译者注：专业术语为 checkpoints），可用于各种规模的移动项目。

[Inception-v3](https://arxiv.org/abs/1512.00567) 是一个能对 1000 种常见事物，例如，“斑马”、“斑点狗”、“洗碗机”等进行非常高精度识别的图像识别模型。这个模型使用了一个卷积神经网络来提取输入图片的一般特征，然后在这些特征的基础上使用全连接和 softmax 层来对这些图片进行分类。

[On Device Smart Reply](https://research.googleblog.com/2017/02/on-device-machine-intelligence.html) 是一种设备内置的模型，该模型通过提示与上下文相关的消息，为传入的文本消息提供一键回复。这种模型专门用于内存受限的设备，如手表和手机，并且已经成功地用于 Android Wear 的智能回复。目前，该模型仅仅用于安卓系统。

这些预训练模型可以在[这里下载](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md)。

### 为自定义的数据集重新训练 Inception-V3 或 MobileNet 模型

这些预训练模型都是使用 *ImageNet* 数据集（一种包含了1000种预定义类型的数据集）进行训练的。如果这些类别不能满足于你使用案例的需求，你就需要对这个模型进行重新训练。这种技术被称为**迁移学习**，这需要使用一个已经基于某个问题训练过的模型，然后在相似的问题下对该模型进行重新训练。从头开始进行深度学习可能需要一些时间，但是使用转移学习技术却相当快。为了做到这一点，你需要生成一个标记为（和你问题）相关的类的自定义数据集。

[TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/) 在 codelab 上展示了一步步地进行再训练的过程。这些代码支持浮点和量化推断。

### 自定义模型训练

开发者可以选择使用 Tensorflow 对自定义的模型进行训练（构建和训练模型的案例可参照 [TensorFlow 教程](../../tutorials/)）。如果你已经写好了一个模型，第一步是把模型导出为一个 `tf.GraphDef` 文件。这个步骤是必需的，因为除代码之外，有些格式并不存储模型结构，而我们必须与框架的其他部分进行通信。为了为自定义模型创造 .pb 文件，你可以参照 [Exporting the Inference Graph](https://github.com/tensorflow/models/blob/master/research/slim/README.md)。

TensorFlow Lite 目前支持一组 TensorFlow 操作符。可通过参考 [TensorFlow Lite 和 TensorFlow 兼容性指南](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/g3doc/tf_ops_compatibility.md)获得现在支持的操作符以及其使用案例。在未来的 Tensorflow Lite 版本中，这组操作符将不断增加。

## 2. 模型格式转换

在前面步骤中生成（或者下载）的模型是一个**标准版**的 Tensorflow 模型，你现在应该已经有了一个 .pb 或者 .pbtxt `tf.GraphDef` 文件了。通过迁移学习（也可称为再训练）或者自定义模型生成的模型必须被转换 —— 但是（在此之前），我们必须首先进行模型固化，将模型转换为 Tensorflow Lite 格式。这个过程将使用到以下一些模型格式：

* `tf.GraphDef` (.pb) —— 一个代表了 TensorFlow 训练或者计算图的 protobuf（译者注：一种轻便高效的结构化数据存储格式）。这个结构包含了操作符、张量和变量定义。
* *CheckPoint* (.ckpt) —— 通过一张 TensorFlow 图得到的序列化变量。因为这个格式没有包含图的结构，因此该格式无法进行自解释。
* `FrozenGraphDef` —— 一个没有包含变量的 `GraphDef` 子类。通过选取一个检查点和一个 `GraphDef`，可以把 `GraphDef` 转化为 `FrozenGraphDef`，并使用从检查点检索到的值将每个变量转换为常量。
* `SaveModel` —— 带有签名的 `GraphDef` 和检查点，该签名将输入和输出参数标记为模型。可以从 `SavedModel` 中提取 `GraphDef` 和检查点。
* **TensorFlow Lite 模型** (.tflite) —— 一个序列化的 [FlatBuffer](https://google.github.io/flatbuffers/)，其中包含了 TensorFlow Lite 操作符和张量，用于TensorFlow Lite 解释器，和 `FrozenGraphDef` 相似。

### 图固化（译者注：指把训练数据和模型固化成 .pb 文件）

为了在 TensorFlow Lite 模型上使用 `GraphDef.pb` 文件，你必须拥有包含已训练权重参数的检查点。`.pb` 文件仅仅包含了图的数据结构。把检查点值和图结构进行合并的操作被称为**图固化**。

你应该已经拥有一个检查点文件夹或者已经从一个预训练模型中下载了检查点（例如，[MobileNets](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)）。

使用如下一些命令来对图进行固化（使用时请修改参数）：
```
freeze_graph --input_graph=/tmp/mobilenet_v1_224.pb \
  --input_checkpoint=/tmp/checkpoints/mobilenet-10202.ckpt \
  --input_binary=true \
  --output_graph=/tmp/frozen_mobilenet_v1_224.pb \
  --output_node_names=MobileNetV1/Predictions/Reshape_1
```

必须启用 `input_binary` 标志位，以便以二进制格式读取和写入 protobuf。设置 `input_graph` 和 `input_checkpoint` 文件。

在构建模型的代码之外，`output_node_names` 可能并不明显。要找到它们，最简单的方法是使用 [TensorBoard](https://codelabs.developers.google.com/codelabs/tensorflow-for poets-2/#3) 或 `graphviz` 来可视化图形。

固化的 `GraphDef` 现在可以转换为 `FlatBuffer` 格式（.tflite），以便在安卓或 iOS 设备上使用。对于安卓来说，Tensorflow 优化转换器工具同时支持浮动模型和量化模型。如下代码将固化的 `GraphDef` 转换为 .tflite 格式:

```
toco --input_file=$(pwd)/mobilenet_v1_1.0_224/frozen_graph.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --output_file=/tmp/mobilenet_v1_1.0_224.tflite \
  --inference_type=FLOAT \
  --input_type=FLOAT \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1 \
  --input_shapes=1,224,224,3
```

`input_file` 参数应该引用包含模型架构的固化 `GraphDef` 文件。这里可以下载使用到的 [frozen_graph.pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz) 文件。`output_file` 是生成 TensorFlow Lite 模型的地方。`input_type` 和 `inference_type` 参数应该设置为浮点数，除非转换为 @{$performance/quantization$quantized model}。设置 `input_array`、`output_array` 和 `input_shape` 参数并不那么简单。找到这些值的最简单的方法是使用 Tensorboard 来研究图形。在 `freeze_graph` 步骤中重用指定输出节点进行推理的参数。

你还可以使用来自 Python 或者命令行（参见 [toco_from_protos.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/toco/python/toco_from_protos.py) 案例）的含有 protobufs 的 Tensorflow 优化转换器。这允许你将转换步骤集成到模型设计工作流中，确保模型可以轻松地转换为移动推理图。例如：

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
out = tf.identity(val, name="out")

with tf.Session() as sess:
  tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [img], [out])
  open("converteds_model.tflite", "wb").write(tflite_model)
```

有关使用情况，请参阅 Tensorflow 优化转换器[命令行工具案例](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/toco/g3doc/cmdline_examples.md)。

参照[运维兼容性指南](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/g3doc/tf_ops_compatibility.md)进行故障诊断帮助，如果你在这份指南里没有获得帮助，请提一个 [issue](https://github.com/tensorflow/tensorflow/issues)。

这份[开发仓库](https://github.com/tensorflow/tensorflow)包含了一个可以在转换之后可视化 TensorFlow Lite 模型的工具。你可以使用 [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/tools/visualize.py) 构建这个工具：

```sh
bazel run tensorflow/contrib/lite/tools:visualize -- model.tflite model_viz.html
```

这会生成一个交互式的 HTML 页面，在这个页面中会列出子图，操作和可视化的图形。


## 3. 在移动 app 中引用 TensorFlow Lite 模型

在完成了上述的步骤之后，你现在应该已经获得了一个 `.tflite` 模型文件了。

### 安卓

因为安卓 app 都是用 Java 语言编写的，同时 TesorFlow 核心库是基于 C++ 编写的，因此还提供了一个 JNI（译者注：JNI 是Java Native Interface 的缩写，它提供了若干的 API，实现了 Java 和其他语言，主要是 C 和 C++ 的通信）接口。这个接口仅用于推断 —— 它提供了加载图形、输入设置和运行模型来计算输出的能力。

这个开源的安卓 demo app 使用了 JNI 接口，这个接口在 [GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/java/demo/app) 上面。你也可以下载一个[预构建 APK](http://download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk)，查看 @{$tflite/demo_android} 指南获取详细信息。

如下这份指南 @{$mobile/android_build} 提供了在安卓上安装 TensorFlow 的方法以及设置 `bazel` 和安装 Android Studio 的方法。

### iOS

要在 iOS 应用程序中集成一个 TensorFlow 模型，请参见 [TensorFlow Lite for iOS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/index/g3doc/ios.md) 指南和 @{$tflite/demo_ios} 指南。

#### Core ML 支持

Core ML 是一个用于苹果产品的机器学习框架。除了直接在你的应用中使用 Tensorflow Lite 模型，你也可以把你的 Tensorflow 模型转换训练成能够应用于苹果设备的 [CoreML](https://developer.apple.com/machine-learning/) 格式。要使用这个转换器，请参见 [Tensorflow-CoreML 转换文档](https://github.com/tf-coreml/tf-coreml)。

### 树莓派

根据下述的 [RPi 构建指导](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/rpi.md) 为树莓派编译 Tensorflow Lite 模型。这个操作编译了一个用于构建你 app 的静态库文件（`.a`）。里面包含了一些用于 Python 绑定的计划和一个 demo app。
