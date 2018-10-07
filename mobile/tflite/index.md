# TensorFlow Lite 简介

TensorFlow Lite 是 TensorFlow 移动和嵌入式设备轻量级解决方案。它使设备机器学习具有低延迟和更小的二进制体积。TensorFlow Lite 同时支持 [Android 神经网络 API](https://developer.android.com/ndk/guides/neuralnetworks/index.html)的硬件加速.

TensorFlow Lite 使用多项技术降低延迟，例如移动 app 内核优化、pre-fused 激活、允许更快更小（定点）模型的量化内核。

目前大部分 TensorFlow Lite 文档放在 [GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite)上。

## TensorFlow Lite 都包含什么？

TensorFlow Lite 支持一系列 quantized 和浮点的核心运算符,并针对移动平台进行了优化。它结合 pre-fused 激活和其他技术来进一步提高性能和量化精度。此外，TensorFlow Lite 还支持在模型中使用自定义操作。

TensorFlow Lite 基于 [FlatBuffers](https://google.github.io/flatbuffers/)定义了一个新的模型文件格式。FlatBuffers 是一个开源的高效的跨平台序列化库。它与 [protocol buffers](https://developers.google.com/protocol-buffers/?hl=en)类似,但主要区别是 FlatBuffers 常与 per-object 内存分配相结合在你直接访问数据时不需要再次解析包。此外，FlatBuffers 的代码体积比 protocol buffers 小很多。

TensorFlow Lite 拥有一个新的移动设备优化的解释器保证应用程序的精简和快速。解释器使用静态图形排序和自定义（less-dynamic）内存分配器来确保最小的负载，初始化和执行延迟。

TensorFlow Lite 针对支持的设备提供了一个利用硬件加速的接口。该接口在 Android 8.1（API level 27）及更高级版本的 [Android 神经网络库 API](https://developer.android.com/ndk/guides/neuralnetworks/index.html) 中提供给大家。

## 为什么我们需要针对移动端的库？

机器学习正在改变计算模式，我们看到了移动和嵌入式设备上使用的新趋势。消费者的期望也趋向于与他们的设备自然而友好地互动，由相机和语音交互模式驱动。

有几个因素促成了这个领域：

- 硅层的创新为硬件加速带来了新的可能性，像 Android Neural Networks API 这样的框架可以很容易地利用这些特性。

- 实时计算机视觉和口头语言理解的最新进展：一些 mobile-optimized 评测模型已开放源代码(例如 MobileNets，SqueezeNet)。

- 广泛使用的智能设备为 on-device 智能创造了新的可能性。

- 更强大的用户隐私保护方案，使用户数据不需要离开移动设备。

- 能够在设备不需要连接到网络的情况下提供"离线"用例。

我们相信下一波机器学习应用将在移动和嵌入式设备上重大进步。

## TensorFlow Lite 开发者预览版亮点

TensorFlow Lite 作为开发者预览版亮点，包括以下内容：

- 一组核心操作符，既有 Quantized 也有浮点值，其中很多已经针对移动平台进行了优化。这些可以用来创建和运行自定义模型。开发人员也可以编写自己的自定义操作符并在模型中使用它们。

- 基于 [FlatBuffers](https://google.github.io/flatbuffers/)模型文件格式。

- On-device 解释器，内核经过优化，可在移动设备上更快执行。

- TensorFlow 转换器将 TensorFlow 训练好的模型转换为 TensorFlow Lite 格式。

- 更小的体积：当所有支持的操作符链接时，TensorFlow Lite 小于 300 KB，而仅使用支持 Inception V3 和 Mobilenet 所需的操作符时，TensorFlow Lite 小于 200 KB。

- **预设模型:**

    以下所有预设模型均可保证开箱即用：

    - Inception V3 是一种用于检测图像中存在的主要对象的流行模型。

    - [MobileNets](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md),
      一系列移动优先计算机视觉模型，旨在有效提高准确性，同时注重 on-device 或嵌入式应用程序的有限资源。它们很小，低延迟，低功耗，模型参数化，以满足各种用例的资源约束。它们可以建立在分类，检测，嵌入和分割之上。MobileNet 模型比较小，但是比 Inception V3 [准确度更低](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)。

    - 在设备智能回复上, 提供能通过建议上下文相关的消息来回复传入的文本消息的 one-touch 的 on-device 模型，该模型专为内存受限的设备而建立，如手表和手机，
      它已被成功应用到 [Android Wear 智能回复](https://research.googleblog.com/2017/02/on-device-machine-intelligence.html)所有官方和第三方应用。

    还可以查看 [TensorFlow Lite 支持的模型的完整列表](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md)，包括模型大小、性能和可下载的模型文件。

- MobileNet 模型 Quantized 版本, 其运行速度比 CPU 上的 non-quantized（浮点）版本快。

- 新的 Android 演示应用程序来说明使用 TensorFlow Lite 与Quantized 的 MobileNet 模型进行对象分类。

- Java 和 C++ API 支持

注意：这是一个开发者版本，很可能在即将到来的版本中会有 API 的变化。我们不保证向后或向前兼容这个版本。

## 入门

我们建议你使用上述的 TensorFlow Lite 的 pre-tested 模型。如果有一个现有的模型，则需要测试模型是否兼容转换器和支持的操作集。要测试你的模型，请看 [GitHub 文档](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite)。

### 为自定义数据集重置 Inception-V3 或 MobileNet

这里的 pre-trained 模型已经在 ImageNet 数据集上进行了训练 ，该数据集由 1000 个预定义的类组成。如果这些类不适合你的用例，那么你需要重新训练这些模型。这种从一个已经被训练过的问题的模型开始，然后在类似的问题上进行再训练叫做迁移学习。从头开始深入学习可能需要几天，但迁移学习可以很快完成。为了做到这一点，你需要生成标有相关类的自定义数据集。

[TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/) 一步一步实现了这个过程，再训练代码支持浮点和量化推理的再训练。

## TensorFlow Lite 架构

下图显示了 TensorFlow Lite 的架构设计：

![tensorflow lite 架构](https://www.tensorflow.org/images/tflite-architecture.jpg)

在硬盘上练习 TensorFlow 模型, 你需要使用 TensorFlow Lite 转换器把模型转换为 TensorFlow Lite（`.tflite`）文件格式。然后，你可以在移动应用程序中使用该转换后的文件。

部署 TensorFlow Lite 模型文件使用：

- Java API: 围绕 Android 上 C++ API 的便捷包装。

- C++ API: 加载 TensorFlow Lite 模型文件并调用解释器。Android 和 iOS 都提供相同的库。

- 解释器: 使用一组内核来执行模型。解释器支持选择性内核加载;没有内核，只有 100 KB，加载了所有内核，有 300 KB。这比 TensorFlow Mobile 要求的 1.5 M 的显著减少。

- 在选定的 Android 设备上，解释器将使用 Android 神经网络 API 进行硬件加速，如果没有可用的，则默认为 CPU 执行。

你也可以使用解释器可以使用的 C++ API 来实现定制的内核。

## 未来工作

在未来的版本中，TensorFlow Lite 支持更多的模型和内置运算符, 包含定点和浮点模型的性能改进，改进工具以简化开发者工作流程以及支持其他更小的设备等等。随着我们的不断发展，我们希望 TensorFlow Lite 能够大大简化针对小型设备定位模型的开发者体验。

未来的计划包括使用专门的机器学习硬件，以获得特定设备上特定型号的最佳性能。

## 下一步

对于开发人员的预览，我们的大部分文档都在 GitHub 上。请查看 [TensorFlow Lite 库](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite)在 GitHub 上获取更多信息和代码示例，演示应用程序等等。
