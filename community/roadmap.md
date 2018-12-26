# 路线图

**最近一次更新：2018.02.15**

TensorFlow 是一个频繁更新并且社区活跃的项目。这份文档旨在提供关于 TensorFlow 高优先级领域以及核心开发成员专注领域的发展路线指南，也包括一些在将来发行的 TensorFlow 版本中备受期待的功能。许多灵感受到了社区用例的启发，我们也欢迎对 TensorFlow 进行更多的[贡献](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)。

## 即将到来的 TensorFlow 2.0

[如同近期宣布的一样](https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/bgug1G6a89A)，我们已经开始了下一个 TensorFlow 主版本的工作。TensorFlow 2.0 着眼于易用性，将是一个意义重大的里程碑。以下是 TensorFlow 2.0 的一些亮点：

* 动态执行是 2.0 版本的核心特征。它使用户对编程模型的期望与 TensorFlow 实际表现更好地一致，并且使 TensorFlow 更易于学习和使用。
* 支持更多平台和语言，并通过标准化交换格式和接口对齐，提升了组件间的兼容性和等价性。
* 我们会移除废弃的 API 并减少重复代码，这样可以避免用户感到困惑。

关于 2.0 版本的更多详细内容和相关的公开设计协商，请参阅[完整声明](https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/bgug1G6a89A)。

## 导览

以下特性没有具体的发布日期。不过，大多数预计在接下去的一到两个版本发布。

### APIs

#### 上层 API

* 对 Keras、Eager 和 Estimator 进行了高度集成，将会使用相同的数据管道、API 和序列化格式（Saved Model）。
* 将 TensorFlow 核心中的常用 ML 模型（如 TimeSeries、RNN、TensorForest 以及更多其它的增强树）和相关功能（如顺序特征列）封装成 Estimator（如果已存在的话，则从软件库迁移）。

#### 动态执行

* 使用分布式策略来充分利用多 GPU 和多 TPU 核心。
* 支持分布式训练（多机部署）。
* 性能提升。
* 简化导出到 GraphDef/SavedModel 的流程。

#### 参考模型

* 内建了一组跨图像识别、对象检测、语音、翻译、推荐和增强学习的[模型](https://github.com/tensorflow/models/tree/master/official)，这些模型能够展示最佳效果，并作为高性能模型开发的起点。
* 扩大了高性能[云 TPU 参考模型](https://github.com/tensorflow/tpu)的规模。

#### 扩展包

* 废弃 `tf.contrib` 中有更好的外部实现的部分。
* 尽可能地将 `tf.contrib` 中的大型项目移到单独的仓库中。
* TensorFlow 2.0 中，`tf.contrib` 将不再维护。将来会在其它仓库进行一系列试验性的开发。

### 平台

#### TensorFlow Lite：

* 增加 TensorFlow Lite 可支持操作的覆盖率。
* 为 TensorFlow Lite 提供更易转换的已训练模型。
* 添加优化移动模型的工具。
* 添加对 Edge TPU 和 TPU AIY 主板的支持。
* 更加易用的文档和教程。

#### TensorFlow.js

* 提高 TensorFlow.js 在浏览器中的性能；使用计算着色器或 WebGPU 实现原型；提高 CPU 性能，提供 SIMD+ Web 程序集支持。
* 扩展对导入 TensorFlow SavedModel 和 Keras 模型的支持，特别是基于音频和文本的模型。
* 发布了一个新的 tfjs-data API，用于高效的数据输入管道，以及一个新的 tfjs-vis 库，用于在浏览器训练模型期间提供交互式的模型可视化。
* 在服务器端 TensorFlow.js 使用 Node —— 通过公开所有 TensorFlow 操作来提高与原生 TensorFlow 操作和模型格式的兼容性；使用 libuv 添加异步模式支持。

#### TensorFlow with Swift

* 延续 2018 整年的工作，对设计和实现进行持续精炼。
* 到 2018 年底，保证核心组件（图形程序提取、基础的自动微分（AutoDiff）、发送/接收）在日常使用中足够可靠。
* 在 2018 年探索在 TensorFlow 中使用 Swift 来构建动态模型的语言。
* 在 2019 年初提供在 Colab 中使用 TensorFlow 的基本教程。

### 性能

#### 分布式 TensorFlow

* 扩展新的分布式策略 API 来在 TPU 和多节点 GPU 上支持 Keras。
* 对出色的开箱即用性能和易于部署的特性的演示。

#### GPU 优化

* 修改通用设计，简化固定精度 API。
* 完成 TensorRT API 并移入核心模块。
* TensorRT 现在支持 SavedModel 和 TF Serving 了。
* 集成 CUDA 10（计划跳过 CUDA 9.2 版本因为跟 CUDA 9.0 相比它在相同版本的 cuDNN 下的改进太小了）。
* 优化对 DGX-2 的支持。

#### 云 TPU 和云 TPU Pods

* 在云 TPU 上扩展对 Keras 的支持，并进一步优化性能。
* 扩展对图像分割的支持 —— 在现有的 [RetinaNet](https://github.com/tensorflow/tpu/tree/master/models/official/retinanet) 和 [DeepLab](https://github.com/tensorflow/tpu/tree/master/models/experimental/deeplab) 语义分割参考模型中加入 Mask R-CNN。
* 优化了新的 Cloud TPU 集成：GKE、CMLE、Cloud Bigtable 和 gRPC 的数据输入。
* 允许在 Cloud TPU Pod 上进行大型模型的并发训练了。
* 优化了 Cloud TPU v3 上参考模型的性能。

#### CPU 优化

* 通过 MKL 为 SkyLake 系列处理器提供 Int8 支持。
* 通过 MKL 提供更快的 3D 操作。
* 动态加载专为 SIMD 优化的内核。
* MKL 的 Linux 和 Windows 支持。

### 其它模块

#### TensorFlow 概率编程工具箱

* 高斯过程的详细实现，包括超参数优化方面的应用。
* 新增贝叶斯结构时间序列模型。
* 对采样和优化方法进行了改进。
* 丰富的在 Colab 上使用 TensorFlow 概率（TFP）的教程。

#### Tensor2Tensor 库

* 新的支持自动编码器、GAN 和 RL 的视频、语音和音乐方面的数据集和模型。
* 使用 TensorFlow 2.0 的最佳实现来提升对各个平台的支持并简化内部结构。
* 用 Mesh TensorFlow 并行训练大型模型。

### 端到端 ML 系统

#### TensorFlow Hub

* 扩展 TF-Hub 模块，添加对 TF Eager、Keras layer 和 TensorFlow.js 集成的支持，并支持 TF-Transform 和 TF-Data 工作流。

#### TensorFlow Extended

* 开源更多的 TensorFlow Extended 平台来促进 TensorFlow 在生产中的应用。
* 封装用于模型评估和校验的 TF Model Analysis。
* 发布用于数据验证的 TFX 库。
* 发布端到端 ML 管道工作流的示例。

### 社区和合作伙伴

#### 特殊兴趣小组（SIG）

* 动员整个社区对重点领域进行合作。
* [评估并生成新的特殊兴趣小组](https://github.com/tensorflow/community/blob/master/governance/SIGS.md)，并将其作为社区生态的一员。

#### 社区

* 在[请求意见稿（RFC）流程](https://github.com/tensorflow/community/blob/master/governance/TF-RFCs.md)中对重要的设计决策进行持续的公开的反馈。
* 创建了贡献者指南来促进[公开管理流程](https://github.com/tensorflow/community/tree/master/governance)。
* 发展全球的 TensorFlow 社区和用户小组。
* 与合作伙伴协作开发并发布研究论文。
* 持续发布博文和 YouTube 视频，展示 TensorFlow 的应用并构建重要应用的用例研究。
