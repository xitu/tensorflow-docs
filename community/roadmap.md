# 路线图
**最近一次更新：2018.02.15**

TensorFlow 是一个频繁更新并且社区活跃的项目。这份文档旨在提供关于 TensorFlow 高优先级领域以及核心开发成员专注领域的发展路线指南，也包括未来的 TensorFlow 新版本期望发行的功能。这里很多功能是靠社区内部的测试用例驱动开发的，此外我们也欢迎对 TensorFlow 更进一步的[讨论](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)。

以下功能并没有具体计划的发行日期，但是大部分功能会在未来的一两个版本中发行。

### APIs
#### 高阶 APIs:
* 基于 Estimators 的友好型 multi-GPU 接口
* Gradient Boosted Trees, Time Series 及其他模型的友好型高阶预制 estimators

#### Eager Execution:
* multiple GPUs 的高效使用接口
* 分布式训练 (多机器训练)
* 性能提升
* 更友好地将模型导出到 GraphDef/SavedModel 

#### Keras API:
* 更好的整合 tf.data (让数据张量有能力直接调用 `model.fit` )
* 完全支持 Eager Execution (包括对常规的 Keras API 的 Eager support 支持，以及通过模型子类创建 Eager 风格的 Keras 模型的能力)
* 更好的支持 distribution/multi-GPU 和 TPU (包括更平滑的 model-to-estimator 工作流)

#### 官方模型：
* 这一系列的
[参考模型](https://github.com/tensorflow/models/tree/master/official)
包含了图像识别、自然语言处理、物体检测、机器翻译等领域，这些都算得上是最佳的练习选择以及学习构建高性能模型训练的理想切入点。

#### Contrib:
* 为 tf.contrib 增加弃用声明，目前 tf.contrib 更倾向于在外部实现该功能。
* 尽可能地将 tf.contrib 的大工程迁移到多个独立的仓库。
* 目前这种形式的 tf.contrib 模块最终会停止开发，将来会在其他的仓库进行试验性开发。


#### 概率推理与统计分析：
* 在 tf.distributions 和 tf.probability 当中有大量的概率推理与统计分析工具可供使用。包括新的采样器、层、优化器、损失和结构化模型。
* 用于假设检验、收敛诊断、样本统计的统计工具。
* Edward 2.0: 用于 probabilistic programming 的高阶 API

### 跨平台
#### TensorFlow Lite:
* 增加 TensorFlow Lite 可支持操作的覆盖率
* 为 TensorFlow Lite 提供更易转换的已训练模型
* 为 TensorFlow Lite 提供 GPU 加速 (包括 iOS 和 Android)
* 通过 Android NeuralNets API 提供硬件加速支持
* 通过量化和其他网络优化提升 CPU 性能 (例如 pruning, distillation)
* 为 Android 和 iOS 以外的设备提升支持 (例如 树莓派, Cortex-M)

### 性能
#### 分布式 TensorFlow:
* 为多种 GPU 拓扑结构提供 Multi-GPU 优化支持
* 为多机器分布式计算改进实现机制

#### 优化：
* 为混合精度的训练提供支持，并且提供基本的示例模型和操作指南
* 原生的 TensorRT 支持
* 通过 MKY 为 SkyLake 提供 Int8 支持
* 动态加载 SIMD-optimized 内核

### 文档与可用性：
* 更新文档、教程以及快速上手指南
* 允许外部为 TensorFlow 提供支持，其中包括教程、文档以及用博客展示最佳的 TensorFlow 练习用例或者酷炫应用

### 社区以及合作伙伴
#### 特别兴趣小组： 
* 动员社区成员一起为重要领域相互协作
* [tf-distribute](https://groups.google.com/a/tensorflow.org/forum/#!forum/tf-distribute): 用于 TensorFlow 工程构建以及打包
* 更多待定的以及急需开展的计划

#### 社区:
* 通过 Request-for-Comment (RFC) 来集合公众对于重大设计决策的意见
* 为 TensorFlow 以及相关项目的外部贡献制定程序流程
* 为 TensorFlow 培育全球性的社区以及用户群体
* 与行业伙伴合作以联合发展并对外发表论文研究成果
