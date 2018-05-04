# 路线图
**最近一次更新：2018.02.15**

TensorFlow 是一个频繁更新并且社区活跃的项目。这份文档旨在提供关于 TensorFlow 高优先级领域以及核心开发成员专注领域的发展路线指南，也包括未来的 TensorFlow 新版本期望发行的功能。这里很多功能是靠社区内部的测试用例驱动开发的，此外我们也欢迎对 TensorFlow 更进一步的[讨论](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md)。

以下功能并没有具体计划的发行日期，但是大部分功能会在未来的一两个发行版中发行。

### APIs
#### 高阶 APIs:
* 基于 Estimators 的友好型 multi-GPU 接口
* Boosted Trees, Time Series 及其他模型的友好型高阶预制 estimators

#### Eager Execution:
* multiple GPUs 的高效使用接口
* 分布式训练 (多机器训练)
* 性能提升
* 更友好地将模型导出到 GraphDef/SavedModel 

#### Keras API:
* 更好的整合 tf.data (让数据张量有能力直接调用 `model.fit` )
* 完全支持 Eager Execution (包括对常规的 Keras API 的 Eager support 支持，以及通过模型子类创建 Eager 风格的 Keras 模型的能力)
* 更好的支持 distribution/multi-GPU 和 TPU (包括更平滑的 model-to-estimator 工作流)

#### Official Models:
* A set of 
[reference models](https://github.com/tensorflow/models/tree/master/official) 
across image recognition, speech, object detection, and 
  translation that demonstrate best practices and serve as a starting point for 
  high-performance model development.

#### Contrib:
* Deprecation notices added to parts of tf.contrib where preferred implementations exist outside of tf.contrib.
* As much as possible, large projects inside tf.contrib moved to separate repositories.
* The tf.contrib module will eventually be discontinued in its current form, experimental development will in future happen in other repositories.


#### Probabilistic Reasoning and Statistical Analysis:
* Rich set of tools for probabilistic and statistical analysis in tf.distributions 
  and tf.probability. These include new samplers, layers, optimizers, losses, and structured models
* Statistical tools for hypothesis testing, convergence diagnostics, and sample statistics
* Edward 2.0: High-level API for probabilistic programming

### Platforms
#### TensorFlow Lite:
* Increased coverage of supported ops in TensorFlow Lite
* Easier conversion of a trained TensorFlow graph for use on TensorFlow Lite
* Support for GPU acceleration in TensorFlow Lite (iOS and Android)
* Support for hardware accelerators via Android NeuralNets API 
* Improved CPU performance by quantization and other network optimizations (eg. pruning, distillation)
* Increased support for devices beyond Android and iOS (eg. RPi, Cortex-M)

### Performance
#### Distributed TensorFlow:
* Multi-GPU support optimized for a variety of GPU topologies
* Improved mechanisms for distributing computations on several machines

#### Optimizations:
* Mixed precision training support with initial example model and guide
* Native TensorRT support
* Int8 support for SkyLake via MKL
* Dynamic loading of SIMD-optimized kernels

### Documentation and Usability:
* Updated documentation, tutorials and Getting Started guides
* Process to enable external contributions to tutorials, documentation, and blogs showcasing best practice use-cases of TensorFlow and high-impact applications

### Community and Partner Engagement
#### Special Interest Groups: 
* Mobilizing the community to work together in focused domains
* [tf-distribute](https://groups.google.com/a/tensorflow.org/forum/#!forum/tf-distribute): build and packaging of TensorFlow
* More to be identified and launched

#### Community:
* Incorporate public feedback on significant design decisions via a Request-for-Comment (RFC) process
* Formalize process for external contributions to land in TensorFlow and associated projects 
* Grow global TensorFlow communities and user groups
* Collaborate with partners to co-develop and publish research papers
