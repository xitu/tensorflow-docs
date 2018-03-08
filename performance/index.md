# 性能

在训练机器学习模型时，性能往往是一个值得关注的问题。本节介绍了数种优化性能的方法。请阅读 @{$performance_guide$Performance Guide} 开始您的研究，阅读 @{$performance_models$High-Performance Models} 更深入得了解相关技术细节：

  * @{$performance_guide$Performance Guide}，本文囊括了一系列优化 TensorFlow 代码的最佳实践。

  * @{$performance_models$High-Performance Models}，本文包括了一系列针对不同类型系统与网络拓扑设计高拓展性模型的进阶技术。

  * @{$performance/benchmarks$Benchmarks}，包含了一些基准测试的结果。

XLA（加速线性代数）是一个正处于实验阶段、用于优化 TensorFlow 计算的线性代数编译器。您可以阅读以下指南探索 XLA：

  * @{$xla$XLA Overview}：XLA 简介。
  * @{$broadcasting$Broadcasting Semantics}，本文介绍了 XLA 的广播语义。
  * @{$developing_new_backend$Developing a new back end for XLA}，本文解释了如何将 TensorFlow 重定位至硬件，以对在特定硬件上运行的计算图进行性能优化。
  * @{$jit$Using JIT Compilation}，本文描述了通过 XLA 编译及运行部分 TensorFlow 图，以优化性能的 XLA JIT 编译器。
  * @{$operation_semantics$Operation Semantics}，本文为 `ComputationBuilder` 接口的操作语义参考手册。
  * @{$shapes$Shapes and Layout}，详细介绍了 `Shape` 协议缓冲区。
  * @{$tfcompile$Using AOT compilation}，本文解释了 `tfcompile` 这一独立工具，它可以将 TensorFlow 图编译为可执行代码以优化性能。

最后，我们还提供了这份指南：

  * @{$quantization$How to Quantize Neural Networks with TensorFlow}，本文介绍了如何使用量化来减少模型占用的存储空间与运行内存。量化可以改善性能，在移动设备上效果尤为明显。

