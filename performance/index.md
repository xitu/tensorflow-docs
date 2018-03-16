# 性能

在训练机器学习模型时，性能往往是一个值得关注的问题。本节介绍了数种优化性能的方法。使用 @{$performance_guide$Performance Guide} 开始您的研究，然后深入了解 @{$performance_models$High-Performance Models} 中详细介绍的技术：

  * @{$performance_guide$Performance Guide}，本文囊括了一系列优化 TensorFlow 代码的最佳实践。

  * @{$performance_models$High-Performance Models}，本文包括了一系列针对不同类型系统和网络拓扑设计高伸缩模型的进阶技术。

  * @{$performance/benchmarks$Benchmarks}，包含了一些基准测试的结果。

XLA（加速线性代数）是一个正处于实验阶段、用于优化 TensorFlow 计算的编译器。您可以阅读以下指南探索 XLA：

  * @{$xla$XLA Overview}：XLA 简介。
  * @{$broadcasting$Broadcasting Semantics}，本文介绍了 XLA 的广播语义。
  * @{$developing_new_backend$Developing a new back end for XLA}，本文解释了如何重定位 TensorFlow 以优化特定硬件的计算图的性能。
  * @{$jit$Using JIT Compilation}，本文描述了通过 XLA 编译和运行部分 TensorFlow 图，以优化性能的 XLA JIT 编译器。
  * @{$operation_semantics$Operation Semantics}，本文为描述 `ComputationBuilder` 接口的操作语义参考手册。
  * @{$shapes$Shapes and Layout}，详细介绍了 `Shape` 协议缓冲区。
  * @{$tfcompile$Using AOT compilation}，本文解释了 `tfcompile` 这一独立工具，它可以将 TensorFlow 图编译为可执行代码以优化性能。

最后，我们还提供了这份指南：

  * @{$quantization$How to Quantize Neural Networks with TensorFlow}，本文介绍了如何使用量化来减少存储和运行时的模型大小。量化可以改善性能，在移动设备上效果尤为明显。

