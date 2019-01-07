# 性能

性能是训练机器学习模型时一个十分值得关注的问题。性能让研究速度加快并且使研究规模化，同时还为最终用户提供近乎即时的预测。这部分提供了利用高级 API 构建和训练高性能模型最佳实践的详细信息，以及用于推断最小延迟和最高吞吐量的量化模型。

* [性能指南](../performance/performance_guide.md) 包括一系列优化你的 TensorFlow 代码的最佳实践。
* [数据输入管道指南](../performance/datasets_performance.md) 描述了构建有效的 TensorFlow 数据输入管道的 tf.data API。
* [基准](../performance/benchmarks.md) 包括一系列针对不同种类硬件配置的基准结果。
* 针对优化 GPU 的推断，请参考[集成 TensorFlow 的 NVIDIA TensorRT™。](https://medium.com/tensorflow/speed-up-tensorflow-inference-on-gpus-with-tensorrt-13b49f3db3fa)

Tensorflow 模型优化工具包是一组用于推理优化模型的技术：

* [概述](../performance/model_optimization.md)，介绍了模型优化工具包。
* [训练后量化](../performance/post_training_quantization.md)，描述了训练后量化。

XLA（加速线性代数）是一个用于优化 TensorFlow 线性代数计算的试验编译器。下面的教程对 XLA 进行了详细的阐述：

* [XLA 概述](../performance/xla/index.md)，介绍加速线性代数 XLA。
* [广播语义](../performance/xla/broadcasting.md)，描述了 XLA 的广播语义。
* [开发一个全新 XLA 后端](../performance/xla/developing_new_backend.md)，这解释了如何重新定位 TensorFlow 以优化特定硬件的计算图的性能。
* [使用 JIT 编译](../performance/xla/jit.md)，它描述了 XLA JIT 编译器，它通过 XLA 编译和运行部分 TensorFlow 图，以优化性能。
* [操作语义](../performance/xla/operation_semantics.md)，这是一个参考手册，描述了 `ComputationBuilder` 接口中操作的语义。
* [形状和布局](../performance/xla/shapes.md)，详细介绍了 `Shape` 协议的缓存。
* [使用 AOT 编译](../performance/xla/tfcompile.md)，这解释了 `tfcompile`，这是一个独立的工具，可以将 TensorFlow 图编译成可执行代码，以优化性能。
