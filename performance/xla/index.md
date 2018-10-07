# XLA 概述

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:50%" src="https://www.tensorflow.org/images/xlalogo.png">
</div>

> 注意： XLA 是实验性的，仍处于 alpha 版本。大部分用例都看不到性能（提高速度或减少内存使用量）上的提高。我们已经发布了 XLA，这样的话，开源社区就可以为它的开发贡献力量了，而且有助于走出一条与硬件加速器整合之路。

XLA （加速线性代数）是一种线性代数的专用编译器，可用于优化 TensorFlow 的计算。其目标旨在提高速度、内存使用量以及对服务器和移动平台的可移植性。最初，大部分用户将不会从 XLA 中得到太大的好处，但是我们欢迎大家通过[即时（JIT）编译](../../performance/xla/jit.md)或[提前（AOT）编译](../../performance/xla/tfcompile.md)来使用 XLA 做实验。特别是那些专注于新硬件加速器的开发者，尤其应该试一试 XLA。

XLA 框架是实验性的，且处于活跃的开发状态。特别是，虽然已有操作的语义不太可能发生改变，但 XLA 不同，可以想见 XLA 中会不断加入更多操作，以覆盖更多重要的用例。XLA 的开发团队欢迎来自于社区的任何反馈，包括缺失的功能，以及通过 GitHub 提交的社区贡献。

## 我们为什么推出 XLA？

让 TensorFlow 用上 XLA，我们追求多个目标：

*   **改进执行速度**：对子图进行编译，以减少短时操作的执行时间，进而消除 TensorFlow 运行时相关的开销；融合管道化的操作以减少内存开销；针对已知张量形状优化，以支持更积极的常数传播。

*   **改进内存使用**：分析并调度内存使用，原则上可消除很多临时的缓存。

*   **减少对定制操作的依赖**：通过提高底层操作自动融合的性能，让其和定制操作中的手工融合一样高效，从而消除很多定制操作的必要性。

*   **减少移动足迹（mobile footprint）**：提前编译子图，并生成一对文件（对象/头文件），它们可以直接编译到另一个应用程序中，从而消除 TensorFlow 运行时。这样做的结果是移动推理（mobile inference）的足迹会减少数个数量级。

*   **改善可移植性**：为新硬件编写新的后端会相对容易一些，因为大部分 TensorFlow 程序不需要怎么修改就可以在新硬件上跑了。这和专门为新硬件定制一体化操作形成对比，因为那样做的话 TensorFlow 程序需要重写才能用这些新的操作。

## XLA 是如何工作的？

XLA 的输入语言称为 “HLO IR”，或简称 HLO （高层优化器）。HLO 的语义在[语义操作](../../performance/xla/operation_semantics.md)页面中有描述。理解 HLO 最方便的方式是将其视为一个 [编译器 IR](https://en.wikipedia.org/wiki/Intermediate_representation)。

XLA 接收 HLO 中定义的计算图，然后将它们编译成不同架构的机器指令。XLA 是模块化的，即在[针对一些新颖的硬件架构](../../performance/xla/developing_new_backend.md)时，XLA 易于接入新的硬件后端。x64 和 ARM64 的 CPU 后端，以及 NVIDIA GPU 后端已经包含在 TensorFlow 的源码树中了。

下面的流程图展示了 XLA 的编译过程：

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img src="https://www.tensorflow.org/images/how-does-xla-work.png">
</div>

XLA 中有一些优化和分析是目标无关的，例如[CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination)，目标无关的操作融合，以及计算时分配运行时内存的缓冲区分析。

在目标无关步骤之后，XLA 将 HLO 计算发送到后端。后端可以执行进一步的 HLO 级优化，这次优化会将目标特定的信息和需求考虑在内。例如，XLA GPU 后端可以对 GPU 编程模型进行具体的操作融合，并决定如何将计算划分为流。在这个阶段，后端还可能与某些操作或组合匹配从而优化库调用。

下一步就是针对特定目标的代码生成了。结合了 XLA的 CPU 和 GPU 后端使用 [LLVM](http://llvm.org) 来处理底层 IR、优化和代码生成。这些后端产生必要的 LLVM IR，用一种高效的方式来表示 XLA HLO 计算，然后调用 LLVM 从这个 LLVM IR 生成本地代码。

这个 GPU 后端目前通过 LLVM NVPTX 后端来支持 NVIDIA GPU；而 CPU 后端支持多种 CPU 指令集架构（ISA）。

## 支持的平台

XLA 目前支持 x86-64 和 NVIDIA GPU 上的[即时（JIT）编译](../../performance/xla/jit.md)；且在 x86-64 和 ARM 上支持[提前（AOT）编译](../../performance/xla/tfcompile.md)。

