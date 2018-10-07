# 为 XLA 开发一个新后端

本初级指南针对的是早期用户，他们希望用一种高效的方式轻易地将 TensorFlow 重定向到他们自己的硬件上。本指南不会手把手地讲解，我们假设读者已经了解 [LLVM](http://llvm.org)、[Bazel](https://bazel.build/)、以及 TensorFlow。

XLA 提供了一个抽象的接口，让新的架构或加速器可以实现并创建可运行 TensorFlow 计算图的后端。在新的硬件上重定向 XLA 要比重新实现 TensorFlow 所有的操作更简单、更具扩展性。

大部分实现都属于下列情形之一：

  1. 已有的 CPU 架构尚没有在官方 XLA 中得到支持，存在或不存在已有的 [LLVM](http://llvm.org) 后端。
  2. 非 CPU 硬件，已有 LLVM 后端。
  3. 非 CPU 硬件，没有 LLVM 后端。

> 注意：LLVM 后端既包括官方发布的 LLVM 后端，也包括内部开发的定制 LLVM 后端。

## 场景 1：已有的 CPU 架构，但官方 XLA 尚不支持

在此场景中，先查看一下已有的 [XLA CPU 后端](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/)。通过 LLVM，XLA 让 TensorFlow 更容易重定向到不同的 CPU，因为对于 CPU 来说， XLA 后端之间的主要区别在于 LLVM 生成的代码。Google 在 x64 和 ARM64 架构下测试了 XLA。

如果硬件厂商已经为他们的硬件开发了 LLVM 后端，将此后端与用 XLA 构建的 LLVM 链接起来是比较容易的。在 JIT 模式中，XLA CPU 后端为主机 CPU 生成代码。对于提前（ahead-of-time）编译，[`xla::AotCompilationOptions`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h) 可提供一个 LLVM 三元组，来配置目标架构。

如果尚不存在 LLVM 后端，但是存在另一种代码生成器，重用已有 CPU 后端的代码也是可能的。

## 场景 2：非 CPU 硬件，已有 LLVM 后端

在已有的 [`xla::CPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/cpu/cpu_compiler.cc) 和 [`xla::GPUCompiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/nvptx_compiler.cc) 类的基础上，是有可能建模出一个新的 [`xla::Compiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h) 实现的。因为它们已经生成了 LLVM IR。
根据硬件的不同，有可能许多 LLVM IR 生成的过程也会不同，但是很多代码是可以和已有的后端共享的。

参考的一个很好的例子是 XLA 的 [GPU 后端](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/gpu/)。这个 GPU 后端针对的非 CPU 的 ISA，因而它的代码生成的很多方面是 GPU 领域专有的。其它类型的硬件，比如，DSP 一类的 Hexagon （它们是有上游的 LLVM 后端的），可以重用 LLVM IR 生成逻辑的部分代码，但是其它部分则是不一样的。


## 场景 3：非 CPU 硬件，没有 LLVM 后端

如果没有可能利用 LLVM，则最好的选择是为你的硬件及 XLA 实现一个全新的后端。这是最难的选择，因为你要实现如下的类：

*   [`StreamExecutor`](https://www.tensorflow.org/code/tensorflow/stream_executor/stream_executor.h)：对于许多设备而言，不是 `StreamExecutor` 中的所有方法都是必要的。更多细节，参见已有 `StreamExecutor` 的实现。
*   [`xla::Compiler`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/compiler.h)：这个类封装了一个 HLO 计算到一个 `xla::Executable` 的编译。
*   [`xla::Executable`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/executable.h)：这个类用于在目标平台上启动一个编译后的计算过程。
*   [`xla::TransferManager`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/transfer_manager.h)：这个类让后端提供针对目标平台的机制，用于从设备内存句柄构造出 XLA 字面量数据。换句话说，它帮助封装了主 CPU 与设备之间的数据传输。
