# TensorFlow 架构

为了针对大规模分布式训练和推理，我们设计了 TensorFlow，同时它也足够灵活，可以支持新的机器学习模型及系统级的优化。

本文描述了使这种规模和灵活性组合成为可能的系统体系结构。我们假设你在阅读本文时已经熟悉了使用 TensorFlow 的基本概念，例如计算图（computation graph）、操作（operation）以及会话（session）。请参考[这个文档](../guide/low_level_intro.md)来了解关于这些主题的介绍，熟悉[分布式 TensorFlow](../deploy/distributed.md) 对理解本文会有帮助。

本文适用于那些受到当前 API 制约从而希望以某种方式扩展 TensorFlow 的开发者、希望优化 TensorFlow 的硬件工程师、大规模分布式机器学习系统的实现人员以及任何希望了解 TensorFlow 底层机制的人。在本文档的结尾部分，你应该了解 TensorFlow 体系架构，以便能够阅读和修改 TensorFlow 的核心代码。

## 概述

TensorFlow 运行时是一个跨平台库。图 1 展示了其总体框架。通过一套 C API 将核心运行时与不同语言的用户级代码分离开来。

![TensorFlow 架构层级](https://www.tensorflow.org/images/layers.png)

**图 1**

本文只关注如下几层：

* **客户端（Client）**：
  * 将计算过程定义为数据流图。
  * 使用 [**`Session`**](https://www.tensorflow.org/code/tensorflow/python/client/session.py) 初始化数据流图的执行
* **分布式主控端（Master）**
  * 修剪图中的某些特殊子图，即 `Session.run()` 中所定义的参数。
  * 将子图划分为在不同进程和设备中运行的多个部分。
  * 将图分发给不同的工作进程。
  * 由工作进程初始化子图的计算。
* **工作进程（Worker service）（每个任务的）**
  * 使用内核实现调度图操作并在合适的硬件（CPU、GPU 等）执行。
  * 向其他工作进程发送或从其接收操作的结果。
* **内核实现**
  * 执行一个独立的图操作计算。

图 2 展示了这些组件之间的交互。`/job:worker/task:0` 和 `/job:ps/task:0` 均为具有工作进程的任务。`PS` 代表参数服务器：负责存储和更新模型参数的任务。其他任务在优化参数时会向这些参数发送更新。任务之间的这种特定分工并不是必须的，但是这在分布式训练中很常用。

![TensorFlow 架构图示](https://www.tensorflow.org/images/diag1.svg)

**图 2**

注意，分布式主控端和工作进程仅存在于分布式 TensorFlow 中。单进程版的 TensorFlow 使用了一种特殊的 Session 实现，与分布式主控端的工作完全一样，不过它只与本地进程中的设备通信。

下面各小节更详细地描述了 TensorFlow 核心层，并以一个示例图来展示其处理步骤。

## 客户端

用户负责编写用于构建计算图的 TensorFlow 客户端程序。这个程序可以直接组成独立的操作或者使用类似于 Estimator API 之类的库组建神经网络层和其他更高层次的抽象。TensorFlow 支持多种客户端语言，我们优先考虑 Python 和 C++，因为我们内部的用户最熟悉这些语言。随着其特性的逐渐稳定与完善，我们通常将它们移植到 C++，这样用户就能够从所有客户端语言中使用优化后的版本了。大部分训练库仍然是 Python，但是 C++ 则是作为更高效率的接口来提供给模型部署后的推断。

客户端创建一个 `Session`，这个 `Session` 会将数据流图的定义根据 `tf.GraphDef` 的缓存协议发送给分布式主控端。当客户端对图中一个或多个节点求值时，会触发分布式主控端初始化计算。

在图 3 中，客户端构建了一个图，其权值（w）与特征向量（x）相乘，然后将其与偏置（b）相加，并最后将结果保存在变量（s）中。

![TensorFlow 架构图示：客户端](https://www.tensorflow.org/images/graph_client.svg)

**图 3**

### 代码

* `tf.Session`

## 分布式主控端

分布式主控端的主要职能有:

* 修剪数据流图，从而获得并发送客户端所需的子图节点；
* 将数据流图为不同的参与设备分配不同的计算子图
* 将计算子图缓存，以便后续复用

由于主控端了解在每一步计算中的整个计算过程，它首先使用了诸如公共子表达式消除、常量拆叠等标准优化方法对计算子图进行优化，然后再负责协调优化后的子图去执行一系列任务。

![TensorFlow 架构图示：Master](https://www.tensorflow.org/images/graph_master_cln.svg)

**图 4**

图 5 展示了一个示例图可能的划分。分布式主控端已将模型的参数分组，以便于将它们存储在参数服务器上。

![划分图](https://www.tensorflow.org/images/graph_split1.svg)

**图 5**

当图的边被分区所切断时，分布式 Master 则会介入并在接受和发送节点间传递任务信息（如图 6）。

![划分图](https://www.tensorflow.org/images/graph_split2.svg)

**图 6**

然后，分布式 Master 会将子图分配给分布式任务。

![分区图](https://www.tensorflow.org/images/graph_workers_cln.svg)

**图 7**

### 代码

* [MasterService API 定义](https://www.tensorflow.org/code/tensorflow/core/protobuf/master_service.proto)
* [Master 的通信接口](https://www.tensorflow.org/code/tensorflow/core/distributed_runtime/master_interface.h)

## 工作进程

每个任务中的工作进程负责：

* 处理来自主控端的请求
* 调度由本地子图组成操作的内核的执行，以及
* 协调任务之间的直接通信

我们优化工作进程来保证以最低的开销来运行大型计算图。我们目前的实现能够在每秒执行数万个子图，这使得大量的副本能够进行快速、细粒度的训练。工作进程将内核分配给本地设备并在可能的情况下并行执行，例如通过使用多个 CPU 内核或 GPU 流。

我们为每对源和目的设备类型定制了 `Send` 和 `Recv` 操作：

* 使用 `cudaMemcpyAsync()` 在本地 CPU 和 GPU 设备间传送数据，从而让计算与数据传输重叠。
* 两个本地 GPU 之间使用点对点 DMA 传输，以避免通过主机 CPU 进行昂贵的复制。

对于任务之间的传输，TensorFlow 使用了多种协议，包括:

* TCP 上的 gRPC
* Converged Ethernet 上的 RDMA

我们也为用于多 GPU 通信的 Nvidia NCCL 库提供了初步支持（见 [`tf.contrib.nccl`](
https://www.tensorflow.org/code/tensorflow/contrib/nccl/python/ops/nccl_ops.py)）。

![分区图](https://www.tensorflow.org/images/graph_send_recv.svg)

**图 8**

### 代码

* [WorkerService API 定义](https://www.tensorflow.org/code/tensorflow/core/protobuf/worker_service.proto)
* [Worker 的通信接口](https://www.tensorflow.org/code/tensorflow/core/distributed_runtime/worker_interface.h)
* [远程 rendezvous (Send 和 Recv 的实现)](https://www.tensorflow.org/code/tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h)

## 内核实现

运行时包含 200 多个标准操作，包括数学、数组操作、控制流和状态管理等操作。每个操作均针对不同设备提供了内核级优化。许多操作内核由 `Eigen::Tensor` 实现，它使用 C++ 模板为多核 CPU 和 GPU 生成高效的并行代码。然而，我们大量的使用了类似于 cuDNN 的库使得一个更加高效的内核实现成为可能。我们还实现了[量化](../performance/quantization.md)，它在移动设备和高吞吐量数据中心应用程序等环境中实现了更快的推断，并使用 [gemmlowp](https://github.com/google/gemmlowp) 低精度矩阵库来加速量化计算。

如果将一个子计算分解为一些操作比较困难或低效时，用户可以使用 C++ 来注册并提供一个更高效的内核实现。例如我们建议注册自己的内核来完成某些与性能直接挂钩的操作（如 ReLU 和 Sigmoid 激活函数及其对应梯度）。[XLA 编译器](../performance/xla/index.md)提供了一个自动内核融合的实验性实现。

### 代码

*   [`OpKernel` 接口](https://www.tensorflow.org/code/tensorflow/core/framework/op_kernel.h)
