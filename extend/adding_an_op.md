# 添加一个新操作（Op）

注意：默认情况下 [www.tensorflow.org](https://tensorflow.org) 显示最新稳定版本的文档。本文档中的说明需要从源代码构建。你很可能想要从 TensorFlow 的 `master` 版本开始构建。那么，你就应该遵循[本文档的 `master` 版本](https://www.tensorflow.org/versions/master/extend/adding_an_op)，以防发生任何更改。

如果你想要创建一个在已有 TensorFlow 库中不存在的操作，我们建议你先从 Python 入手，即写一个已有 Python 操作或函数的复合操作。如果这样不可行，你可以定制一个 C++ 操作。下面是你可能需要这样做的一些理由：

*   将你的操作表示成现有操作的组合不太容易或不可能。
*   已有基本操作的组合操作效率不高。
*   你想手工实现一些基本操作的组合，因为未来的编译器做这种融合可能会比较困难。

例如，想象一下，你想实现类似于“最大值池化（MaxPool）”的“中值池化”操作，只不过不再是计算最大值，而是在滑动窗口上计算中值。这种操作是可能通过操作组合实现的，比如使用 ExtractImagePatches 和 TopK，但是这可能在性能上、或内存开销上不如原生操作，因为你可以在单一的融合操作中采用一些高明的策略。大体上，首先尝试用组合操作来实现你的想法总是值得一试的，只有当组合操作很困难或低效时才考虑添加一个新的操作。

为了加入一个定制操作，你需要：

1.  在 C++ 文件中注册这个新操作。操作的注册为此操作的功能定义了一个接口（规范）。比如，操作的注册定义了此操作的名称和它的输入输出。它还定义了 shape 函数，用于获取张量的形状。
2.  在 C++ 中实现这个操作。操作的实现称为内核，它是你在步骤 1 中注册的规范的具体实现。对于不同的输入输出类型或架构（比如不同的 CPUs 或 GPUs），可能有多个内核。
3.  创建一个 Python 包装器（可选）。这个包装器是用于在 Python 中创建操作的公共 API。操作的注册可以产生一个默认的包装器，它可以直接使用，或添加。
4.  为操作编写一个函数来计算梯度（可选）。
5.  测试操作。为方便起见，我们通常在 Python 中测试，但你也可以在 C++ 中测试。如果你定义了梯度，你可以使用 Python `tf.test.compute_gradient_error` 来验证。参见脚本 [`relu_op_test.py`](https://www.tensorflow.org/code/tensorflow/python/kernel_tests/relu_op_test.py)，它提供了一个例子，展示如何测试类似于 Relu 的算子的前向函数及梯度。

编写新操作代码前，你需要：

*   熟悉 C++ 。
*   必须已安装 [TensorFlow 二进制文件](../install)，或必须 [下载有 TensorFlow 源文件](../install/source.md)，并能够进行构建。

[TOC]

## 定义操作接口

操作接口的定义是通过在 TensorFlow 系统中注册来实现的。在此注册过程中，需要指定操作名称、输入（类型和名称）和输出（类型和名称），以及文档字符串和此操作要求的任何[属性](#属性)。

下面展示注册的具体过程。假设你想创建一个操作，其输入是一个 `int32` 类型的张量，而输出是此张量的一个副本，副本除第一个元素设为零之外其它都不变。为此，创建一个名为 `zero_out.cc` 的文件。然后调用 `REGISTER_OP` 宏，以定义你的操作：

```c++
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

于是，我们注册了一个名为 `ZeroOut` 的操作，它的输入（命名为 `to_zero`）和输出（命名为 `zeroed`）都是 32 位整数类型的张量。此操作利用一个 shape 函数来确保输出张量与输入张量保持一致。比如，如果输入张量为 [10, 20]，则此 shape 函数将输出张量也指定为 [10, 20]。

>   关于命名的备注：操作名称必须首字母大写，而且不能和库中已经注册的其它操作重名。

## 实现操作的内核

定义接口后，接下来就需要为此操作提供一个或多个内核实现了。为了实现这些内核，创建一个继承自 `OpKernel` 的类，并重载 `Compute` 方法。`Compute` 方法有一个类型为 `OpKernelContext*` 的参数 `context`，从中可以访问输入和输出张量等有用的信息。

将你的内核加到上面创建的文件中。这个内核的代码形如：

```c++
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 得到输入张量
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<int32>();

    // 创建输出张量
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // 除第一个元素外，输出张量的其它所有元素都设置为 0 
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output_flat(i) = 0;
    }

    // 如果可能的话，保留第一个输入值
    if (N > 0) output_flat(0) = input(0);
  }
};
```

实现完内核之后，将其注册到 TensorFlow 系统中。在注册中，你还要指定此内核运行时的不同约束条件。比如，你可能有一个内核是针对 CPU 的，而还有一个是针对 GPU 的。

为了给 `ZeroOut` 操作加上约束条件，将下面的代码加到 `zero_out.cc` 文件中：

```c++
REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
```

> 重要提示：你的 OpKernel 实例有可能会被并发访问，所以 `Compute` 方法必须是线程安全的。可以用线程互斥锁来保护类成员的每一次访问。更好的办法是，不要通过类成员来共享状态！可以考虑使用一个 [`ResourceMgr`](https://www.tensorflow.org/code/tensorflow/core/framework/resource_mgr.h)来跟踪操作的状态。

### 多线程 CPU 内核

为了编写一个多线程 CPU 内核，可使用 [`work_sharder.h`](https://www.tensorflow.org/code/tensorflow/core/util/work_sharder.h) 中的 Shard 函数。在 intra-op 线程模式下，此函数将计算函数分片到各个线程执行（参见 [`config.proto`](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto) 中定义的 intra_op_parallelism_threads 模式）。

### GPU 内核

一个 GPU 内核的实现包括两个部分：OpKernel 子类、CUDA 内核及其启动代码。

有时候 OpKernel 实现可由 CPU 和 GPU 内核共享，这一部分代码可以完成诸如检查输入和分配输出之类的任务。如果采用这种方案，则我们建议用如下实现方式：

1. 在设备上定义模板化的 OpKernel，并定义张量的基本类型
2. 为了完成输出的实际计算， Compute 函数要调用一个模板化的函子结构
3. 此函子针对 CPU 设备（CPUDevice）的特性化可在同一个文件中定义，但针对 GPU 设备（GPUDevice）的特性化要单独定义在一个 .cu.cc 文件中，因为它需要用 CUDA 编译器来编译。

下面是一个实现的示例：

```c++
// kernel_example.h
#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

template <typename Device, typename T>
struct ExampleFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename Eigen::GpuDevice, typename T>
struct ExampleFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif KERNEL_EXAMPLE_H_
```

```c++
// kernel_example.cc
#include "example.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// 实际计算的 CPU 模板特性化
template <typename T>
struct ExampleFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, int size, const T* in, T* out) {
    for (int i = 0; i < size; ++i) {
      out[i] = 2 * in[i];
    }
  }
};

// OpKernel 子类的定义
// 模板参数 <T> 为张量的数据类型
template <typename Device, typename T>
class ExampleOp : public OpKernel {
 public:
  explicit ExampleOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // 获得输入张量
    const Tensor& input_tensor = context->input(0);

    // 创建输出张量
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));

    // 执行计算
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    ExampleFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        output_tensor->flat<T>().data());
  }
};

// 注册 CPU 内核
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ExampleOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// 注册 GPU 内核
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* 在 kernel_example.cu.cc 中显式声明模板实例化 */ \
  extern template ExampleFunctor<GPUDevice, T>;              \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Example").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ExampleOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
```

```c++
// kernel_example.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "example.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// 定义 CUDA 内核
template <typename T>
__global__ void ExampleCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = 2 * ldg(in + i);
  }
}

// 定义启动 CUDA 内核的 GPU 实现
template <typename T>
void ExampleFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int size, const T* in, T* out) {
  // 启动 CUDA 内核
  //
  // 参见 core/util/cuda_kernel_helper.h 中的计算线程块数目和每块线程数（thread_per_block）的示例
  int block_count = 1024;
  int thread_per_block = 20;
  ExampleCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
}

// 显式实例化函子，这些函子用于处理注册的那些 OpKernel 支持的类型
template struct ExampleFunctor<GPUDevice, float>;
template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA
```

## 构建操作的库文件
### 用系统编译器来编译操作（TensorFlow 二进制安装）

你可以用 `C++` 编译器来编译 `zero_out.cc`，比如你的系统上的 `g++` 或 `clang` 都是可以的。用 PIP 包管理器来安装二进制 TensorFlow 时，已经包含了编译操作所需的头文件和库文件，具体的安装目录则取决于你的操作系统。不过，TensorFlow 的 python 库提供了 `get_include` 函数来获得头文件目录，也提供了 `get_lib` 函数来获得链接所需库文件的目录位置。下面是 Ubuntu 机器上这两个函数的输出结果：

```bash
$ python
>>> import tensorflow as tf
>>> tf.sysconfig.get_include()
'/usr/local/lib/python2.7/site-packages/tensorflow/include'
>>> tf.sysconfig.get_lib()
'/usr/local/lib/python2.7/site-packages/tensorflow'
```

假如你的系统上安装了 `g++`，下面的命令可于将你的操作编译成一个动态库。

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
```

在 Mac OS X 上，构建 `.so` 文件时还需要额外的编译标志 "-undefined dynamic_lookup" 。

>   注意，如果 `gcc` 版本 `>=5`，则 gcc 使用的新的 C++ [ABI](https://gcc.gnu.org/gcc-5/changes.html#libstdcxx)。TensorFlow 官网上提供的二进制 pip 包用的是 `gcc4` 构建的，即它用的是较早的 ABI。如果你用 `gcc>=5` 来编译你的操作库文件，在命令行中加入 `-D_GLIBCXX_USE_CXX11_ABI=0` 来让生成的库文件与旧的 ABI 兼容。此外，如果你使用从源码构建 TensorFlow ，记得在用 bazel 命令编译 Python 包时中加上编译选项 `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"`。

### 使用 bazel 编译操作（TensorFlow 源码安装）

如果你安装了 TensorFlow 源码，则你可以利用 TensorFLow 的构建系统来编译你的操作。把一个 BUILD 文件放在 [`tensorflow/core/user_ops`][user_ops] 目录中，其中包含 Bazel 的构建规则，内容如下：

```python
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)
```

运行下列命令来构建 `zero_out.so`.

```bash
$ bazel build --config opt //tensorflow/core/user_ops:zero_out.so
```

> 注意：虽然你可以用标准 `cc_library` 规则来生成一个共享库文件（`.so` 文件），我们还是强烈推荐使用 `tf_custom_op_library` 宏。这个宏加了一些必要的依赖项，而且还包含一些检查，以确保输出的共享库文件与 TensorFlow 的插件加载机制兼容。

## 在 Python 中使用新的操作

TensorFlow Python API 提供了 `tf.load_op_library` 函数来加载动态链接库，并将其注册到 TensorFlow 框架中。`load_op_library` 返回一个 Python 模块，其中就包含了你的新操作的 Python 包装器，以及它的内核。因而，一旦你构建完操作，你就可以按下面的方式中在 Python 中让它运行起来了：

```python
import tensorflow as tf
zero_out_module = tf.load_op_library('./zero_out.so')
with tf.Session(''):
  zero_out_module.zero_out([[1, 2], [3, 4]]).eval()

# 打印
array([[1, 0], [0, 0]], dtype=int32)
```

需要注意，生成的函数采用蛇形命令规则（snake\_case），这是为了遵守 [PEP8](https://www.python.org/dev/peps/pep-0008/) 规范。所以，如果你的操作在 C++ 代码中命名为 `ZeroOut`，则它的 Python 函数名会变成 `zero_out`。

为了让该操作可以像常规函数一样从某个模块中导入（`import`），则可以在 Python 源码中调用 `load_op_library` 函数：

```python
import tensorflow as tf

zero_out_module = tf.load_op_library('./zero_out.so')
zero_out = zero_out_module.zero_out
```

## 验证操作正常运行

确认你编写的操作是否可成功运行的一个好办法是写一个测试。创建文件 `zero_out_op_test.py`，内容如下：

```python
import tensorflow as tf

class ZeroOutTest(tf.test.TestCase):
  def testZeroOut(self):
    zero_out_module = tf.load_op_library('./zero_out.so')
    with self.test_session():
      result = zero_out_module.zero_out([5, 4, 3, 2, 1])
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])

if __name__ == "__main__":
  tf.test.main()
```

然后，运行该测试（假设你已经安装了 TensorFlow）：

```sh
$ python zero_out_op_test.py
```

## 在操作中加入高级功能

现在你已经知道如何实现和构建一个基本的操作（更恰当地说，是一个受限的操作），那么接下来，我们将介绍你在编写新操作时通常会用到的一些更复杂的功能，包括：

*   [条件检查和验证](#conditional-checks-and-validation)
*   [操作注册](#op-registration)
	 *    [属性](#attrs)
	 *    [属性类型](#attr-types)
	 *    [多态](#polymorphism)
	 *    [输入输出](#inputs-and-outputs)
	 *    [后向兼容](#backwards-compatibility)
*   [GPU 支持](#gpu_support)
	 *    [为 GPU 设备编译内核](#compiling-the-kernel-for-the-gpu-device)
*   [在 Python 中实现梯度计算](#implement-the-gradient-in-python)
*   [C++ 中的形状函数](#shape-functions-in-c)

### 条件检查和验证

上述示例假定操作适用于任意形状的张量。但如果我们只处理矢量呢？那么我们就需要在 OpKernel 的实现中加入一个检查：

```c++
  void Compute(OpKernelContext* context) override {
    // 获得输入张量
    const Tensor& input_tensor = context->input(0);

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("ZeroOut expects a 1-D vector."));
    // ...
  }
```

这里我们加了一个断言，它要求输入是一个矢量，否则将设置 `InvalidArgument` 状态。[`OP_REQUIRES` 宏][validation-macros] 有三个参数：

*   上下文 `context`：既可以是一个 `OpKernelContext`，也可以是一个 `OpKernelConstruction` 指针（参见 [`tensorflow/core/framework/op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op_kernel.h) 文件），用于其 `SetStatus()` 方法。
*   条件：关于验证张量形状的更多函数，参见文件 [`tensorflow/core/framework/tensor_shape.h`](https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.h)
*   错误本身：它由一个 `Status` 对象表示，参见文件 [`tensorflow/core/lib/core/status.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/status.h)。一个 `Status` 对象包含一个类型（常为 `InvalidArgument`，但能看到类型列表）和一条消息。构建一个错误的函数参见文件 [`tensorflow/core/lib/core/errors.h`][validation-macros]。

另外，如果你想测试从某个函数返回的 `Status` 对象是否为错误，则使用宏 [`OP_REQUIRES_OK`][validation-macros]。这两个宏都会在错误报错时返回错误对象。

### 操作的注册

#### 属性

操作可以有属性，当一个操作被加到计算图中时，它的属性就会被赋值。这些属性用于配置此操作，它们的值既可以在内核实现中访问，也可以在操作注册时的输入输出类型中进行访问。相较于输入，参数的使用要尽量避免，因为输入更为灵活一些。这是因为属性是常数，必须在计算图构造时定义。相反，输入作为张量，它的值是动态的；即输入的值在每一步都可以修改，比如使用 feed。属性主要用于无法使用输入的场合：任何影响特征（输入输出的数量和类型）的配置，或无法在每一步修改的时候。

你需要在注册操作时定义属性，定义时要指定名称和使用 `Attr` 方法的类型，此方法的参数规范如下：

```
<name>: <attr-type-expr>
```

其中 `<name>` 以字母开头，由数字、字母和下划线组成，而 `<attr-type-expr>` 一个类型表达式（参见[下方](#attr_types)）。

比如，如果你想让 `ZeroOut` 操作保留用户指定的索引，而不是仅保留第 0 个元素，你可以按下面的方式来注册操作：

```c++
REGISTER_OP("ZeroOut")
    .Attr("preserve_index: int")
    .Input("to_zero: int32")
    .Output("zeroed: int32");
```

（注意，[属性类型](#attr_types)与输入输出的 `tf.DType` 是不一样的。）

你实现的内核可以在构造函数中通过 `context` 参数来访问属性：

```c++
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {
    // 获取待保存的索引值
    OP_REQUIRES_OK(context,
                   context->GetAttr("preserve_index", &preserve_index_));
    // Check that preserve_index is positive
    OP_REQUIRES(context, preserve_index_ >= 0,
                errors::InvalidArgument("Need preserve_index >= 0, got ",
                                        preserve_index_));
  }
  void Compute(OpKernelContext* context) override {
    // ...
  }
 private:
  int preserve_index_;
};
```

还可以在 `Compute` 方法中使用这个参数：

```c++
  void Compute(OpKernelContext* context) override {
    // ...
    // 我们用保存的属性来检查动态输入的合法性
    // 所以，我们检查 preserve_index 是否在允许的值域范围内
    OP_REQUIRES(context, preserve_index_ < input.dimension(0),
                errors::InvalidArgument("preserve_index out of range"));

    // 将输出张量中所有元素设置为 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }
    // 保存指定位置的输入值
    output_flat(preserve_index_) = input(preserve_index_);
  }
```

#### 属性类型

属性支持下列数据类型：

* `string`：任意字节序列（不要求是 UTF8 编码）
* `int`：有符号整数
* `float`: 浮点数
* `bool`: True 或 false
* `type`： [`DataType`][DataTypeString] 的其中一个（非引用）值
* `shape`：一个 [`TensorShapeProto`][TensorShapeProto]
* `tensor`：一个 [`TensorProto`][TensorProto]
* `list(<type>)`： `<type>` 的列表，其中 `<type>` 为其中一种上述类型
  注意： `list(list(<type>))` 是非法的。

欲了解限定性列表，参见 [`op_def_builder.cc:FinalizeAttr`][FinalizeAttr]。

##### 默认值和约束

属性可以有默认值，有一些属性则还可以有约束。为了定义一个有约束的属性，可以使用下列属性类型表达式（`<attr-type-expr>`）：

* `{'<string1>', '<string2>'}`：表示在 `<string1>` 或 `<string2>` 这两种取值中二选一。当你使用这种语法时，系统自动推断出属性类型为 `string`。这相当于模仿构造了一个枚举：

```c++
REGISTER_OP("EnumExample")
    .Attr("e: {'apple', 'orange'}");
```

* `{<type1>, <type2>}`: 属性类型为 `type`，表示取值是 `<type1>` 类型或 `<type2>` 类型二者之一，其中 `<type1>` 和 `<type2>` 为两种 `tf.DType`。同样，你也不需要指定属性类型为 `type`，因为这个信息是可以从 `{...}` 这个张量类型列表推断出来的。比如，下面的例子中属性 `t` 必须是 `int32`、`float` 或 `bool` 中的一种类型：

```c++
REGISTER_OP("RestrictedTypeExample")
    .Attr("t: {int32, float, bool}");
```

* 常用的类型约束可以有如下别名：
	* `numbertype`：`type` 类型被限制为数值类型（不是字符串，也不是布尔类型）
	* `realnumbertype`：类似于 `numbertype` 类型，但不包括复数类型
	* `quantizedtype`：类型于 `numbertype` 类型，但只包括量化数值类型
    
    属性所支持的类型列表可通过 [`tensorflow/core/framework/types.h`](https://www.tensorflow.org/code/tensorflow/core/framework/types.h) 中的一些函数来定义（比如 `NumberTypes()`）。在本例中，属性 `t` 必须是下面一种数值类型：

    ```c++
    REGISTER_OP("NumberType")
        .Attr("t: numbertype");
    ```

    对于这个操作：

    ```python
    tf.number_type(t=tf.int32)  # 合法
    tf.number_type(t=tf.bool)   # 不合法
    ```

    列表可以和其他列表及单一类型组合。下面的操作允许属性 `t` 为任意数值类型或布尔类型：

    ```c++
    REGISTER_OP("NumberOrBooleanType")
        .Attr("t: {numbertype, bool}");
    ```

    对于这个操作：

    ```python
    tf.number_or_boolean_type(t=tf.int32)  # 合法
    tf.number_or_boolean_type(t=tf.bool)   # 合法
    tf.number_or_boolean_type(t=tf.string) # 不合法
    ```

* `int >= <n>`：取值必须是整型，且要求大于等于 `<n>`，其中 `<n>` 是一个自然数。

  比如，下列操作注册中，指定了属性 `a` 必须为一个至少为 `2` 的值：

  ```c++
  REGISTER_OP("MinIntExample")
      .Attr("a: int >= 2");
  ```

* `list(<type>) >= <n>`: 取值为`<type>` 类型的一个列表，其长度大于等于 `<n>`。

  比如，下列操作注册指定属性 `a` 是一个类型列表（要么是 `int32`，要么是 `float`），且要求长度大于等于 `3`：

  ```c++
  REGISTER_OP("TypeListExample")
      .Attr("a: list({int32, float}) >= 3");
  ```

为设置一个属性的默认值（让它在生成代码中成为可选项），可以在最后加上 `= <default>`，如下面代码所示：

```c++
REGISTER_OP("AttrDefaultExample")
    .Attr("i: int = 0");
```

这种默认值的支持语法正是计算图的 GraphDef 定义的协议缓存表达中所用的语法。

下面的示例展示如何为所有类型指定默认值：

```c++
REGISTER_OP("AttrDefaultExampleForAllTypes")
   .Attr("s: string = 'foo'")
   .Attr("i: int = 0")
   .Attr("f: float = 1.0")
   .Attr("b: bool = true")
   .Attr("ty: type = DT_INT32")
   .Attr("sh: shape = { dim { size: 1 } dim { size: 2 } }")
   .Attr("te: tensor = { dtype: DT_INT32 int_val: 5 }")
   .Attr("l_empty: list(int) = []")
   .Attr("l_int: list(int) = [2, 3, 5, 7]");
```

注意：若值类型为 `type`，则使用 `tf.DType`。

#### 多态

##### 类型多态性

有些操作支持不同类型的输入或产生不同类型的输出，这时你可以在此操作的注册中为[一个输入或输出类型](#输入和输出)指定[一个属性](#属性)。通常，你还要为支持的每种类型注册一个 `OpKernel`。

比如，如果你想让 `ZeroOut` 操作既支持 `int32` 数值类型的张量，还要支持 `float` 类型，那么此操作的注册过程将类似于：

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

现在，此操作在注册中指定了输入类型必须是 `float` 或 `int32`，而它的输出类型将保持一致，因为都是 `T` 类型。

> <a id="naming"></a> 关于命名的备注：输入、输出和属性一般都应该使用蛇形命名。
> 不过有一个例外情况，那就是属性被用作输入类型、或用于输入类型时。这样的属性会在操作被加入到计算图中自动推断出来，即它们不会在操作的函数中出现。比如，ZeroOut 最终的定义将产生一个如下的 Python 函数：
>
> ```python
> def zero_out(to_zero, name=None):
>   """...
>   参数：
>     to_zero: 表示一个 `Tensor`。必须是两种类型之一： `float32`、 `int32`。
>     name: 操作的名称（可选）
>
>   返回值：
>     一个 `Tensor`，与 `to_zero` 类型相同
>   """
> ```
>
> 如果 `to_zero` 中传入一个 `int32` 张量，则 `T` 自动被设置为 `int32` （实际上是 `DT_INT32`）。
> 这时推断出来的属性的命名方式为首字母大小或单词首字母大写。
>
> 与这种情况不同的是，有时候我们需要为用一个类型属性来为操作指定输出类型：
>
> ```c++
> REGISTER_OP("StringToNumber")
>     .Input("string_tensor: string")
>     .Output("output: out_type")
>     .Attr("out_type: {float, int32} = DT_FLOAT");
>     .Doc(R"doc(
> 将输入张量中的每个字符串转换为指定的数值类型。
> )doc");
> ```
>
> 这时，用户需要指定输出类型，如 Python 代码所示：
>
> ```python
> def string_to_number(string_tensor, out_type=None, name=None):
>   """将输入张量中的每个字符串转换为指定的数值类型。
>
>   参数：
>     string_tensor: `string` 类型的一个 `Tensor`
>     out_type: 可选的 `tf.DType`，即 `tf.float32` 和 `tf.int32` 二者之一，默认为 `tf.float32`。
>     name: 操作名称（可选）
>
>   返回值：
>     类型为 `out_type` 的一个 `Tensor`
>   """
> ```

```c++
#include "tensorflow/core/framework/op_kernel.h"
 class ZeroOutInt32Op : public OpKernel {
  // 和前面一样
};
 class ZeroOutFloatOp : public OpKernel {
 public:
  explicit ZeroOutFloatOp(OpKernelConstruction* context)
      : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // 获得输入张量
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
    
    // 产生输出张量
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->template flat<float>();
    
    // 将输出张量中的所有元素设置为 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }
    
    // 保留第一个输入值
    if (N > 0) output_flat(0) = input(0);
  }
};

// 注意：TypeConstraint<int32>("T") 表示属性 `T`（定义在操作注册代码中）必须是 `int32` 类型的，即将模板实例化了。
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("T"),
    ZeroOutOpInt32);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    ZeroOutFloatOp);
```

> 为了[后向兼容](#后向兼容)，在将属性加到已有操作中时，你需要指定一个[默认值](#默认值约束)：
>
> ```c++
> REGISTER_OP("ZeroOut")
>   .Attr("T: {float, int32} = DT_INT32")
>   .Input("to_zero: T")
>   .Output("zeroed: T")
> ```

如果你还想添加更多类型，比如说 `double` 类型，你要稍微修改一下注册代码：

```c++
REGISTER_OP("ZeroOut")
    .Attr("T: {float, double, int32}")
    .Input("to_zero: T")
    .Output("zeroed: T");
```

为了避免像上面的代码一样为多个 `OpKernel` 编写冗余代码，你可以使用 C++ 模板。不过，你仍然需要为每一次加载注册一个内核（调用 `REGISTER_KERNEL_BUILDER`）。

```c++
template <typename T>
class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {
    // 获得输入张量
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<T>();
    
    // 产生输出张量
    Tensor* output = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_tensor.shape(), &output));
    auto output_flat = output->template flat<T>();
    
    // 将输出张量中的所有元素设置为 0
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }
    
    // 保留第一个输入值
    if (N > 0) output_flat(0) = input(0);
  }
};

// 注意：TypeConstraint&lt;int32&gt;("T") 表示属性 `T` （定义在操作注册代码中）必须是 `int32` 类型的，即将模板实例化了。
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<int32>("T"),
    ZeroOutOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<float>("T"),
    ZeroOutOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("ZeroOut")
    .Device(DEVICE_CPU)
    .TypeConstraint<double>("T"),
    ZeroOutOp<double>);
```

如果加载次数还不少，那你可以将注册放入宏中。

```c++
#include "tensorflow/core/framework/op_kernel.h"

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
```

根据你为内核注册的类型列表的不同，你还可以使用 [`tensorflow/core/framework/register_types.h`][register_types] 中提供的宏：

```c++
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

REGISTER_OP("ZeroOut")
    .Attr("T: realnumbertype")
    .Input("to_zero: T")
    .Output("zeroed: T");

template <typename T>
class ZeroOutOp : public OpKernel { ... };

#define REGISTER_KERNEL(type)                                       \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("ZeroOut").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      ZeroOutOp<type>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
```

##### 列表作为输入输出

除了能够接受或产生不同类型之外，操作还消耗或产生数目不一的张量。

在下一个例子中，属性 `T` 保存了类型列表，并被用作输入 `in` 和输出 `out` 的类型。即输入和输出都是该类型的张量列表（并且输入输出张量的大小和类型都是完全一样的，因为它们都具有类型 `T`）。

```c++
REGISTER_OP("PolymorphicListExample")
    .Attr("T: list(type)")
    .Input("in: T")
    .Output("out: T");
```

你也可以对列表中元素的类型施加限制。在下一个例子中，输入是 `float` 或 `double` 类型张量的列表。比如，若输入类型是 `(float, double, float)`，而输出类型也必须是 `(float, double, float)`。

```c++
REGISTER_OP("ListTypeRestrictionExample")
    .Attr("T: list({float, double})")
    .Input("in: T")
    .Output("out: T");
```

如果你要求列表中所有张量的类型都相同，则你可以这样：

```c++
REGISTER_OP("IntListInputExample")
    .Attr("N: int")
    .Input("in: N * int32")
    .Output("out: int32");
```

此例中，输入是 `int32` 类型的张量的列表，其中 `int` 属性 `N` 用来指定此列表的长度。

我们也可以实现 [类型多态性](#类型多态性)。在下一个示例中，输入是长度为 `N` 的张量列表，这些张量的类型为 `T`（但还没指定），而输出则为指定类型的单个张量：

```c++
REGISTER_OP("SameListInputExample")
    .Attr("N: int")
    .Attr("T: type")
    .Input("in: N * T")
    .Output("out: T");
```

默认情况下，张量列表的长度至少为 1。你可以用 [相应属性上的 `">="` 约束](#默认值约束) 来修改默认值。在下一个示例中，输入是长度至少为 2 的 `int32` 张量列表：

```c++
REGISTER_OP("MinLengthIntListExample")
    .Attr("N: int >= 2")
    .Input("in: N * int32")
    .Output("out: int32");
```

同样的语法也可以用到 `"list(type)"` 类型的属性上：

```c++
REGISTER_OP("MinimumLengthPolymorphicListExample")
    .Attr("T: list(type) >= 3")
    .Input("in: T")
    .Output("out: T");
```

#### 输入和输出

下面对前面的示例做个总结，一个操作注册可以指定多个输入输出：

```c++
REGISTER_OP("MultipleInsAndOuts")
    .Input("y: int32")
    .Input("z: float")
    .Output("a: string")
    .Output("b: int32");
```

每个输入或输出的规范的格式如下：

```
<name>: <io-type-expr>
```

其中 `<name>` 以字母开头，可以由字母、数字和下划线组成。`<io-type-expr>` 是下列表达式之一：

* `<type>`：支持的输入类型，比如 `float`、`int32`、`string`。这个表达式指定了 `type` 类型的单个张量。

  参见 `tf.DType`。

  ```c++
  REGISTER_OP("BuiltInTypesExample")
      .Input("integers: int32")
      .Input("complex_numbers: complex64");
  ```

* `<attr-type>`：一个[属性](#属性)的名称，此属性的类型可以是 `type` 或 `list(type)`（可以有类型限制）。这个语法可以实现[多态操作](#多态)。 

  ```c++
  REGISTER_OP("PolymorphicSingleInput")
      .Attr("T: type")
      .Input("in: T");

  REGISTER_OP("RestrictedPolymorphicSingleInput")
      .Attr("T: {int32, int64}")
      .Input("in: T");
  ```

  引用类型为 `list(type)` 的属性可以让你接受一个张量序列。

  ```c++
  REGISTER_OP("ArbitraryTensorSequenceExample")
      .Attr("T: list(type)")
      .Input("in: T")
      .Output("out: T");

  REGISTER_OP("RestrictedTensorSequenceExample")
      .Attr("T: list({int32, int64})")
      .Input("in: T")
      .Output("out: T");
  ```

  注意，输出 `out` 中的张量的类型和数目与输入 `in` 是一样的，因为它们都是 `T` 类型。

* 相同类型的张量序列：`<number> * <type>`，其中 `<number>` 为类型为 `int` 的一个[属性](#属性)。`<type>` 可以是 `tf.DType` 或类型为 `type` 的一个属性的名称。第一种情况中，操作可接受 `int32` 张量的列表，示例如下：

  ```c++
  REGISTER_OP("Int32SequenceExample")
      .Attr("NumTensors: int")
      .Input("in: NumTensors * int32")
  ```

  此操作接受任意类型的张量列表，只要它们的类型都一样：

  ```c++
  REGISTER_OP("SameTypeSequenceExample")
      .Attr("NumTensors: int")
      .Attr("T: type")
      .Input("in: NumTensors * T")
  ```

* 对单个张量的引用：`Ref(<type>)`，其中 `<type>` 是上述类型中的一种。

> 关于命名的备注：输入的类型中用到的任何属性都会被推断出来。按惯例，这些被推断的属性名要首字线大写（比如 `T` 或 `N`）。其它情况下，输入、输出和属性的名称和函数参数命名方式一致，比如 `num_outputs`。更多细节，参考 [前面关于命名的备注](#命名)。

更多细节，参考 [`tensorflow/core/framework/op_def_builder.h`][op_def_builder]。

#### 后向兼容性

假设你已经编写了一个很好的定制操作，并分享给他人，让你的客户开心地使用了。然而，你还想要进一步修改这个操作。

一般情况下，对已有的已上线的规范进行修改需要考虑后向兼容性：对一个操作的规范进行修改必须保证由旧规范构造出来的序列化 `GraphDef` 协议缓存仍然能用。`GraphDef` 的兼容性的细节描述[参见这里](../guide/version_compat.md#compatibility of graphs and checkpoints)。

保持后向兼容性的方法有很多，下面列出了一些：

1. 添加到一个操作的任何新属性必须定义默认值，而在默认值下，此操作的行为必须与原来相同。要将操作从非多态转换为多态，你*必须*为新属性指定默认值，让它在默认情况下保持原来的行为。比如，如果你的操作为：

       REGISTER_OP("MyGeneralUnaryOp")
           .Input("in: float")
           .Output("out: float");

  你可以在保持后向兼容的情况下让它变得多态：

       REGISTER_OP("MyGeneralUnaryOp")
           .Input("in: T")
           .Output("out: T")
           .Attr("T: numerictype = DT_FLOAT");

2. 对于一个属性，你总是可以安全地施加更严格的约束。比如，你可以将 `{int32, int64}` 变成`{int32, int64, float}` 或 `type` 。你也可以将 `{"apple", "orange"}`变成`{"apple", "banana", "orange"}` 或 `string` 。

3. 你可以将单个输入/输出变成列表形式的输入/输出，前提是列表类型的默认值与原来的接口一致。

4. 你可以添加一个新的列表形式的输入/输出，只要它的默认值为空。

5. 将你创建新创建的任何操作放在命名空间中，即在操作前面加上前缀以区别于工程中的其它操作。这可以让你的操作避免与 TensorFlow 未来版本中新引入的操作相冲突。

6. 提前计划好！尝试构想此操作的未来用途。有些接口修改无法以兼容方式修改（比如，将相同类型列表变成变化类型列表）。

安全和不安全修改的完整列表可以在源码 [`tensorflow/core/framework/op_compatibility_test.cc`](https://www.tensorflow.org/code/tensorflow/core/framework/op_compatibility_test.cc) 中找到。如果你无法在兼容要求下修改此操作，那么最好是另起炉灶，创建一个新的操作，取一个新的名字，来表示你的新的语义。

还要注意的是，除了维持 `GraphDef` 的兼容性，生成的 Python 代码还是有可能变得与旧的调用它的代码不兼容。因而，为保持兼容性，Python API 的修改要非常小心，最好是手写 Python 包装代码，而且只在旧的接口函数的最后面加上新的可选参数。一般而言，不兼容的改变只会发生在 TensorFlow 的大的版本变动时，而且必须遵从 [`GraphDef` 的版本语义](../guide/version_compat.md#compatibility_of_graphs_and_checkpoints)。

### GPU 支持

你可以实现不同的内核操作（OpKernel），然后一个注册到 CPU 上，另一个注册到 GPU 上,就像你可以[为不同的类型注册内核](#多态)一样。TensorFlow 提供了多个支持 GPU 的内核的例子，参见源码 [`tensorflow/core/kernels`](https://www.tensorflow.org/code/tensorflow/core/kernels/)。注意，有些内核的 CPU 版本在一个 `.cc` 文件中，其 GPU 版本在一个 `_gpu.cu.cc` 文件中，它们共享的代码则在一个 `.h` 文件中。

比如，`tf.pad` 的 CPU 内核代码位于 [`tensorflow/core/kernels/pad_op.cc`](pad_op) 中，它的 GPU 内核在 [`tensorflow/core/kernels/pad_op_gpu.cu.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op_gpu.cu.cc) 中，而共享代码在 [`tensorflow/core/kernels/pad_op.h`](https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.h) 中。我们以这种方式组织代码有两个原因：它允许在 CPU 和 GPU 实现之间共享代码，并将 GPU 实现放入单独的文件中，以便只能由 GPU编译器编译。

值得注意的一点是，即使使用的是 `pad` 操作的 GPU 内核版本，它仍然需要用到 CPU 内存中的 `"paddings"` 输入。为标记这种 CPU 上的输入或输出，在内核注册时添加一个 `HostMemory()` 调用，比如：


```c++
#define REGISTER_GPU_KERNEL(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Pad")                  \
                              .Device(DEVICE_GPU)      \
                              .TypeConstraint<T>("T")  \
                              .HostMemory("paddings"), \
                          PadOp<GPUDevice, T>)
```

#### 为 GPU 设备编译内核

代码 [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc) 中给出了使用 CUDA 内核实现操作的一个例子。`tf_custom_op_library` 接受一个 `gpu_srcs` 参数，其中包含 CUDA 内核 (`*.cu.cc` 文件)的源文件列表。如果你使用的是 Tensorflow 的二进制安装，这些 CUDA 内核代码必须用 NVIDIA 的 `nvcc` 编译器进行编译。为了将 [cuda_op_kernel.cu.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cu.cc)和[cuda_op_kernel.cc](https://www.tensorflow.org/code/tensorflow/examples/adding_an_op/cuda_op_kernel.cc)这两个源码编译成一个动态加载库，你需要使用如下命令：


```bash
nvcc -std=c++11 -c -o cuda_op_kernel.cu.o cuda_op_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_op_kernel.so cuda_op_kernel.cc \
  cuda_op_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
```

通过 `tf.load_op_library` 函数，上述命令产生的 `cuda_op_kernel.so` 可以像通常的动态链接库一样在 Python 中加载。注意，如果 CUDA 库没有安装在 `/usr/local/lib64` 中，你需要在 上面第二个命令（g++）中显式指定其路径。比如，你的 CUDA 安装在 `/usr/local/cuda-8.0` 中，则需要在命令行中添加 `-L /usr/local/cuda-8.0/lib64/`。
> 注意，在某些 Linux 设置中，`nvcc` 编译步骤还需要其他选项。将 `-D_MWAITXINTRIN_H_INCLUDED` 添加到 nvcc 命令行以避免 `mwaitxintrin.h` 中的错误。

### 在 Python 中实现梯度计算

给定一个由操作构成的计算图，TensorFlow 使用自动微分（反向传播）来添加新的操作，用于表示已有的操作的梯度（参见 [梯度计算](../api_guides/python/train.md#Gradient_Computation)）。为了让新实现的操作也支持这种自动微分，你必须注册一个梯度函数，用于在给定关于此操作输出的梯度的情况下计算出关于此操作输入的梯度。

在数学上，如果一个操作计算 \\(y = f(x)\\)，为它注册的梯度操作将损失函数 \\(L\\) 关于 \\(y\\) 的梯度 \\(\partial L/ \partial y\\) 转化为关于 \\(x\\) 的梯度 \\(\partial L/ \partial x\\)，它使用的是链式法则：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial f}{\partial x}.$$

以 `ZeroOut` 为例，输入中只有一项会影响输出，所以关于输入的梯度是一个稀疏的 "one hot" 张量。代码如下：

```python
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):  
    """ `zero_out` 的梯度
    参数:    
      op: 待求微分的 `zero_out` 操作，通过它，我们可以找到原操作的输入输出。    
      grad: 关于 `zero_out` 操作的输出的梯度。
    返回值:    
      关于 `zero_out` 输入的梯度。  
    """  
    to_zero = op.inputs[0]  
    shape = array_ops.shape(to_zero)  
    index = array_ops.zeros_like(shape)  
    first_grad = array_ops.reshape(grad, [-1])[0]  
    to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)  
    return [to_zero_grad]  # 只有一个张量的列表，因为我们只有一个输入
```
用 `tf.RegisterGradient` 注册梯度函数的详情如下：

  * 对于只有一个输出的操作，梯度函数的参数为一个 `tf.Operation` `op`，和一个 `tf.Tensor` `grad`，然后它会根据张量 [`op.inputs[i]`](../../api_docs/python/framework.md#Operation.inputs)、[`op.outputs[i]`](../../api_docs/python/framework.md#Operation.outputs)、及 `grad` 来构建新操作。关于任何属性的信息可通过 `tf.Operation.get_attr` 来找到。

  * 如果操作有多个输出，其梯度函数的参数为 `op` 和 `grads`，其中 `grads` 是关于每个输出的梯度。此梯度函数的返回值为一个张量列表，表示的关于每个输入的梯度。
  * 如果对某个输入没有良定义的梯度，比如用作指标的整数输入，相应的梯度应该为 `None`。比如，一个操作的一个输入是浮点型张量 `x`，另一个输入是一个整数指标 `i`，则梯度函数应该返回 `[x_grad, None]`。
  * 如果一个操作根本就没有任何有意义的梯度，那么就没有必要注册梯度函数了。只要你不会用到操作的梯度，不注册也不会有什么问题。在有些情况下，一个操作没有良定义的梯度，但可能会参与到梯度计算中。在这种情况下，可以使用 `ops.NotDifferentiable` 来自动反向传播零值。

注意，调用梯度函数时，只能访问到操作的数据流图，而不是张量数据本身。因此，所有梯度计算都必须使用其它 TensorFlow 操作执行，以便在计算图执行时运行。


### C++ 中的形状函数

TensorFlow API 有一个功能叫做“形状推断”，可以无需执行计算图而获得张量的形状信息。形状推断是由“形状函数”来支撑的，每个操作类型都会在其 C++ `REGISTER_OP` 声明中注册形状函数，它们有两种作用：在计算图的构造函数中声明输入的形状是兼容的，为输出指定形状。

形状函数定义为 `shape_inference::InferenceContext` 类上的操作。比如，在 ZeroOut 的形状函数中：

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```

`c->set_output(0, c->input(0));` 声明第一个输出的形状必须为第一个输入的形状。如果输出是按上面示例中的索引选择的，则 `set_output` 的第二个参数应该是一个 `ShapeHandle` 对象。你可以通过默认构造函数来创建一个空的 `ShapeHandle` 对象。索引为 `idx` 的输入的 `ShapeHandle` 对象可通过 `c->input(idx)` 来获得。

TensorFlow 已经提供了大量的通用形状函数，可适用于许多操作，比如 `shape_inference::UnchangedShape` 可在源码 [common_shape_fns.h](https://www.tensorflow.org/code/tensorflow/core/framework/common_shape_fns.h) 中找到，其用法如下：

```c++
REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn(::tensorflow::shape_inference::UnchangedShape);
```

一个形状函数也可用于约束输入的形状。对于[具有矢量形状约束的 `ZeroOut` 版本](#validation)，其形状函数定义如下：

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input));
      c->set_output(0, input);
      return Status::OK();
    });
```

`WithRank` 函数验证输入形状 `c->input(0)` 是否有一个精确的维度（或者如果输入形状未知，则输出形状将是一个未知维度的向量）。

对于[具有多个输入的多态](#多态)操作，可以使用 `InferenceContext` 的成员函数来确定需要检查的形状数目，并用 `Merge` 成员函数来验证这些形状都是兼容的（或者用 `InferenceContext::GetAttr` 访问表示长度的属性，此函数可以访问操作的属性）。

```c++
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle input;
      ::tensorflow::shape_inference::ShapeHandle output;
      for (size_t i = 0; i < c->num_inputs(); ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 2, &input));
        TF_RETURN_IF_ERROR(c->Merge(output, input, &output));
      }
      c->set_output(0, output);
      return Status::OK();
    });
```

由于形状推断是可选特征，且张量的形状可能会动态改变，因此形状函数必须能够处理任意输入可能的形状信息不完整的情况。[`InferenceContext`](https://www.tensorflow.org/code/tensorflow/core/framework/shape_inference.h) 的 `Merge` 方法允许在两个形状信息不完整的情况下（至少有一个不完整）断言它们是相同的。TensorFlow 的所有核心操作都定义了形状函数，并提供了许多不同的用法示例。

`InferenceContext` 类中有很多可用于定义形状函数操作的函数。比如，你可以使用 `InferenceContext::Dim` 和 `InferenceContext::WithValue` 来验证一个特定的维度是否具有一个特定的值；我们还可以用 `InferenceContext::Add` 和 `InferenceContext::Multiply` 指定输出维度为两个输入维度的和 / 乘积。参见 `InferenceContex` 类的定义中所有可用的形状操作方法。下面的例子将第一个输出的形状设置为 (n,3)，将第一个输入的形状设置为 (n,...)。

```c++
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->Matrix(c->Dim(c->input(0), 0), 3));
    return Status::OK();
});
```

对于复杂的形状函数，应该考虑添加一个测试，来验证多个输入形状组合可产生预期的输出形状组合。这种测试的编写方法参见源码 [core ops tests](https://www.tensorflow.org/code/tensorflow/core/ops/array_ops_test.cc)。（`INFER_OK` 和 `INFER_ERROR` 的语法会让人感觉有点神秘，不过还是在测试中尽量让表示输入输出形状的规范简洁一些。目前，可以在已有的测试中看看注释，了解如何编写形状的规范。）

[core-array_ops]: https://www.tensorflow.org/code/tensorflow/core/ops/array_ops.cc
[python-user_ops]: https://www.tensorflow.org/code/tensorflow/python/user_ops/user_ops.py
[tf-kernels]: https://www.tensorflow.org/code/tensorflow/core/kernels/
[user_ops]: https://www.tensorflow.org/code/tensorflow/core/user_ops/
[pad_op]: https://www.tensorflow.org/code/tensorflow/core/kernels/pad_op.cc
[standard_ops-py]: https://www.tensorflow.org/code/tensorflow/python/ops/standard_ops.py
[standard_ops-cc]: https://www.tensorflow.org/code/tensorflow/cc/ops/standard_ops.h
[python-BUILD]: https://www.tensorflow.org/code/tensorflow/python/BUILD
[validation-macros]: https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h
[op_def_builder]: https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.h
[register_types]: https://www.tensorflow.org/code/tensorflow/core/framework/register_types.h
[FinalizeAttr]: https://www.tensorflow.org/code/tensorflow/core/framework/op_def_builder.cc
[DataTypeString]: https://www.tensorflow.org/code/tensorflow/core/framework/types.cc
[python-BUILD]: https://www.tensorflow.org/code/tensorflow/python/BUILD
[types-proto]: https://www.tensorflow.org/code/tensorflow/core/framework/types.proto
[TensorShapeProto]: https://www.tensorflow.org/code/tensorflow/core/framework/tensor_shape.proto
[TensorProto]: https://www.tensorflow.org/code/tensorflow/core/framework/tensor.proto
