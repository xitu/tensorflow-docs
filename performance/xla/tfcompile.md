# 使用提前编译

## 什么是 tfcompile？

`tfcompile` 是一个将 TensorFlow 图提前（AOT）编译成可执行代码的独立工具。它可以减少二进制文件的大小，同时避免一些运行时开销。`tfcompile` 一个典型的用途是将推理图编译成用于移动设备的可执行代码。

TensorFlow 图通常由 TensorFlow 运行时执行。这会导致图中的每个节点执行时的运行时开销。同时也增加了二进制文件的大小，因为除了图自身，TensorFlow 运行时的代码也需要可用。而由 `tfcompile` 生成的可执行代码不使用 TensorFlow 运行时，并且只依赖实际用于计算的内核。

编译器构建在 XLA 框架的基础上。桥接 TensorFlow 到 XLA 框架的代码位于 [tensorflow/compiler](https://www.tensorflow.org/code/tensorflow/compiler/) 目录下，这个目录同时包含对 TensorFlow 图表[即时（JIT）编译](../../performance/xla/jit.md)的支持。

## tfcompile 做了什么？

`tfcompile` 接受一个子图，子图由反馈和提取（TensorFlow 中的概念）确定，并生成一个实现这个子图的函数。`feed` 是函数的入参，`fetch` 是函数的出参。所有的输入必须由反馈声明；输出的裁剪后的子图不能包含位置占位符和变量节点。通常将所有的位置占位符和变量声明为反馈，以确保输出的子图中不再包含这些节点。生成的函数会和一个导出函数签名的头文件，以及一个包含实现的对象文件一起作为  `cc_library` 打包。用户可以视情况编写代码调用生成的函数。

## 使用 tfcompile

这一节详细介绍如何使用 `tfcompile` 从 TensorFlow 子图生成一个可执行二进制文件。步骤如下：

*   步骤一：配置编译的子图
*   步骤二：使用 `tf_library` 构建宏来编译子图
*   步骤三：编写代码调用子图
*   步骤四：创建最终的二进制文件

### 步骤一：配置编译的子图

对应生成的函数的入参和出参，确定反馈和提取。然后在 [`tensorflow.tf2xla.Config`](https://www.tensorflow.org/code/tensorflow/compiler/tf2xla/tf2xla.proto) 协议中配置 `feed` 和 `fetch`。

```textproto
# Each feed is a positional input argument for the generated function.  The order
# of each entry matches the order of each input argument.  Here “x_hold” and “y_hold”
# refer to the names of placeholder nodes defined in the graph.
feed {
  id { node_name: "x_hold" }
  shape {
    dim { size: 2 }
    dim { size: 3 }
  }
}
feed {
  id { node_name: "y_hold" }
  shape {
    dim { size: 3 }
    dim { size: 2 }
  }
}

# Each fetch is a positional output argument for the generated function.  The order
# of each entry matches the order of each output argument.  Here “x_y_prod”
# refers to the name of a matmul node defined in the graph.
fetch {
  id { node_name: "x_y_prod" }
}
```

### 步骤二：使用 tf_library 构建宏来编译子图

这一步通过 `tf_library` 构建宏将图转变成 `cc_library`。`cc_library` 包含一个从图生成的代码的对象文件，以及一个提供对生成代码的访问权限的头文件。`tf_library` 利用 `tfcompile` 将 TensorFlow 图编译成可执行代码。

```build
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

# Use the tf_library macro to compile your graph into executable code.
tf_library(
    # name is used to generate the following underlying build rules:
    # <name>           : cc_library packaging the generated header and object files
    # <name>_test      : cc_test containing a simple test and benchmark
    # <name>_benchmark : cc_binary containing a stand-alone benchmark with minimal deps;
    #                    can be run on a mobile device
    name = "test_graph_tfmatmul",
    # cpp_class specifies the name of the generated C++ class, with namespaces allowed.
    # The class will be generated in the given namespace(s), or if no namespaces are
    # given, within the global namespace.
    cpp_class = "foo::bar::MatMulComp",
    # graph is the input GraphDef proto, by default expected in binary format.  To
    # use the text format instead, just use the ‘.pbtxt’ suffix.  A subgraph will be
    # created from this input graph, with feeds as inputs and fetches as outputs.
    # No Placeholder or Variable ops may exist in this subgraph.
    graph = "test_graph_tfmatmul.pb",
    # config is the input Config proto, by default expected in binary format.  To
    # use the text format instead, use the ‘.pbtxt’ suffix.  This is where the
    # feeds and fetches were specified above, in the previous step.
    config = "test_graph_tfmatmul.config.pbtxt",
)
```

> 为了给示例生成 GraphDef 协议（test_graph_tfmatmul.pb），运行 [make_test_graphs.py]("https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/make_test_graphs.py")，并使用 --out_dir 标志指定输出地址。

典型图包含 [`Variables`](../../api_guides/python/state_ops.md)，表示通过训练学习的权重，但 `tfcompile` 无法编译一个包含 `Variable` 的子图。[freeze_graph.py](https://www.tensorflow.org/code/tensorflow/python/tools/freeze_graph.py) 工具使用存储在检查点文件中的值将变量转化为常量。为方便起见，`tf_library` 宏支持传入运行工具的 `freeze_checkpoint` 参数。更多示例可以查看 [tensorflow/compiler/aot/tests/BUILD](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/BUILD)。

> 在编译的子图中显示的常量将直接编译到生成的代码中。为了将常量传入而不是编译进生成的函数，只需将它们作为反馈传入。

更多关于 `tf_library` 构建宏的细节，查看 
[tfcompile.bzl](https://www.tensorflow.org/code/tensorflow/compiler/aot/tfcompile.bzl)。

更多关于底层 `tfcompile` 工具，查看 
[tfcompile_main.cc](https://www.tensorflow.org/code/tensorflow/compiler/aot/tfcompile_main.cc)。

### 步骤三：编写代码调用子图

此步骤使用在前几步中通过 `tf_library` 构建宏生成的头文件（`test_graph_tfmatmul.h`）来调用生成的代码。头文件位于和构建包相同的 `bazel-genfiles` 目录下，并基于 `tf_library` 构建宏中设置的 name 属性命名。例如，为 `test_graph_tfmatmul` 生成的头文件是 `test_graph_tfmatmul.h`。下面是生成文件的简化版。在`bazel-genfiles` 中生成的文件还会包含其他有用的注释。

```c++
namespace foo {
namespace bar {

// MatMulComp represents a computation previously specified in a
// TensorFlow graph, now compiled into executable code.
class MatMulComp {
 public:
  // AllocMode controls the buffer allocation mode.
  enum class AllocMode {
    ARGS_RESULTS_AND_TEMPS,  // Allocate arg, result and temp buffers
    RESULTS_AND_TEMPS_ONLY,  // Only allocate result and temp buffers
  };

  MatMulComp(AllocMode mode = AllocMode::ARGS_RESULTS_AND_TEMPS);
  ~MatMulComp();

  // Runs the computation, with inputs read from arg buffers, and outputs
  // written to result buffers. Returns true on success and false on failure.
  bool Run();

  // Arg methods for managing input buffers. Buffers are in row-major order.
  // There is a set of methods for each positional argument.
  void** args();

  void set_arg0_data(float* data);
  float* arg0_data();
  float& arg0(size_t dim0, size_t dim1);

  void set_arg1_data(float* data);
  float* arg1_data();
  float& arg1(size_t dim0, size_t dim1);

  // Result methods for managing output buffers. Buffers are in row-major order.
  // Must only be called after a successful Run call. There is a set of methods
  // for each positional result.
  void** results();


  float* result0_data();
  float& result0(size_t dim0, size_t dim1);
};

}  // end namespace bar
}  // end namespace foo
```

根据 `tf_library` 宏中声明的 `cpp_class`，生成了 `foo::bar` 命名空间下的 `MatMulComp` C++ 类。所有生成的类都具有类似的 API，唯一的区别在于处理参数和结果缓冲区的方法。这些方法在缓冲区的数量和类型上存在差异，而这又取决于 `tf_library` 宏接受的 `feed` 和 `fetch` 参数。

在生成的类中有三种类型的缓冲区：`args` 代表输入，`results` 代表输出，以及 `temps` 代表用于内部执行计算的临时缓存。默认情况下，生成类的每个实例都会分配和管理所有缓存。通过设置构造函数参数 `AllocMode` 可以改变这一行为。所有缓冲区都与 64 字节边界对齐。

生成的 C++ 类只不过是在 XLA 生成的底层代码基础上的一层封装。

基于 [`tfcompile_test.cc`](https://www.tensorflow.org/code/tensorflow/compiler/aot/tests/tfcompile_test.cc) 的调用生成函数的示例：

```c++
#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include <iostream>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/aot/tests/test_graph_tfmatmul.h" // generated

int main(int argc, char** argv) {
  Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());


  foo::bar::MatMulComp matmul;
  matmul.set_thread_pool(&device);

  // Set up args and run the computation.
  const float args[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::copy(args + 0, args + 6, matmul.arg0_data());
  std::copy(args + 6, args + 12, matmul.arg1_data());
  matmul.Run();

  // Check result
  if (matmul.result0(0, 0) == 58) {
    std::cout << "Success" << std::endl;
  } else {
    std::cout << "Failed. Expected value 58 at 0,0. Got:"
              << matmul.result0(0, 0) << std::endl;
  }

  return 0;
}
```

### 步骤四：创建最终的二进制文件

这一步将第二步中 `tf_library` 生成的库和第三步中编写的代码结合起来，创建最终的二进制文件。下面是 `bazel` 构建文件的一个示例。

```build
# Example of linking your binary
# Also see //tensorflow/compiler/aot/tests/BUILD
load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")

# The same tf_library call from step 2 above.
tf_library(
    name = "test_graph_tfmatmul",
    ...
)

# The executable code generated by tf_library can then be linked into your code.
cc_binary(
    name = "my_binary",
    srcs = [
        "my_code.cc",  # include test_graph_tfmatmul.h to access the generated header
    ],
    deps = [
        ":test_graph_tfmatmul",  # link in the generated object file
        "//third_party/eigen3",
    ],
    linkopts = [
          "-lpthread",
    ]
)
```
