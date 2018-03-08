# 使用即时编译

> 注意: 为了支持 XLA（加速线性代数），TensorFlow 必须从源文件编译。

## 为什么使用即时编译？

TensorFlow / XLA 即时编译器通过 XLA 编译并运行一部分 TensorFlow 图模型。与标准的 TensorFlow 方法
相比，XLA 能把多个运算符融合成一系列已编译内核（内核融合）。和 TensorFlow 执行器逐个运行操作符相比，
融合运算符能减少对内存带宽的要求，提升计算性能。

## 通过 XLA 计算 TensorFlow 图模型

通过 XLA 计算 TensorFlow 图模型的方式有两种：一是用 CPU 或 GPU 设备上的即时编译操作符，二是
把操作符放到 XLA_CPU 或 XLA_GPU TensorFlow 设备上。把操作符直接放到一个 TensorFlow XLA
设备上将强制执行此操作符，因此这种方法主要用于测试。

> 注意：大部分情况下，XLA CPU 底层会生成快速、单线程的代码，但是不会如 TensorFlow CPU 底层一样并行化。 XLA GPU 后端与标准的 TensorFlow 后端大相径庭，运行速度时快时慢。

### 开启即时编译

即时编译可以从会话层开启，或选定操作运算手动开启。两种方式都是零拷贝的 --- 数据在同台设备的
已编译 XLA 内核和 TensorFlow 操作符之间传递时，无需另行复制。

#### 会话层开启

在会话层开启即时编译时，系统将尽可能把所有操作符编译成 XLA 计算操作。每个 XLA 计算操作将被编译
成单个或多个设备底层内核。

受某些限制影响，如果图模型中有两个相邻的操作符都要使用 XLA，它们将会被编译成单个 XLA 计算。

只需通过把 `global_jit_level` 设置成 `tf.OptimizerOptions.ON_1`，并在会话初始化阶段传入
配置，就可以在会话层开启即时编译。

```python
# 配置开启即时编译
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

sess = tf.Session(config=config)
```

> 注意：在会话层开启即时编译将不会导致为 CPU 编译操作符。CPU 运算的即时编译必须通过下面描述的手
动方法开启，原因在于 CPU 底层是单线程的。

#### 手动开启

对于单个或多个操作符，可以手动开启即时编译，只需要通过给要编译的操作符的 `_XlaCompile=true` 属性
做标记。最简单的方法就是通过 [`tensorflow/contrib/compiler/jit.py`](https://www.tensorflow.org/code/tensorflow/contrib/compiler/jit.py)
里定义的 `tf.contrib.compiler.jit.experimental_jit_scope()` 。
使用范例：

```python
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

    x = tf.placeholder(np.float32)
    with jit_scope():
      y = tf.add(x, x)  # add 将被 XLA 编译
```

目前，`_XlaCompile` 属性并未支持所有操作符。如果一个操作符无法编译，TensorFlow 将默认退回到常规方法。

### 将操作符加载到 XLA 设备中

另一个通过 XLA 执行计算的方式是将操作符载入到特定的 XLA 设备中。这个方法通常只用于测试。可行设备
包括 `XLA_CPU` 或 `XLA_GPU`。

```python
with tf.device("/job:localhost/replica:0/task:0/device:XLA_GPU:0"):
  output = tf.add(input1, input2)
```

不同于标准 CPU 和 GPU 设备上的即时编译，这些设备在传入和传出数据时需要建立数据副本。这个副本导致
在同一个图模型中混合使用 XLA 和 TensorFlow 操作符的开销变得很大。

## 教程

这个教程涵盖了一个简单版的 MNIST softmax 训练模型。在会话层开启了即时编译，只支持 GPU。

在开始前，先确认 LD_LIBRARY 环境变量或者 ldconfig 包括了有用于 CUDA 特征描述
[(CUPTI)](http://docs.nvidia.com/cuda/cupti/index.html) 的
工具库 `$CUDA_ROOT/extras/CUPTI/lib64`。TensorFlow 用 CUPTI 从 GPU 获取追踪信息。

### 步骤 #1: 准备代码范例

下载或把[mnist_softmax_xla.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_softmax_xla.py)
放到 TensorFlow 源码之外的文件夹内。

### 步骤 #2: 不用 XLA 运行代码

执行 python 代码，不用 XLA 训练模型。

```shell
python mnist_softmax_xla.py --xla=''
```

用 Chrome 事件追踪特征描述器 (导航到 chrome://tracing)，当代码执行完时打开时间线文件 `timeline.ctf.json`。
渲染出来的时间线应该和下图相似，可能跨多个 GPU 有一些标有 `MatMul` 的绿块。
<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/jit_timeline_gpu.png">
</div>

### 步骤 #3：用 XLA 运行代码

执行 python 代码，用 XLA 训练模型。打开 XLA 调试工具，用环境变量输出 XLA 图。

```shell
TF_XLA_FLAGS=--xla_generate_hlo_graph=.* python mnist_softmax_xla.py
```

打开时间线文件(`timeline.ctf.json`)。渲染出来的时间线应该和下图相似，可能有一个标有 `_XlaLaunch`
的长块。
<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/jit_timeline_gpu_xla.png">
</div>

通过查看控制台类似下面的输出来了解在 `_XlaLaunch` 里到底发生了什么:

```shell
computation cluster_0[_XlaCompiledKernel=true,_XlaNumConstantArgs=1].v82 [CPU:
pipeline start, before inline]: /tmp/hlo_graph_0.dot

```

控制台显示了包含 XLA 创建的图模型信息的 `hlo_graph_xx.dot` 文件位置。XLA 融合操作符的过程可以
从 `hlo_graph_0.dot` 开始逐个查看分析图了解到。

为了将 .dot 文件渲染成 png 格式，需安装
[GraphViz](http://www.graphviz.org/Download..php) 并运行:

```shell
dot -Tpng hlo_graph_80.dot -o hlo_graph_80.png
```

结果如下图：
<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/jit_gpu_xla_graph.png">
</div>
