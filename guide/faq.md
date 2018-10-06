# 常见问题

本文将会回答一些关于 TensorFlow 常见问题。 如果这里没有你想问的问题，那么建议你去 TensorFlow [社区资源](../about/index.md)中寻找答案。

[TOC]

## 功能和兼容性

#### 我可以使用多台计算机进行分布式训练吗？

没问题！TensorFlow  0.8 的版本[已经支持了分布式计算](../deploy/distributed.md)。TensorFlow 现在支持在一台或者多台计算机上的多个设备（CPUs 和 GPU）。

#### TensorFlow 可以在 Python 3 上运行吗？

早在 2015 年 12 月份，我们发行的 0.6.0 版本就已经支持了 Python 3.3+。

## 建立一个 TensorFlow graph

请查看[有关构建图的 API 文档](../api_guides/python/framework.md)。

#### 为什么 `c = tf.matmul(a, b)` 不会立即执行矩阵乘法？

在 TensorFlow 的 Python API 中，`a`，`b`，和 `c` 都是 `tf.Tensor` 对象。`Tensor` 对象虽然是一个操作结果的符号句柄，但是它不会包含操作输出的值。相反，TensorFlow 还鼓励用户去构建复杂的表达式（类似于整个神经网络和它的梯度）来作为一个数据流图。你相当于节约了整个一个 TensorFlow `tf.Session` 的数据流图（或者它的子图）的计算，这样它就可以更有效率的执行整个计算，而不是一步一步的执行操作。

#### 设备是如何命名的？

对于 CPU 来说支持的设备名字是 `"/device:CPU:0"`（或者 `"/cpu:0"`），第 i 个 GPU 设备则会被命名为 `"/device:GPU:i"`（或者 `"/gpu:i"`）。

#### 如何在一个特定的设备上执行操作？

为了在一个设备上执行一组操作，你可以在 `tf.device` 的上下文中创建它们。查看如何[使用支持 TensorFlow 的 GPU](../guide/using_gpu.md) 的文档可以了解到 TensorFlow 给设备分配操作的详细信息，[CIFAR-10 教程](../tutorials/images/deep_cnn.md)这篇文档则展示了使用多个 GPU 的示例模型。

## 执行一个 TensorFlow 计算

请查看[关于运行图表的 API 文档](../api_guides/python/client.md)。

#### 什么是 feeding 和 placeholder？

Feeding 是 TensorFlow Session API 的一个机制，它允许你在运行时为一个或多个 tensors 替换不同的值。`tf.Session.run` 的参数 `feed_dict` 是一个将 `tf.Tensor` 对象映射到 numpy 数组（或其他的类型）的字典，这个字典将会在执行步骤的时候作为 tensors 的数值被使用。

#### `Session.run()` 和 `Tensor.eval()` 的区别是什么？

如果 `t` 是一个 `tf.Tensor` 对象，`tf.Tensor.eval` 是 `tf.Session.run`，`sess` 是当前的 `tf.get_default_session` 的快捷方式。下面的两段代码是等价的：

```python
# 使用 `Session.run()`.
sess = tf.Session()
c = tf.constant(5.0)
print(sess.run(c))

# 使用 `Tensor.eval()`.
c = tf.constant(5.0)
with tf.Session():
  print(c.eval())
```

在第二个例子中，sesion 表现的像一个 [context manager](https://docs.python.org/2.7/reference/compound_stmts.html#with)，把它作为默认的 session 来安装，会影响整个 `with` 块的生命周期。这个 context manager 方法可以在一些简单的情况（例如单元测试）下让代码更简洁。如果你的代码需要处理多个 graphs 和 sessions，直接调用 `Session.run()` 显然会更简单一点。

#### Session 有生命周期吗？中间的 tensors 呢？

Session 可以拥有一些资源，例如 `tf.Variable`、`tf.QueueBase` 和 `tf.ReaderBase`。这些资源有时可以使用大量内存，并且可以在通过调用 `tf.Session.close` session 被关闭时释放。

作为调用 [`Session.run()`](../api_guides/python/client.md) 的一部分而被创建的中间的 tensors 将会在调用或者调用之前被释放掉。

#### 运行时的 graph 执行是并行的吗？

TensorFlow 运行时的并行图执行穿插着很多不同的维度：

* 单独的操作有并发的实现，它会使用的 CPU 的多个核心或者一个 GPU 的多个线程。
* TensorFlow graph 中的独立节点可以在多个设备上同时运行，这样才能加速 [CIFAR-10 training using multiple GPUs](../tutorials/images/deep_cnn.md)。
* Session API 允许多个并发的步骤（例如，并行的调用 `tf.Session.run`）。这样可以在运行时获得更大的吞吐量，尤其是当一个单独步骤不会使用到你计算机的全部资源时。

#### TensorFlow 支持哪些客户端语言？

TensorFlow 就是为支持多客户端语言设计的。现在，我们支持的最好的客户端语言是 [Python](../api_docs/python/index.md)。执行和构建 graphs 的一些实验性的接口对于 [C++](../api_docs/cc/index.md)，[Java](../api_docs/java/reference/org/tensorflow/package-summary.html) 和 [Go](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go) 也是可用的。

同时 TensorFlow 还有一个 [C-based client API](https://www.tensorflow.org/code/tensorflow/c/c_api.h)，它可以帮助我们创建和支持更多的客户端语言。我们也邀请了一些贡献者一起完成对新语言的支持。

支持其他各种语言（例如 [C#](https://github.com/migueldeicaza/TensorFlowSharp)、[Julia](https://github.com/malmaud/TensorFlow.jl)、[Ruby](https://github.com/somaticio/tensorflow.rb) 和 [Scala](https://github.com/eaplatanios/tensorflow_scala)）的工作，将会由开源社区来创建和支持，这些支持都是基于 TensorFlow 维护者提供的 C API 来构建的。

#### TensorFlow 可以使用我机器上的所有设备（GPU 和 CPU）吗？

TensorFlow 支持多个 GPU 和 CPU。查看如何[使用支持 TensorFlow 的 GPU](../guide/using_gpu.md) 可以获得关于 TensorFlow 是如何给设备分配操作的详细信息。并且 [CIFAR-10 教程](../tutorials/images/deep_cnn.md)还提供了一个使用多个 GPU 的示例模型。

注意：TensorFlow 仅仅使用计算性能大于 3.5 的 GPU 设备。

#### 为什么使用一个读取器或者队列的时候 `Session.run()` 会挂起？

`tf.ReaderBase` 和 `tf.QueueBase` 类提供了一些可以**阻塞**进程直到输入可用（或者有界队列释放出空间）的特殊操作。这些操作可以让你建立复杂的[输入管道](../api_guides/python/reading_data.md)，当然了，这样做的代价就是 TensorFlow 的计算也会变的复杂。看看如何[使用 `QueueRunner` 对象来驱动 queues 和 readers](../api_guides/python/reading_data.md#creating_threads_to_prefetch_using_queuerunner_objects) 的文档，来了解关于如何使用它们的更多信息。

## 变量

请查看关于[变量](../guide/variables.md)和[变量 API 的文档](../api_guides/python/state_ops.md)。

#### 什么是一个变量的生命周期？

当你第一次在 session 对变量执行 `tf.Variable.initializer` 这个操作时，这个变量就会被创建。当执行 `tf.Session.close` 这个操作时，它就会被销毁。

#### 当这些变量被并发调用时，表现如何？

变量允许并发的执行读和写操作。但是在被并发更新的时候从一个变量中读取的数值可能会改变。默认情况下，在没有互斥的前提下，对一个变量
并发的赋值操作是没问题的。通过给 `tf.Variable.assign` 传递 `use_locking=True` 这样一个参数，可以在分配变量时获得一个锁。

## Tensor shapes

请查看 `tf.TensorShape`。

#### 在 Python 中我如何决定一个 tensor 的 shape ？

在 TensorFlow 中，一个 tensor 同时拥有一个静态的（推测出的）的 shape 和 一个动态的（真实的）的 shape。这个静态的 shape 可以使用方法 `tf.Tensor.get_shape` 获取到：这个 shape 是通过我们过去创建 tensor 的操作推测出来的，也有可能是推测出来的（静态的 shape 可能包含 `None`）。如果静态的 shape 并没有被完全的定义，那么一个 `tf.Tensor` 的动态 shape `t` 可以通过调用 `tf.shape(t)` 来确定。

#### `x.set_shape()` 和 `x = tf.reshape(x)` 的区别是什么？

`tf.Tensor.set_shape` 方法会更新一个 `Tensor` 对象的静态 shape，当无法直接通过推测来获取 shape 信息时，这个方法就是最典型的获得额外 shape 信息的方法了。当然了，它并不会改变这个 tensor 的动态 shape。

`tf.reshape` 这个操作会使用一个不同的动态 shape 来创建一个新的 tensor。

#### 如何才能创建一个可变批量大小的 graph ？

通常来说建立一个拥有可变批量大小的 graph 都是非常有用的，因为这样的话，同样的代码即可以使用在（mini-）批训练，又可以使用在单实例推理上训练。作为结果的 graph 可以是 `tf.Graph.as_graph_def` 和 `tf.import_graph_def`。

当创建一个可变大小的 graph 时，最重要的事情就是不要把批处理的大小编码成一个 Python 的常量，而是要使用一个符号 `Tensor` 去代表它。下面的这些 tips 或许会对你有帮助：

* 使用 [`batch_size = tf.shape(input)[0]`](../api_docs/python/array_ops.md#shape) 来从一个 `Tensor` 中获取表示批维度的 `input` 变量，并且把它储存在一个叫 `batch_size` 的 `Tensor` 中。

* 使用 `tf.reduce_mean` 来代替 `tf.reduce_sum(...) / batch_size`。

## TensorBoard

#### 如何可视化一个 TensorFlow graph ？

查看[图形可视化教程](../guide/graph_viz.md)。

#### 向 TensorBoard 发送数据的最简单方法是什么？

给你的 TensorFlow graph 添加一些摘要的操作，并且把这些摘要写在一个日志文件中。然后使用下面的命令启动 TensorBoard：

    python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory

更多详细的信息，请查看 [Summaries 和 TensorBoard 教程](../guide/summaries_and_tensorboard.md)。

#### 每次我启动 TensorBoard 时，就会得到一个网络安全的弹出框！

你可以使用 --host=localhost 这个参数将 TensorBoard 运行在 localhost，而不是 '0.0.0.0'。这样就不会有安全警告了。

## 拓展 TensorFlow

查看如何[向 TensorFlow 添加新操作](../extend/adding_an_op.md)。

#### 我的数据是自定义的格式，怎样做才能让 TensorFlow 正确的读取它们呢？

对于自定义格式的数据，我们有下面 3 个主要的方式来处理。

最简单的方式就是使用 Pyhton 编写解析的代码，将数据转换成 numpy 的 array。然后使用 `tf.data.Dataset.from_tensor_slices` 来从内存数据中创建出一个输入的 pipeline。

如果你的数据不适合放在内存中，那么可以试试在数据集的 pipeline 中解析。使用一个合适的文件读取器，像 `tf.data.TextLineDataset`。然后通过映射 `tf.data.Dataset.map` 适当的操作来转换数据集。最好是预定义 TensorFlow 的一些操作，像 `tf.decode_raw`、`tf.decode_csv`、`tf.parse_example` 或者 `tf.image.decode_png`。

如果你的数据不太好用 TensorFlow 内建的一些操作来解析，那么考虑下转换它吧，在离线模式下转换成一种容易被解析的格式，比如 `tf.python_io.TFRecordWriter` 格式。

自定义解析行为的更有效的方法是[添加一个用 C++ 编写的新操作](../extend/adding_an_op.md)，这个新添加的操作是可以解析你的数据格式的。[处理新数据格式指南](../extend/new_data_formats.md)有关于如何操作的更详细的步骤。

## 其他

#### TensorFlow 的代码风格是什么样的？

TensorFlow Python API 的代码风格是遵循 [PEP8](https://www.python.org/dev/peps/pep-0008/) 的约定。<sup>*</sup>要注意的是，我们使用 `CamelCase` 来对类进行命名，使用 `snake_case` 来对函数，方法以及属性进行命名。同时我们也坚持 [Google Python style guide](https://google.github.io/styleguide/pyguide.html)。

TensorFlow C++ 代码风格遵循 [Google C++ style guide](https://google.github.io/styleguide/cppguide.html) 的约定。

(<sup>*</sup>有一个例外是：我们使用 2 个空格进行缩进，而不是 4 个。)
