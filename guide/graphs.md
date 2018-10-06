# 流图与会话

TensorFlow 使用**数据流图**（Dataflow graph，后文以“流图”简称。）表示计算过程，它是依据各个操作之间的依赖关系生成的。这是一个底层的编程模型，你需要先定义数据流图，然后创建一个 TensorFlow **会话**以在多个本地或远程的设备上运行流图的各个部分。

如果你打算直接使用底层编程模型，这份指南将会非常有用。更高级的 API（例如 `tf.estimator.Estimator` 和 Keras）为最终用户隐藏了流图和会话的使用细节，但是这份指南能有效帮助你了解这些 API 是如何实现的。

## 为什么要用数据流图？

![](../images/tensors_flowing.gif)

[数据流](https://en.wikipedia.org/wiki/Dataflow_programming)是一种常见的并行计算的编程模型。在数据流图中，节点代表计算单元，边表示计算所使用或产生的数据。例如，在 TensorFlow 流图中，`tf.matmul` 操作将对应一个具有两个输入边（矩形因子）和一个输出边（矩阵乘积）的节点。

<!-- TODO(barryr): Add a diagram to illustrate the `tf.matmul` graph. -->

TensorFlow 利用数据流模型来执行程序，有如下几个优点：
  
* **并行编程**使用显式边来表示操作间的依赖关系，系统就能很容易的识别出可以并行的操作。
    
* **分布执行**用显式边来表示操作间传递的值，TensorFlow 可以将你的程序分布到不同机器的不同设备（CPU, GPU 和 TPU）上，并完成必要的设备间通信和协调工作。

* **高效编译**TensorFlow 的 [XLA 编译器](../performance/xla/index.md)可以使用数据流图中的信息生成更快的代码，例如，通过合并相邻的操作让代码运行更快。

* **可移植性**数据流图在模型中是一种语言无关的表述形式。你可以使用 Python 构建数据流图，将其存储在 [SavedModel](../guide/saved_model.md) 中，然后在 C++ 程序中重载来实现快速推理。

## `tf.Graph` 是什么?

一个 `tf.Graph` 中包含了以下两种重要的数据信息：

* **流图结构**图的节点和边表明了各个独立操作组合在一起的方式，但并没有说明他们的用法。流图结构很像汇编代码：查看代码能够得到一些有用的信息，但它并不能传达给你所有源代码中的有效信息。

* **流图集合**TensorFlow 提供了一种通用机制来将元数据的集合存储到 `tf.Graph` 中。`tf.add_to_collection` 函数能够将一个对象列表关联到一个键值（其中 `tf.GraphKeys` 定义了一些标准键值），`tf.get_collection` 能够查找与键值相关联的所有对象。TensorFlow 库的很多部分都使用这个工具：例如，当你创建一个 `tf.Variable`，它默认被添加到代表“全局变量”和“可训练变量”的集合中。稍后你再创建 `tf.train.Saver` 或 `tf.train.Optimizer` 时，这些集合中的变量会作为 `tf.train.Saver` 或 `tf.train.Optimizer` 的默认参数。

## 构建一个 `tf.Graph`

大部分 TensorFlow 程序都始于构建一个数据流图，在这个阶段，你调用 TensorFlow API 函数来构建新的 `tf.Operation`（节点）和 `tf.Tensor`（边）对象，并将它们添加到 `tf.Graph` 实例中。TensorFlow 提供了一个**默认流图** ，它是同处相同上下文环境的所有 API 函数的隐藏参数。例如：

* 调用 tf.constant(42.0) 来创建一个数值为 42.0 的 `tf.Operation` 并将其添加到默认的图中，同时返回一个代表了这个常量的值的 `tf.Tensor` 对象。

* 调用 tf.matmul(x, y) 来创建一个 `tf.Operation` 将 `tf.Tensor` 对象中的 x 和 y 相乘，并将其添加到默认图中，返回代表其乘积的 `tf.Tensor 对象。

* 执行 `v = tf.Variable(0)` 会向流图中添加一个 `tf.Operation`，它会保存一个可写入的 tensor 值，且在 `tf.Session.run` 调用过程中持续存在。 `tf.Variable` 对象封装了这个操作，并且可以[作为 tensor 对象使用](#可用作 tensor 的对象) 一样使用，它将读取当前存储值，`tf.Variable` 对象也有一些方法，例如 `tf.Variable.assign` 和 `tf.Variable.assign_add`，它们创建 `tf.Operation` 对象，在执行时能够更新存储值。（有关变量的更多信息，请参阅 [Variables](../guide/variables.md)。）

* 调用 `tf.train.Optimizer.minimize` 可以向默认图添加计算梯度的操作和张量，并返回一个 `tf.Operation`，在其运行时能够将计算出的梯度应用到一组变量上。

大多数程序仅依赖默认流图，但是，参考使用多个计算图来学习更多高级的使用方法。高级 API 可以替你管理默认的计算图，比如 `tf.estimator.Estimator` API 可以创建不同的计算图用于训练和求值。

注意：调用 TensorFlow API 中的大部分函数只是将操作和张量添加到默认的计算图中，实际上并没有执行计算。当你不断地组合这些函数，直到得到一个可以表示整体计算（比如，梯度下降计算中的某一步）的 `tf.Tensor` 或 `tf.Operation` 对象，然后将该对象传递给 `tf.Session` 来运行计算。更多细节请参考章节“在 `tf.Session` 执行计算图”。

## 操作的命名

一个 `tf.Graph` 对象为它包含的 `tf.Operation` 对象定义了一个**命名空间**。TensorFlow 自动为流图中的每个操作生成一个唯一的名称，但给操作取一个描述性名称可方便代码阅读和程序调试。TensorFlow API 提供了两种方法来 给操作重新命名：

* 每个 API 函数创建一个新的 `tf.Operation` 或返回一个新的 `tf.Tensor` 接受一个可选的 `name` 参数 。例如，`tf.constant(42.0, name="answer")` 创建一个叫做 `"answer"` 的新 `tf.Operation` 和一个叫做 `"answer:0"` 的新 `tf.Tensor`。如果默认流图已经包含名为 `"answer"` 的操作，则 TensorFlow 会在名称后加上 `"_1"` 和 `"_2"` 等，以使其唯一。

*  `tf.name_scope` 函数可以为所有在特定上下文环境中创建的操作添加一个**命名空间**前缀。当前名称作用域前缀是一个 `"/"`，用于界定所有激活状态的  `tf.name_scope` 上下文环境管理器的名称列表。如果在当前上下文环境中已经使用了名称范围，则 TensorFlow 会添加`"_1"` 和 `"_2"` 等。例如：

  ```python
  c_0 = tf.constant(0, name="c")  # => operation named "c"

  # 已经被使用过的名称将会变为 "uniquified"。
  c_1 = tf.constant(2, name="c")  # => operation named "c_1"

  # 命名空间为在同一上下文环境中创建的所有操作添加一个前缀。
  with tf.name_scope("outer"):
    c_2 = tf.constant(2, name="c")  # => 操作被命名为"outer/c"

    # 命名空间像分层文件系统中嵌套的路径。
    with tf.name_scope("inner"):
      c_3 = tf.constant(3, name="c")  # => 操作被命名为 "outer/inner/c"

    # 退出命名空间,将返回到前一个前缀名称所表示的命名空间。
    c_4 = tf.constant(4, name="c")  # => 操作被命名为 "outer/c_1"

    # 已经被使用过的名称将会变为 "uniquified"。
    with tf.name_scope("inner"):
      c_5 = tf.constant(5, name="c")  # => 操作被命名为 "outer/inner_1/c"
  ```

图形可视化工具使用命名空间来将操作分组，并减少图形的视觉复杂性。请参阅[将流图可视化](#将流图可视化)来获取更多信息。

请注意，`tf.Tensor` 对象隐式地以生成此张量的 `tf.Operation` 名字命名。tensor 命名格式为`"<OP_NAME>:<i>"`，其中：

* `"<OP_NAME>"` 是生成 tensor 的操作名称。
* `"<i>"` 是一个整数，表示操作的输出中 tensor 的索引。

## 在不同设备部署操作

如果你想在多个不同的设备上运行 TensorFlow 程序，`tf.device` 函数提供了一种便捷的请求方式，它使得在某一特定的上下文下创建的所有操作都会在指定的同一个（或同一类）设备上运行。

**设备规范**有以下的形式：

```
/job:<JOB_NAME>/task:<TASK_INDEX>/device:<DEVICE_TYPE>:<DEVICE_INDEX>
```
分别表示：

* `<JOB_NAME>` 是一个可以包含字母与数字的字符串，但不能以数字开头。
* `<DEVICE_TYPE>` 是一个已注册的设备类型（例如 `GPU` 或者 `CPU`）。
* `<TASK_INDEX>` 是一个非负整数，表示以 `<JOB_NAME>` 命名的 job 中 task 的索引。作业和任务的详细介绍请查看 `tf.train.ClusterSpec`。
* `<DEVICE_INDEX>` 是一个非负整数，表示设备的索引，例如，用来区分在同一进程中使用的不同 GPU 设备。

你无需指定设备规范中的所有参数。例如，如果你正在单 GPU 的设备上使用单机配置运行，则可以使用 `tf.device` 将某些操作分配到 CPU 和 GPU 上：

```python
# 创建时没有指定任何设备的操作将在“尽可能最好”的设备上运行。
# 例如，如果你有一个 GPU 和一个 CPU 可用，并且该操作具有 GPU
# 实现，则 TensorFlow 将选择该 GPU。
weights = tf.random_normal(...)

with tf.device("/device:CPU:0"):
  #  此上下文中创建的操作会被分配到 CPU 上。
  img = tf.decode_jpeg(tf.read_file("img.jpg"))

with tf.device("/device:GPU:0"):
  # 当前上下文环境中创建的操作会被分配到 GPU 上。
  result = tf.matmul(weights, img)
```

如果你要在[典型的分布式配置](../deploy/distributed.md)中部署 TensorFlow，你可以通过指定 job 名称和 task ID 来实现将变量部署在参数服务器的 job（`"/job:ps"`）的 task 中，并将其他操作部署在工作机的 job（`"/job/worker"`）的 task 中：

```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(tf.truncated_normal([784, 100]))
  biases_1 = tf.Variable(tf.zeroes([100]))

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(tf.truncated_normal([100, 10]))
  biases_2 = tf.Variable(tf.zeroes([10]))

with tf.device("/job:worker"):
  layer_1 = tf.matmul(train_batch, weights_1) + biases_1
  layer_2 = tf.matmul(train_batch, weights_2) + biases_2
```

`tf.device} 为单个操作或 TensorFlow 计算图子图的部署提供了很大的灵活性。很多时候，简单的启发式工作也很有效。例如`tf.train.replica_device_setter` API 能够和 `tf.device` 一起使用，来部署 **并行数据分布式训练**。再例如，下面这段代码展示了 `tf.train.replica_device_setter` 如何将不同的分配策略应用于 `tf.Variable` 对象和其他操作上：

```python
with tf.device(tf.train.replica_device_setter(ps_tasks=3)):
  # tf.Variable  默认情况下以轮询调度的方式部署在 "/job:ps" 的任务列表中
  w_0 = tf.Variable(...)  # placed on "/job:ps/task:0"
  b_0 = tf.Variable(...)  # placed on "/job:ps/task:1"
  w_1 = tf.Variable(...)  # placed on "/job:ps/task:2"
  b_1 = tf.Variable(...)  # placed on "/job:ps/task:0"

  input_data = tf.placeholder(tf.float32)     # 部署 "/job:worker"
  layer_0 = tf.matmul(input_data, w_0) + b_0  # 部署 "/job:worker"
  layer_1 = tf.matmul(layer_0, w_1) + b_1     # 部署 "/job:worker"
```

## 可用作 Tensor 的对象

很多 TensorFlow 操作使用一个或多个 `tf.Tensor` 对象用作参数。例如，`tf.matmul` 接受两个 `tf.Tensor` 对象作为参数，`tf.add_n` 则使用一个包含 `n` 个 `tf.Tensor` 对象的列表。为了方便，这些函数都会接受一个**可用作 tensor 的对象**而不是 `tf.Tensor`，并使用 `tf.convert_to_tensor` 方法隐式的将其转换为一个 `tf.Tensor`。以下类型可用作 tensor 对象中的元素。

* `tf.Tensor`
* `tf.Variable`
* [`numpy.ndarray`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)
* `list` (和可用做 tensor 对象的列表)
* Python 标量类型: `bool`, `float`, `int`, `str`

你可以使用 `tf.register_tensor_conversion_function` 注册附加的可用作 tensor 的对象。

注意：TensorFlow 默认会在每次创建同样的可用作 tensor 的对象时创建一个新的 `tf.Tensor`。如果可用作 tensor 的对象很大（比如一个包含训练样本集合的 numpy.ndarray 对象），在反复使用的过程中可能会耗尽内存。为了避免这种情况，你可以调用 `tf.convert_to_tensor` 手动将可用作 tensor 对象装换为 tensor，并使用返回的 `tf.Tensor` 对象。

## 在 `tf.Session` 中执行一个流图

TensorFlow 使用 `tf.Session` 类来代表客户端程序（通常是 Python 程序，其他语言里也有类似的接口）和 C++ runtime 之间的连接。`tf.Session` 对象提供了对设备的访问方式，包括本地机器的设备和分布式运行的远程设备。它还会缓存有关 `tf.Graph` 的信息，以便可以高效地多次运行相同的计算。

### 创建一个 `tf.Session`

如果你在使用一个底层的 TensorFlow API，你能用如下代码为当前的默认流图创建一个 `tf.Session`：

```python
# 创建一个默认的进程内的会话。
with tf.Session() as sess:
  # ...

# 创建一个远程会话。
with tf.Session("grpc://example.org:2222"):
  # ...
```

由于 `tf.Session` 占用物理资源（例如 GPUs 和网络连接），它通常作为上下文管理器使用（用于 with 代码块），并且在退出时会自动关闭当前会话。在不使用 with 代码块的情况下，也可以创建一个会话，但你应当在执行结束时显式地调用 `tf.Session.close` 来释放资源。

注意：高级 API，例如 `tf.train.MonitoredTrainingSession` 或者 `tf.estimator.Estimator` 将为你创建和管理  `tf.Session`。这些 API 接受可选的 `target` 和 `config` 参数（直接作为参数或者 `tf.estimator.RunConfig` 对象其一部分），参数形式与下文所讲的并无二致：

`tf.Session.__init__` 接收三个可选参数：

* **`target`.** 如果此参数为空（默认值），则此会话将仅使用本地机器中的设备。但是，你也可以指定一个 `grpc://` URL 来指定 TensorFlow 服务器的地址，从而使会话可以访问该服务器控制的机器上的所有设备。有关如何创建 TensorFlow 服务器的详细信息，请参阅 `tf.train.Server` 。例如，在常见的**流图间复制**配置中，`tf.Session` 在客户端进程中连接到 `tf.train.Server`。[分布式 TensorFlow](../deploy/distributed.md) 部署指南介绍了其他一些常见的情况。

* **`graph`.** 默认情况下，一个新的 `tf.Session` 将能被绑定到当前默认流图中且只能运行其中的操作。如果你在程序中使用多个流图编程（参阅[使用多个流图编程](#使用多个流图编程)获取更多信息），则可以在构建会话时指定特定的 `tf.Graph`。

* **`config`.** 这个参数允许你指定一个控制会话行为的 `tf.ConfigProto`。例如一些配置选项：

  * `allow_soft_placement.` 将其设置为 `True` 以启用“软”设备分配算法，该算法将忽略 `tf.device` 注解中尝试将仅在 CPU 上运行的操作分配到 GPU 的算法，直接将操作分配到 CPU 上。

  * `cluster_def.` 使用分布式 TensorFlow 时，此选项能够指定在计算中使用的机器，并提供作业名称、任务索引和网络地址之间的映射关系。有关详细信息，请参阅 `tf.train.ClusterSpec.as_cluster_def`。

  * `graph_options.optimizer_options`. 提供对 TensorFlow 在执行流图前对其优化的控制。

  * `gpu_options.allow_growth`. 将其设置为 `True` 来更改 GPU 内存分配器，以便逐渐增加分配的内存量，而不是在启动时分配大部分内存。


### 使用 `tf.Session.run` 执行操作

`tf.Session.run` 方法是运行 `tf.Operation` 评估或对 `tf.Tensor` 求值的主要机制。你可以将一个或多个 `tf.Operation` 或 `tf.Tensor` 对象传递给 `tf.Session.run`，TendorFlow 将执行计算结果所需的操作。

`tf.Session.run` 要求你指定一个**提取**列表，其决定了返回值，列表中的元素可以是 `tf.Operation`，`tf.Tensor` 或者[可用作 Tensor 的对象](#可用作 Tensor 的对象)（比如 `tf.Variable`）。这些提取列表确定了必须执行 `tf.Graph` 中的哪些**子流图**以获取结果：包含所有在提取列表中命名的操作的子流图，以及包含其依赖操作的子图。例如，下面的代码片段显示了 `tf.Session.run` 的不同参数如何引导不同的子流图被执行：

```python
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
output = tf.nn.softmax(y)
init_op = w.initializer

with tf.Session() as sess:
  # 运行 `w` 的初始化操作。
  sess.run(init_op)

  # 对 `output` 求值。`sess.run(output)` 将返回一个包含计算结果的 NumPy 数组。
  print(sess.run(output))

  # 对 `y` 和 `output` 求值。请注意， `y` 只会计算一次，计算结果既作为 `y_val` 的值返回，
  # 又会作为 `tf.nn.softmax()` 操作的输入。`y_val` 和 `output_val` 都是 NumPy 数组。
  y_val, output_val = sess.run([y, output])
```

`tf.Session.run` 还有一个字典类型的可选参数 **feeds** ，该参数将 `tf.Tensor` 对象（通常是 `tf.placeholder` 张量）映射到值（通常是 Python 张量、列表或者 NumPy 数组），在执行中这些张量将被对应的值替换。例如：

```python
# 定义一个三个浮点值向量的占位符和一个依赖于它的计算。
x = tf.placeholder(tf.float32, shape=[3])
y = tf.square(x)

with tf.Session() as sess:
  # 提供不同的参数值会求得不同的 `y` 值。
  print(sess.run(y, {x: [1.0, 2.0, 3.0]}))  # => "[1.0, 4.0, 9.0]"
  print(sess.run(y, {x: [0.0, 0.0, 5.0]}))  # => "[0.0, 0.0, 25.0]"

  # 抛出 `tf.errors.InvalidArgumentError`，因为你必须先给 `tf.placeholder()` 赋值，然后才能计算器依赖关系之上的 tensor 。
  sess.run(y)

  # 抛出 `ValueError`，因为 `37.0` 与 `x` 形状不匹配。
  sess.run(y, {x: 37.0})
```

`tf.Session.run` 接受一个可以指定调用选项的 `options` 可选参数和一个可以收集执行信息的 `run_metadata` 的可选参数，使你可以收集有关执行的元数据。例如，你可以使用这些选项来收集有关执行的跟踪信息：

```
y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
  # 定义调用 `sess.run()` 的选项。
  options = tf.RunOptions()
  options.output_partition_graphs = True
  options.trace_level = tf.RunOptions.FULL_TRACE

  # 定义一个用于接受返回元数据的容器。
  metadata = tf.RunMetadata()

  sess.run(y, options=options, run_metadata=metadata)

  # 打印出在每个设备上执行的子流图。
  print(metadata.partition_graphs)

  # 打印出每次执行操作的执行时间。
  print(metadata.step_stats)
```

## 将流图可视化

TensorFlow 提供一些工具帮助用户理解流图中的代码。**流图可视化工具**是 TensorBoard 中的一个组件，可以在浏览器中直观的呈现流图的结构。创建可视化的最简单方法是在创建 `tf.summary.FileWriter` 时传入一个 `tf.Graph`：

```python
# 构建你的流图
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
# ...
loss = ...
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
  # `sess.graph` 提供了对 `tf.Session` 中流图的访问。
  writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)

  # 执行你的计算...
  for i in range(1000):
    sess.run(train_op)
    # ...

  writer.close()
```

注意：如果你正在使用 `tf.estimator.Estimator`，则流图（以及任意摘要）的日志信息将自动到创建估计器时指定的 `model_dir` 目录中。

然后，你可以在 `tensorboard` 中打开日志，在 “Graph” 页面查看高度可视化的计算图结构。请注意， 一个典型的 TensorFlow 计算图——尤其是自动计算梯度的训练计算图——会由于节点太多而不能立刻全部可视化。计算图使用命名空间来将相关操作归纳到父节点。你可以点解父节点的橙色 "+" 按钮来展开其内部的子图。

![](../images/mnist_deep.png)

要了解更多关于如何使用 TensorBoard 来可视化你的 TensorFlow 应用，请参阅 [TensorBoard 指南](./summaries_and_tensorboard.md)。

## 使用多个流图编程

注意：常用的组织代码方式是一个流图用来训练模型，另一个流图用训练好的模型来求值或者推断结果。用于求值或者推断结果的流图在很多方面都有别于训练流图：例如，dropout 和 batch normalization 等技术在不同的场景下会执行不同的操作。此外，像 `tf.train.Saver` 这样的工具在保存的快照中默认使用 `tf.Variable` 对象的名字（其名字又基于下层的 `tf.Operation` 的名字）来标识每个变量。这样，你可以使用完全独立的 Python 线程来构建和执行计算图，也可以在同一个线程中使用多个流图。本节介绍了如何在同一进程中使用多个流图。

如上所述， TensorFlow 提供了一个“默认流图”，隐式传递给同一上下文中的所有 API 函数。对于许多个应用程序来说，一张流图就足够了。但是，TensorFlow 也提供了操作默认流图的方法，这在更高级的用例中是有用处的。例如：

* `tf.Graph` 定义 `tf.Operation` 对象的命名空间：单个流图中的每个操作都必须具有唯一的名称。如果所请求的名称已被占用，TensorFlow 将在名字末尾加上 `"_1"` 和 `"_2"` 等来确保操作名称的唯一性，显式地创建多个计算图，让你在命名操作的时候有更大的控制权。

* 默认流图存储了其内部所有的 `tf.Operation` 和 `tf.Tensor` 的信息。如果你的程序中创建了许多互无联系的子图，使用不同的 `tf.Graph` 来构建每个子流图可能会更有效率，这样就能对无关的状态做垃圾回收处理。

你可以使用 `tf.Graph.as_default` 上下文管理来安装一个不同的 `tf.Graph` 来改变默认流图：

```python
g_1 = tf.Graph()
with g_1.as_default():
  # 这作用域中创建的操作会加到  `g_1` 中。
  c = tf.constant("Node in g_1")

  # 这个作用域中创建的会话会运行 `g_1` 中的操作。
  sess_1 = tf.Session()

g_2 = tf.Graph()
with g_2.as_default():
  # 这作用域中创建的操作会加到  `g_2` 中。
  d = tf.constant("Node in g_2")

# 或者，你可以在构建 `tf.Session` 时传递一个流图：
# `sess_2`会运行 `g_2` 中的操作。
sess_2 = tf.Session(graph=g_2)

assert c.graph is g_1
assert sess_1.graph is g_1

assert d.graph is g_2
assert sess_2.graph is g_2
```

调用 `tf.get_default_graph` 可访问到当前的默认流图，它会返回一个 `tf.Graph` 对象：

```python
# 打印默认流图中所有的操作。
g = tf.get_default_graph()
print(g.get_operations())
```
