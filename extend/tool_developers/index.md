# 工具开发者指南：TensorFlow 模型文件

大部分用户不需要关心 TensorFlow 存储磁盘文件的内部细节，但如果你是一个工具开发者，就不一样了。
比如，你可能想要分析模型，或者希望在 TensorFlow 格式和其它格式之间来回转换。TensorFlow 模型
数据存储在常用的那些文件中，而本指南就是尝试解释你在用这些文件时所需的一些细节，从而让开发相关工具更加容易一些。

[目录]

## 协议缓存（Protocol Buffers）

所有的 TensorFlow 文件格式都是基于 [协议缓存](https://developers.google.com/protocol-buffers/?hl=en) 
这个 Google 开发的工具（全称是 Protocol Buffers，简称为 protobuf），所以正式开始之前有必要先熟悉一下这个工具的原理。
简单来说，你用文本文件来定义数据结构，然后 protobuf 工具据此生成 C、Python 或其它语言中的数据结构（比如类）,这样，
开发者就可以用一种友好的方式来访问数据了。通常协议缓存（Protocol Buffers）被称为 protobufs，本指南会一直使用这个简称。

## GraphDef 类

TensorFlow 中的计算基础为 `Graph` 对象，称为计算图。计算图保存了由结点构成的一个网络，每个结点表示一种操作，
而其他结点可作为这个操作的输入或输出，从而让所有结点相互联结起来。创建了 `Graph` 对象之后，你可以通过调用 `as_graph_def()`
将它存起来，该调用返回的是一个 `GraphDef` 对象。

GraphDef 类是 ProtoBuf 库根据 [tensorflow/core/framework/graph.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto) 的定义创建的一个对象。
protobuf 工具解析这个文本文件，然后生成代码来加载、存储和操控图定义。如果你看到表示模型的单个 TensorFlow 文件，则它很有可能
包含了某个 `GraphDef` 对象的序列化版本，而且它是用 protobuf 代码保存的。

这种生成的代码被用于在磁盘上保存和加载这些 GraphDef 文件，比如下面的模型加载方式：

```python
graph_def = graph_pb2.GraphDef()
```

这一行创建了一个空的 `GraphDef` 对象，这个类是根据 graph.proto 中的文本定义而得到的。接下来我们就用这个对象
来从文件中读取数据。

```python
with open(FLAGS.graph, "rb") as f:
```

这里，根据脚本参数得到了一个文件句柄

```python
  if FLAGS.input_binary:
    graph_def.ParseFromString(f.read())
  else:
    text_format.Merge(f.read(), graph_def)
```

## 文本还是二进制？

ProtoBuf 实际上支持两种不同的文件保存格式。TextFormat 是一种人眼可读的文本形式，这在调试和编辑时是很方便的，
但它在存储数值数据时会变得很大，比如我们常见的权重数据。这种格式的一个小的示例参见 [graph_run_run2.pbtxt](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/demo/data/graph_run_run2.pbtxt)。

相比于文本格式，二进制格式的文件会小得多，缺点就是它是人眼不可读的。在脚本中，我们会要求用户提供一个标志，来选择
到底用二进制还是文本，然后再根据这个标志来调用相应的函数。你可以在 [inception_v3 archive](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz) 
中找到一个较大的二进制文件的示例 `inception_v3_2016_08_28_frozen.pb` 。

它们的应用编程接口（API）可能会让你感到迷惑：二进制格式的调用实际上是 `ParseFromString()`，而文本格式的加载则用到 `text_format` 模块
中的一个工具函数。

## 结点

一旦一个文件被加载到 `graph_def` 变量中，你就可以在变量里访问这个数据了。对于大多数实际应用场合，
我们最需要关注的部分是它的 node 成员，这实际上是一个结点列表。下面的代码演示了如何遍历这些结点：

```python
for node in graph_def.node
```

每个结点都是一个 `NodeDef` 对象，它的定义参见 [tensorflow/core/framework/node_def.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto)。
它们是构造 TensorFlow 计算图的基石，每一个结点都定义了一个操作，用来处理它的输入。下面是 `NodeDef` 的成员及其描述。

### `name`

`name` 表示结点名称，也是其唯一标识。每个结点都应该有唯一标识，而且不能和计算图中其它结点冲突。
如果在用 Python API 构造一个计算图时没有指定名称，
TensorFlow 会采用能反映操作类型的默认名称，比如 “MatMul”，然后在后面添加单调递增的数字，比如 “5”，
将其作为此结点的名称。这个名称在一些场合会用到，比如联结结点时，或在计算图运行时设置输入输出。

### `op`

`op` 表示操作类型，它定义了该结点运行何种操作。比如，`"Add"`，`"MatMul"`，或 `"Conv2D"`。当一个计算图运行时，
这个操作名被用于在 TensorFlow 注册表中寻找其具体实现。这个注册表通过调用宏 `REGISTER_OP` 来获得，类似于 
[tensorflow/core/ops/nn_ops.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc)。


### `input`

`input` 是一个字符串列表，其中每个字符串为另一个结点的名称，后面可选择性地加上冒号和那个结点的输出端口号。
比如，一个结点若有两个输入，则这个列表类似于 `["结点1的名称", "结点2的名称"]`，它又等价于
 `["结点1的名称:0", "结点2的名称:0"]`，意思是此结点的第一个输入是名为“结点1的名称”的结点的第一个输出，
而第二个输入是名为“结点2的名称”的结点的第一个输出。

### `device`

`device` 表示该结点使用的设备。在大多数情况下，你可以忽略这个成员，因为它主要针对分布式环境，
或者当你强行让它运行在 CPU 或 GPU 上时会用到。

### `attr`

`attr` 是一个字典数据结构，用键/值存储了一个结点的所有属性。它们是结点的永久性属性，即在运行时不会变化的性质，
比如卷积过滤器的尺寸，或者常值操作的值。因为属性值的类型非常之多，比如字符串、整型、张量值的数组，等等，所以需要有
一个专门的 protobuf 来定义存储这些属性的数据结构，详情参考 [tensorflow/core/framework/attr_value.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)。

每个属性都有唯一的名称字符串，在定义一个操作时，它会有一些默认的属性。如果没有为一个结点指定某个属性，
且此属性在操作的定义中又有默认值，则该默认值会用于计算图的生成。

在 Python 中，你可以通过 `node.name`，`node.op`之类的语法访问所有这些成员。
`GraphDef` 中所存的结点列表构成了计算图模型框架的完整定义。

## 冻结（Freezing）

我们常常会迷惑, 为什么训练时的权值一般并没有存储在上述文件格式中？事实上，它们被存到了单独的检查点（checkpoint）文件中。
而计算图中包含一些 `Variable` 操作（op），用于初始化时加载最新的检查点文件中的值。但是在部署到生产环境中时，使用分离的
文件并不是很方便，所以就有了脚本 [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)。
它的作用是将一个计算图定义和一些检查点文件冻结为单个文件。

在这个过程中，脚本会先加载 `GraphDef`，然后从最新的检查点文件中提取那些变量的值，
然后将每个 `Variable` 操作（op）替换为一个 `Const` 操作，这时权值被存储在了它的属性中。
之后，所有与前向推理无关的多余结点都会被剔除，最终的 `GraphDef` 被输出到了一个文件中。


## 权值格式

如果你要用 TensorFlow 模型来表示神经网络，最常见的一个问题是如何对权值进行提取和理解。

常用的存储方式是用 freeze_graph 脚本，将权值作为 `Tensors` 存储在 `Const` 操作（op）中。

这些权值定义在 [tensorflow/core/framework/tensor.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)
中，其中包含了数据大小和类型，以及这些值本身。在 Python 中，若要从表示 `Const` 操作（op）的一个 `NodeDef` 中获得一个 `TensorProto` 对象，可以
用类似于 `some_node_def.attr['value']` 的方式。

这样会得到表示权值数据的一个对象。数据本身被存到了名称以 \_val 为后缀的其中一个列表中，列表名称反映了此对象的类型，比如 `float_val` 表示32比特浮点类型。

在不同框架之间转换时，卷积层权值的存储顺序往往有些让人捉摸不透。在 TensorFlow 中，二维卷积 `Conv2D` 操作的过滤器权值被存在第二个输入上，其存储顺序为
`[filter_height, filter_width, input_depth, output_depth]`，其中 output_depth 增加1表示在内存中移向下一个相邻值。


希望通过这样一个概述，你能了解关于 TensorFlow 模型文件内部的更多细节，如果有一天你需要操作这些模型文件了，但愿本文能够有所帮助。
