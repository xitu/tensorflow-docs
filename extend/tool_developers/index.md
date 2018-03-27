# 工具开发者指南：TensorFlow 模型文件

大部分用户不需要关心 TensorFlow 如何在磁盘上存储数据的内部细节，但如果你是一个工具开发者，就不一样了。比如，你可能想要分析模型，或者在 TensorFlow 格式和其它格式之间来回转换。本指南试图解释如何处理保存模型数据的主文件的一些细节, 以使开发这些工具变得更容易。

[目录]

## 协议缓存（Protocol Buffers）

所有的 TensorFlow 文件格式都是基于 [Protocol Buffers](https://developers.google.com/protocol-buffers/?hl=en) ，所以正式开始之前有必要先熟悉一下它的工作原理。简单来说，你用文本文件定义数据结构，然后使用 protobuf 工具生成 C、Python 和其它语言中的类，这样，开发者就可以用一种友好的方式加载、保存和访问数据。我们经常将协议缓存（Protocol Buffers）称为 protobufs，本指南会一直使用该约定。

## GraphDef 类

TensorFlow 计算的基础是 `Graph` 对象，称为计算图。计算图拥有一个节点网络，每个节点表示一种操作，彼此连接作为输入和输出。创建了 `Graph` 对象之后，你可以通过调用 `as_graph_def()`将它保存起来，该调用返回的是一个 `GraphDef` 对象。

GraphDef 类是由 ProtoBuf 库根据 [tensorflow/core/framework/graph.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto) 定义创建的对象。protobuf 工具分析此文本文件，并生成用于加载、存储和操作计算图定义的代码。如果你看到表示模型的单个 TensorFlow 文件，则它很有可能包含了某个 `GraphDef` 对象的序列化版本，而且它是用 protobuf 代码保存的。

这种生成的代码被用于在磁盘上保存和加载 GraphDef 文件，实际加载模型的代码如下所示：

```python
graph_def = graph_pb2.GraphDef()
```

此行创建一个空的 `GraphDef` 对象，即从 graph.proto 中的文本定义创建的类。接下来我们就用这个对象来从文件中读取数据。

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

ProtoBuf 实际上支持两种不同的文件保存格式。TextFormat 是一种人眼可读的文本形式，这在调试和编辑时是很方便的，但它在存储数值数据时会变得很大，比如我们常见的权重数据。这种格式的一个小的示例参见 [graph_run_run2.pbtxt](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/demo/data/graph_run_run2.pbtxt)。

相比于文本格式，二进制格式的文件会小得多，缺点就是它不易读。在脚本中，我们会要求用户提供一个标志，指示输入文件是二进制还是文本，然后再根据这个标志来调用相应的函数。你可以在 [inception_v3 archive](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz) 中找到一个较大的二进制文件的示例 `inception_v3_2016_08_28_frozen.pb` 。

API 本身可能有点混乱 - 二进制格式的调用实际上是 `ParseFromString()`，而文本格式的加载则用到 `text_format` 模块中的一个工具函数。

## 节点

一旦将文件加载到 `graph_def` 变量中，你现在就可以访问其中的数据了。对于大多数实际应用场合，重要部分是存储在节点成员中的节点列表。下面的代码演示了如何遍历这些节点：

```python
for node in graph_def.node
```

每个节点都是一个 `NodeDef` 对象，定义在 [tensorflow/core/framework/node_def.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto)。它们是构造 TensorFlow 计算图的基石，每一个节点都定义了一个简单操作以及输入连接。下面是 `NodeDef` 的成员及其描述。

### `name`

每个节点都应该有一个唯一的标识符，而且不能和计算图中的其它节点冲突。如果在用 Python API 构造一个计算图时没有指定名称，TensorFlow 会采用能反映操作类型的默认名称，比如 “MatMul”，然后在后面添加单调递增的数字，比如 “5”，将其作为此节点的名称。这个名称在一些场合会用到，比如连接节点时，或在计算图运行时设置输入输出。

### `op`

`op` 定义了要运行的操作，比如，`"Add"`，`"MatMul"`，或 `"Conv2D"`。当一个计算图运行时，
这个操作名被用于在 TensorFlow 注册表中寻找其具体实现。这个注册表通过调用宏 `REGISTER_OP` 来获得，类似于 
[tensorflow/core/ops/nn_ops.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc)。


### `input`

`input` 是一个字符串列表，其中每个字符串为另一个节点的名称，后面可选择性地加上冒号和那个节点的输出端口号。比如，一个节点若有两个输入，则这个列表类似于 `["some_node_name", "another_node_name"]`，它又等价于 `["some_node_name:0", "another_node_name:0"]`，意思是此节点的第一个输入是名为`“some_node_name”`的节点的第一个输出，而第二个输入是名为`“another_node_name”`的节点的第一个输出。

### `device`

`device` 表示该节点使用的设备。在大多数情况下，你可以忽略这一点，因为它主要针对分布式环境，
或者当你强行让它运行在 CPU 或 GPU 上时会用到。

### `attr`

`attr` 是一个字典数据结构，用键/值存储了一个节点的所有属性。它们是节点的永久属性，即在运行时不会变化，比如卷积过滤器的尺寸，或者常值操作的值。因为属性值的类型非常之多，从字符串、到整型、到张量值的数组，等等，所以需要有一个专门的 protobuf 文件来定义存储这些属性的数据结构，详情参考 [tensorflow/core/framework/attr_value.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)。

每个属性都有一个唯一的名称字符串，并且在定义操作时会列出预期的属性。如果节点中不存在某个属性，
但它在操作定义中又列出了默认值，则在创建计算图时将使用默认值。

在 Python 中，你可以通过调用 `node.name`，`node.op` 等方法访问所有这些成员。`GraphDef` 中存储的节点列表构成了计算图模型框架的完整定义。

## 冻结（Freezing）

令人困惑的是，训练时的权值通常不会存储在上述文件格式中。相反，它们被保存在单独的检查点（checkpoint）文件中，而计算图中包含一些 `Variable` 操作（op），用于初始化时加载最新的检查点文件中的值。但是在部署到生产环境中时，使用分离的文件并不是很方便，所以就有了脚本 [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)。它的作用是将一个计算图定义和一些检查点文件冻结为单个文件。

在这个过程中，脚本会先加载 `GraphDef`，然后从最新的检查点文件中提取那些变量的值，
然后将每个 `Variable` 操作替换为一个 `Const` 操作，这时权值被存储在了它的属性中。
之后，所有与前向推理无关的多余节点都会被剔除，最终的 `GraphDef` 被输出到了一个文件中。


## 权值格式

如果你要用 TensorFlow 模型来表示神经网络，最常见的一个问题是如何对权值进行提取和理解。常用的存储方式是用 freeze_graph 脚本，将权值作为 `Tensors` 存储在 `Const` 操作中。这些权值定义在 [tensorflow/core/framework/tensor.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)
中，其中包含了数据大小和类型，以及这些值本身。在 Python 中，通过调用诸如 `some_node_def.attr['value'].tensor` 之类的操作，可以从表示 `Const` 操作的 `NodeDef` 中获得一个 `TensorProto` 对象。

这样会得到表示权值数据的一个对象。数据本身被存储到了名称以 \_val 为后缀的其中一个列表中，列表名称反映了此对象的类型，比如 `float_val` 表示 32位 浮点数据类型。

在不同框架之间转换时，卷积层权值的存储顺序往往有些让人捉摸不透。在 TensorFlow 中，二维卷积 `Conv2D` 操作的过滤器权值被存储在第二个输入上，其存储顺序为
`[filter_height, filter_width, input_depth, output_depth]`，其中 filter_count 增加 1 表示在内存中移向下一个相邻值。

希望通过这样一个概述，你能更好地了解关于 TensorFlow 模型文件的内部细节，如果有一天你需要操作这些模型文件了，但愿本文能够有所帮助。
