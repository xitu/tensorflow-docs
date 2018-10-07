# TensorFlow 版本兼容性

本文适用于需要保持不同版本 TensorFlow 代码及数据向后兼容性的使用者，以及旨在修改 TensorFlow 的同时保持兼容性的开发人员。

## 语义化版本 2.0

TensorFlow 的公共 API 沿袭自语义化版本 2.0（[semver](http://semver.org)）。 每个 TensorFlow 发布版本号都以 `MAJOR.MINOR.PATCH` 的形式命名（译注：“主版本.副版本.补丁版本”）。例如，TensorFlow 1.2.3 版本的 `MAJOR` 为 1，`MINOR` 为 2，`PATCH` 为 3。每个版本号的更改具有以下含义：

* **MAJOR**： 更改可能不具有向后兼容性。之前发布的版本中所运行的代码和数据在新版本中用不上了。然而，有些情况下现有的 TensorFlow 图和检验点最好可以迁移到新版本。查看 [Compatibility of graphs and checkpoints](#compatibility_of_graphs_and_checkpoints) 以获取数据兼容性的细节。

* **MINOR**：向后兼容特性和速度的改善等。之前发布的版本中所运行的代码和数据仅依赖于公共 API，它们将不加改动地继续运行。如果想查阅公共 API 和非公共 API 的细节信息，请移步 [What is covered](#what_is_covered)。

* **PATCH**：向后兼容性 bug 的修复。

例如 1.0.0 发布版本基于 0.12.1 发布版本引入了不具有向后兼容性的改动。然而，1.1.1 发布版本则向后兼容 1.0.0 发布版本。

## 涉及的内容

TensorFlow 中只有公共 API 在副版本和补丁版本之间兼容。公共 API 由以下几部分组成：

* `tensorflow`模块及其子模块中记录在册的全部 [Python](../api_docs/python) 函数和类，除了：
    * `tf.contrib` 中的函数和类
    * 以 `_` 开头命名的函数和类（因为它们是私有的）
    * 名称以 `experimental` 开头的函数、参数、属性和类，或者限定名称必须包含 `experimental` 的模块
  请注意 `examples/` 和 `tools/` 路径下的代码无法通过 `tensorflow` 的 Python 模块访问，因此无法保证其兼容性。

  如果某个符号可以被 `tensorflow` 模块及其子模块调用但没有被记录在册，它被认为不属于公共 API 的一部分。

* [C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h)。

* 下列 protocol buffer 文件：
    * [`attr_value`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto)
    * [`config`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto)
    * [`event`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/event.proto)
    * [`graph`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto)
    * [`op_def`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_def.proto)
    * [`reader_base`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/reader_base.proto)
    * [`summary`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto)
    * [`tensor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto)
    * [`tensor_shape`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto)
    * [`types`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto)

<a name="not_covered"></a>
## 未涉及的内容

某些 API 函数被显式标记为“实验性”，它们可以在不同副版本之间进行非兼容性改动，包括：

* **实验性的 APIs**：Python 中的 `tf.contrib` 块及其子模块、C API 中的全部函数，以及 protocol buffers 中标记为实验性的字段。尤其是某个 protocol buffer 中被叫做“实验性”的域，其内的全部字段和子消息可以随时改动。

* **其他语言**：除 Python 和 C 外的其他语言编写的 TensorFlow APIs，这些语言包括：

  - [C++](../api_guides/cc/guide.md)（通过头文件 [`tensorflow/cc`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/cc) 公开）
  - [Java](../api_docs/java/reference/org/tensorflow/package-summary)
  - [Go](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)
  - [JavaScript](https://js.tensorflow.org)

* **组合操作的细节**：许多 Python 中的公共函数扩展为图中的若干原语，这些细节将会是任何以 `GraphDef` 形式保存到磁盘中的图的一部分。这些细节可能在副版本之间发生改动。 特别地，即使图的行为并未发生变化且存在将要运行的检验点，回归测试也可能跨副版本以检测各图之间的严格适配性。

* **浮点数细节**：操作数计算所得的具体浮点数随时可能发生改变。使用者应该仅依赖于浮点数的大约的精度特性和数值稳定性，而不是具体的按位计算。在副版本或在补丁版本中改变数值计算公式应得到同样或更高的精度。额外说明一下，在机器学习中，提高特定公式的精度可能会使整体系统精度下降。

* **随机数**：由 [random ops](../api_guides/python/constant_op.md#Random_Tensors) 计算所得的具体随机数随时可能发生改变。使用者应该仅依赖于随机数的大约的分布精确性和统计强度，而不是具体的按位计算。然而我们很少甚至绝不在补丁版本中改变随机数位，当然，相关文档会发生改变。

* **分布式 Tensorflow 中的版本偏差**： 不支持在同一个群集中运行两种不同的 TensorFlow 版本。多从机通信有线协议不会保证向后兼容性。

* **Bugs**：如果当前的实现明显出现故障，也就是说，实现与文档矛盾或由于 bug 而无法恰当地表现出已知且明确的预期行为。在这种情况下，我们保留修改向后不兼容行为（并非由API实现）的权利。例如，如果一个已知的算法优化程序因为 bug 的存在而无法适配那个算法，我们就会修复这个程序。我们将根据错误行为侵入代码并定位到错误源头。我们将会在发布版本中记录这种改变。

* **错误信息**：我们保留改变错误信息文本的权利。另外，只有那些在文档中记录的特定错误类型才可能发生改变。例如在文档中，可能抛出 `InvalidArgument` 异常的某个函数仍然会抛出  `InvalidArgument` 异常，改变的只是供我们阅读的信息内容。

## 图和检验点的兼容性

有时你需要保留图形和检验点。图形描述训练期间将被执行的操作的数据流和训练结果，检验点包含图中已保存的变量的张量值。

许多 TensorFlow 使用者将图和训练好的模型保存到磁盘以期为后续评估或另外的训练所使用，最终却在后续发布版本中运行它们。遵从语义版本的约定，TensorFlow 生成的任何图或检验点能够被后续相同主版本号的 TensorFlow 加载和评估。然而，如果可能的话，我们甚至会在不同主版本号之间尽力保留向后兼容性，以使序列化文件能够长期使用。

图形可以通过 `GraphDef` protocol buffer 序列化。为了推进图的向后不兼容改变的实行（极少出现这种情况），每个 `GraphDef` 都有一个独立于 TensorFlow 版本的版本号。例如，`GraphDef` 版本 17 不推荐使用 `inv` 操作，而推荐使用 `reciprocal`。它有以下语义：

* 每个 TensorFlow 版本支持 GraphDef 间隔版本。间隔版本不随补丁版本发布而改变，只有副版本发布时才会改变。在 TensorFlow 主版本发布时，才会发生放弃对某个 `GraphDef` 版本的支持这种情况。

* 新建的图会分配到最新的 `GraphDef` 版本号。

* 如果某个给定的 TensorFlow 版本支持一个图的  `GraphDef` 版本，它将会以和当初生成这个图的 TensorFlow 版本同样的行为加载并评估它（浮点数数值细节和随机数除外），不论 TensorFlow 的主版本是多少。尤其要注意，所有的检验点文件将会是兼容的。

* 如果一个发布版（副版本）中的 `GraphDef` 增加到了上限 X，那么要等下限增加到 X,至少需要六个月。例如（在这里，版本号是我们假定的）：
    * TensorFlow 1.2 版本可能支持 4 到 7 版本号的 `GraphDef` 版本。
    * TensorFlow 1.3 可以添加 `GraphDef` 版本 8 并且支持 4 to 8 版本号的版本。
    * 至少六个月后，TensorFlow 2.0.0 才能放弃支持 4 到 7 版本号的 `GraphDef` 版本而仅保留版本 8的 `GraphDef`。

最后，当 `GraphDef` 版本不再被支持时，我们将会尝试提供一系列工具用以将图自动转换为更新的被支持的 `GraphDef` 版本。

## 扩展 TensorFlow 时的图和检验点兼容性

只有 `GraphDef` 格式产生不兼容的更改时，本节内容才具有相关性。这些更改包括添加操作、移除操作以及更改现有操作的功能。对于多数使用者，前面几节已经足够。

<a id="backward_forward"/>

### 向后兼容性和部分向前兼容性

我们的版本体系有三个要求：

* **向后兼容性**应支持加载由老版本 TensorFlow 创建的图和检验点。
* **向前兼容性**应支持图和检验点的生产者先于消费者升级 TensorFlow 的版本这种情况。
* 能够以不兼容方式更迭。例如，移除操作、增加属性以及移除属性。

需要注意的是，虽然 `GraphDef` 版本机制独立于 TensorFlow 版本，不向后兼容的 `GraphDef` 格式改变仍受限于语义化版本号。这意味着只能在主版本之间移除或更改功能，比如 `1.7` 版本升至 `2.0` 版本。此外，各个补丁版本之间被强制保持向前兼容性，比如 `1.x.1` 版本升至 `1.x.2` 版本。 

为了实现向后和向前兼容性，并且为了解何时该强制改变格式，图和检验点拥有描述自己何时生成的元数据。下节将详述 `GraphDef` 的 TensorFlow 实现和版本更迭指南。

### 独立数据版本体系

图和检验点有不同的数据版本。这两种数据格式的更迭速率彼此不同，也不同于 TensorFlow 的更迭速率。两种版本系统均定义于 [`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h)。
每添加一个新版本，就会有一条记录被添加到头文件以详细描述更改的内容以及更改日期。

### 数据、生产者以及消费者

我们会区分以下种类的数据版本信息：

* **生产者**：生产数据的二进制文件。生产者拥有一个（`producer`）版本，以及一个和（`min_consumer`）兼容的最低消费者版本。
* **消费者**：生产数据的二进制文件。消费者有一个（`consumer`）版本，以及一个和（`min_producer`）兼容的最低生产者版本。

每个版本数据段都有一个 [`VersionDef
versions`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto) 字段用于记录生成这些数据的 `producer`、与之兼容的 `min_consumer`，以及一个不被接受的 `bad_consumers` 版本列表。

默认情况下，当一名生产者生成一些数据，这些数据会继承生产者的 `producer` 和 `min_consumer` 版本。如果已经知道特定版本包含 bugs 且必须被避免，也可以设置 `bad_consumers` 。如果下列条件全部成立，消费者就可以接受数据段：

* `consumer` >= 数据的 `min_consumer`
* 数据的 `producer` >= 消费者的 `min_producer`
* `consumer` 不存在于数据的 `bad_consumers` 内

由于生产者和消费者均来源于 TensorFlow 代码库，[`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h) 包含一个视上下文及其 `min_consumer` 和  `min_producer`（分别为生产者和消费者所需）情况而解析为 `producer` 或 `consumer` 的主数据版本。特别地，

* 对于 `GraphDef` 版本，我们使用 `TF_GRAPH_DEF_VERSION`、	`TF_GRAPH_DEF_VERSION_MIN_CONSUMER`，以及 `TF_GRAPH_DEF_VERSION_MIN_PRODUCER` 标识。
* 对于检验点版本，我们使用 `TF_CHECKPOINT_VERSION`、`TF_CHECKPOINT_VERSION_MIN_CONSUMER`，以及 `TF_CHECKPOINT_VERSION_MIN_PRODUCER` 标识。

### 向现有 op 添加一个新的属性（默认）

遵循下面的指导，只有 ops 集合未修改的情况下，才能给出前向兼容性：

1. 如果需要向前兼容，请将 `strip_default_attrs` 设置为 `True`。在导出模型时，使用 `tf.saved_model.builder.SavedModelBuilder.add_meta_graph_and_variables` 和 `tf.saved_model.builder.SavedModelBuilder.add_meta_graph` 方法的 `SavedModelBuilder` 类，或者是 `tf.estimator.Estimator.export_savedmodel`。
2. 这将在生成/导出模型时去掉默认值属性。这确保在使用默认值时导出的 `tf.MetaGraphDef` 不包含新的 op 属性。
3. 使用此控件可以允许过时的消费者（例如，提供落后于训练二进制文件的二进制文件）继续加载模型并防止模型服务中心的中断。

### GraphDef 版本更迭

本节将阐述如何利用版本机制来区分 `GraphDef` 格式的改变。

#### 增加一个操作

同时为消费者和生产者添加一个新操作，并且不改变任何 `GraphDef` 版本。这种类型的改变自动向后兼容，而且不影响向前兼容性计划，因为现有生产者脚本不会突然使用新功能。

#### 增加一个操作并使现有 Python 包装器转而使用它

1.  实现新的消费者功能并递增 `GraphDef` 版本。
2.  如果能使包装器使用过去没有的新功能，那么可以立刻更新包装器。
3.  更改 Python 包装器以使用新功能。不要递增 `min_consumer` ，因为未使用新功能的机型并不会停止运行。

#### 移除或限制某个操作的功能

1. 修复全部生产者脚本（不是 TensorFlow 本身）以避免使用禁用的操作和功能。
2. 递增 `GraphDef` 版本，实现新的禁止在新版本及其后续版本中移除 GraphDefs 操作或功能的消费者功能。可能的话，可以使用禁用功能使 TensorFlow 不生成 `GraphDefs`。想要这么做的话，请添加 [`REGISTER_OP(...).Deprecated(deprecated_at_version,message)`](https://github.com/tensorflow/tensorflow/blob/b289bc7a50fc0254970c60aaeba01c33de61a728/tensorflow/core/ops/array_ops.cc#L1009)。
3. 等待兼顾向后兼容性的主版本的发布。
4. 在第（2）点中递增 GraphDef 的 `min_producer`，并且完全移除其功能。

#### 更改某个操作的功能

1. 增加一个新的名为 `SomethingV2` 或 similer 的相似操作，并经历添加且使现有 Python 包装器转而使用它的过程。要确保向前兼容性，请在更改 Python 包装时进行 [compat.py](https://www.tensorflow.org/code/tensorflow/python/compat/compat.py) 中建议的检查。
2. 移除旧操作（由于要保持向后兼容性，只能改变主版本号）。
3. 递增 `min_consumer` 以使消费者无法使用旧操作，将旧操作以别名 `SomethingV2` 添加回去，并经历添加且使现有 Python 包装器转而使用它的过程。
4. 经历移除 `SomethingV2` 的过程。

#### 禁用单一不安全版本

1. 为全部的新 GraphDef 替换掉 `GraphDef` 版本，并且在 `bad_consumers` 中添加错误版本。如果可能的话，只为包含确定操作或相似操作的 GraphDef 添加 `bad_consumers`。
2. 如果现有消费者拥有错误版本，请尽快淘汰这些版本。
