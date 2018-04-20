# 保存和恢复

[需要翻译]The @{tf.train.Saver} class provides methods to save and restore models. The @{tf.saved_model.simple_save} function is an easy way to build a @{tf.saved_model$saved model} suitable for serving. [Estimators](/programmers_guide/estimators) automatically save and restore variables in the `model_dir`.

## 保存和恢复变量

[需要翻译]TensorFlow @{$variables} are the best way to represent shared, persistent state manipulated by your program. The `tf.train.Saver` constructor adds `save` and `restore` ops to the graph for all, or a specified list, of the variables in the graph. The `Saver` object provides methods to run these ops, specifying paths for the checkpoint files to write to or read from.

`Saver` restores all variables already defined in your model. If you're loading a model without knowing how to build its graph (for example, if you're writing a generic program to load models), then read the [Overview of saving and restoring models](#models) section later in this document.
 
TensorFlow saves variables in binary *checkpoint files* that map variable names to tensor values.
 
Caution: TensorFlow model files are code. Be careful with untrusted code. See [Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md) for details.

### 保存变量

用 `tf.train.Saver()` 方法创建一个 `Saver` 来管理模型中的所有变量。例如，如下代码演示了如何调用 `tf.train.Saver.save` 方法将变量保存到快照文件中：

```python
# 创建变量
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# 添加初始化变量的操作
init_op = tf.global_variables_initializer()

# 添加保存和恢复这些变量的操作
saver = tf.train.Saver()

# 然后，加载模型，初始化变量，完成一些工作，并保存这些变量到磁盘中
with tf.Session() as sess:
  sess.run(init_op)
  # 使用模型完成一些工作
  inc_v1.op.run()
  dec_v2.op.run()
  # 将变量保存到磁盘中
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)
```

### 恢复变量

`tf.train.Saver` 对象不仅能够将变量保存到快照文件中，它也能够恢复变量。注意恢复变量不需要预先初始化。例如，下面这个示例代码片段演示了如何调用 `tf.train.Saver.restore` 方法并将变量从快照文件中恢复：

```python
tf.reset_default_graph()

# 创建一些变量
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# 添加保存和恢复这些变量的操作
saver = tf.train.Saver()

# 然后，加载模型，使用 saver 从磁盘中恢复变量，并使用变量完成一些工作
with tf.Session() as sess:
  # 从磁盘中恢复变量
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # 检查变量的值
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

[需要翻译]Note: There is not a physical file called `/tmp/model.ckpt`. It is the **prefix** of filenames created for the checkpoint. Users only interact with the prefix instead of physical checkpoint files.

### 选择需要保存和恢复的变量

如果您没有传递任何参数给 `tf.train.Saver()`，保存程序将默认对计算图中所有的变量进行保存或恢复操作。每个变量都会以原变量名保存。

为快照文件中的变量明确指定名称有时是很有用的。例如，在您训练的模型中包含一个名为 `"weights"` 的变量，而你想要把 `"weights"` 变量的值恢复到名为 `"params"` 的变量中。

有时仅对部分变量进行保存和恢复操作也很有用。例如，您有一个已经训练好的五层的神经网络模型，现在想复用其权重值来训练一个六层的神经网络。那么您可以使用保存程序仅恢复前五层的权重。

通过传递如下之一的参数给 `tf.train.Saver()` 构造器，您可以轻易的指定保存和加载的名称和变量：

* 变量列表（将会以原变量名保存）。
* 一个 Python 字典，键是要使用的名称，值是要管理的变量。

继续之前展示的保存/恢复示例：

```python
tf.reset_default_graph()
# 创建一些变量
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# 使用名称“V2”创建只保存和恢复 `v2` 的操作
saver = tf.train.Saver({"v2": v2})

# 在此之后正常的使用 saver 对象
with tf.Session() as sess:
  # 由于 saver 没有初始化 v1，初始化 v1。
  v1.initializer.run()
  saver.restore(sess, "/tmp/model.ckpt")

  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
```

注意：

*  你可以随心所欲地创建多个 `Saver` 对象来保存变量的不同部分。同一变量可以在多个 `saver` 对象中列出；只有在 `Saver.restore()` 方法运行时它的值才会改变。

*  如果您仅在会话开始时恢复部分模型变量，那么您必须为其他变量运行一个初始化操作。更多信息请查阅 @{tf.variables_initializer}。

*  您可以使用 [`inspect_checkpoint`](https://www.tensorflow.org/code/tensorflow/python/tools/inspect_checkpoint.py) 库检查快照文件中的变量，`print_tensors_in_checkpoint_file` 函数尤为好用。

*  默认情况下，`Saver` 使用每个变量的 @{tf.Variable.name} 来保存变量。但是，你也可以在创建 `Saver` 对象时为快照文件中的每个变量指定名字。


### 检查快照文件中的变量

使用
[`inspect_checkpoint`](https://www.tensorflow.org/code/tensorflow/python/tools/inspect_checkpoint.py) 库可以迅速检查快照文件中的变量.

继续之前展示的保存/恢复示例：

```python
# 导入 inspect_checkpoint 库
from tensorflow.python.tools import inspect_checkpoint as chkp

# 打印快照文件中的所有张量
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# 只打印快照文件中的张量 v1
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]

# pr只打印快照文件中的张量 v2
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]
```


<a name="models"></a>
## 保存和恢复模型

[需要翻译]Use `SavedModel` to save and load your model—variables, the graph, and the graph's metadata. This is a language-neutral, recoverable, hermetic serialization format that enables higher-level systems and tools to produce, consume, and transform TensorFlow models. TensorFlow provides several ways to interact with `SavedModel`, including the @{tf.saved_model} APIs, @{tf.estimator.Estimator}, and a command-line interface.

## 创建并加载一个 SavedModel

### Simple save[需要翻译]

The easiest way to create a `SavedModel` is to use the @{tf.saved_model.simple_save} function:

```python
simple_save(session,
            export_dir,
            inputs={"x": x, "y": y},
            outputs={"z": z})
```

This configures the `SavedModel` so it can be loaded by [TensorFlow serving](/serving/serving_basic) and supports the [Predict API](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/predict.proto). To access the classify, regress, or multi-inference APIs, use the manual `SavedModel` builder APIs or an @{tf.estimator.Estimator}.

### Manually build a SavedModel

If your use case isn't covered by @{tf.saved_model.simple_save}, use the manual
@{tf.saved_model.builder$builder APIs} to create a `SavedModel`.

The @{tf.saved_model.builder.SavedModelBuilder} class provides functionality to save multiple `MetaGraphDef`s.  A **MetaGraph** is a dataflow graph, plus its associated variables, assets, and signatures.  A **`MetaGraphDef`** is the protocol buffer representation of a MetaGraph.  A **signature** is the set of inputs to and outputs from a graph.

如果需要将资源保存、写入或拷贝到磁盘，那么可以在添加第一个 `MetaGraphDef` 时提供这些资源。如果多个 `MetaGraphDef` 与同名资源相关联，则仅保留第一个版本。

添加到 SavedModel 的 `MetaGraphDef` 必须由用户指定注解的标签。标签提供了一种方法来表示要加载和恢复的特殊 `MetaGraphDef`，以及共享的变量和资源集。通常，这些标签会给 `MetaGraphDef` 添加功能性的注解（比如保存或者训练），也可以指定硬件（如 GPU）来进行注释。

例如，如下代码展示了一种使用 `SavedModelBuilder` 创建 SavedModel 的典型方法:

```python
export_dir = ...
...
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph_and_variables(sess,
                                       [tag_constants.TRAINING],
                                       signature_def_map=foo_signatures,
                                       assets_collection=foo_assets)
...
# 为推理添加一秒的 MetaGraphDef
with tf.Session(graph=tf.Graph()) as sess:
  ...
  builder.add_meta_graph([tag_constants.SERVING])
...
builder.save()
```

### 在 Python 中加载 SavedModel

Python 版本的 SavedModel
@{tf.saved_model.loader$loader}
为 SavedModel 提供了加载和恢复的能力。`load` 操作需要如下信息：

* 恢复计算图定义和变量的会话。
* 用于标识加载的 MetaGraphDe 的标签。
* SavedModel 的位置（目录）。

加载时， 指定的 MetaGraphDef 中的部分变量、资源和签名将会被恢复到目标会话中。


```python
export_dir = ...
...
with tf.Session(graph=tf.Graph()) as sess:
  tf.saved_model.loader.load(sess, [tag_constants.TRAINING], export_dir)
  ...
```


### 在 C++ 中加载 SavedModel

C++ 版本的 SavedModel
[加载器](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/loader.h)
提供了一个从路径加载 SavedModel 的API, 同时允许指定
`SessionOptions` 和 `RunOptions` 参数。
您必须指定出与被加载计算图相关的标签。SavedModel 会作为 `SavedModelBundle`加载，其中包含了 MetaGraphDef 和当前会话。

```c++
const string export_dir = ...
SavedModelBundle bundle;
...
LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagTrain},
               &bundle);
```

### 在 TensorFlow 服务中加载一个 SavedModel 并构建服务

你可以通过 ensorFlow Serving Model Server 二进制文件简便的加载 SavedModel 并构建服务。查看 [instructions](https://www.tensorflow.org/serving/setup#installing_using_apt-get)了解怎样安装服务，或者你也可以构建它。

一旦你安装好 Model Server，使用以下语句运行它：
```
tensorflow_model_server --port=port-numbers --model_name=your-model-name --model_base_path=your_model_base_path
```

设置端口和模块名称标识。model_base_path 标志应该是一个根目录，其中模型的每个版本都以数字命名子文件夹。如果只有模型一个版本，直接将其以如下方式放入子文件夹：
* 将模型放入 /tmp/model/0001
* 设置 model_base_path 为 /tmp/model

将不同版本的模型保存在同一根目录下以数字命名的子文件夹中。例如，加载根目录是 `/tmp/model`。如果你只有模型的一个版本，将其保存在 `/tmp/model/0001`。如果有模型的两个版本，保存第二个版本在`/tmp/model/0002`，以此类推。设置 `--model-base_path` 为根目录（此例中为 `/tmp/model`）。TensorFlow Model Server 会根据根目录下最高数字的子文件夹中模型构建服务。

### 标准常量

SaveModel 为多种使用案例提供了创建和加载 TensorFlow 计算图的灵活性。对于最为常见的使用案例，SavedModel 的 API 提供了一组 Python 和 C++ 中的常量，易于重复使用和一致的跨工具共享。

#### 标准 MetaGraphDef 标签

您可以使用一组标记来唯一地标识保存在 SavedModel 中的 `MetaGraphDef`。一个常用标签的子集在：

* [Python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/tag_constants.py)
* [C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/tag_constants.h)


#### 标准 SignatureDef 常量

[**SignatureDef**](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto)
是一个 Protocol Buffer，定义了计算图支持的计算中的签名。常用输入键、输出键以及方法名称在：

* [Python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/signature_constants.py)
* [C++](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/signature_constants.h)

## 配合 Estimators 使用 SavedModel

训练好 `Estimator` 模型之后，您可能想要从这个模型创建一个执行请求并返回结果的服务。您可以在您的设备上本地运行该服务，或者在云上动态部署。

要为服务准备一个训练好的 Estimator，您必须以标准的 SavedModel 格式输出它。本节介绍了如何：

* 指定能够提供的输出节点以及相应的
  [APIs](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto)
 （分类，回归或预测）。
* 以 SavedModel 格式输出模型。
* 在本地服务器上运行模型并做出预测。


### 准备运行时的输入

在训练时，@{$premade_estimators#input_fn$`input_fn()`} 提取数据并传递给模型。在服务运行时，类似的 `serving_input_receiver_fn()` 会接收推理请求并传递给模型。这个函数有以下目的：

*  为系统运行时的推理请求添加占位符。
*  添加任意额外需要的操作，用于将输入数据转换成模型所需要的特征 `Tensor`。

该该函数返回一个 @{tf.estimator.export.ServingInputReceiver} 对象，该对象将占位符和生成的特征 `Tensor` 封装到一起。

典型的模式是推理请求以序列化 `tf.Example` 的形式到达, 因此 `serving_input_receiver_fn ()` 创建一个字符串占位符来接收它们。  `serving_input_receiver_fn ()` 之后也负责解析 `tf.Example`，通过在计算图中添加 @{tf.parse_example} op。

编写这样的 `serving_input_receiver_fn ()` 时, 您必须传递一个解析说明给 @{tf.parse_example}, 以便告知分析器期望的功能名称以及如何将它们映射到 `Tensor`。解析说明是一个从功能名称映射到 @{tf.FixedLenFeature}, @{tf.VarLenFeature} 和 @{tf.SparseFeature} 的字典。注意，该解析说明不应包含任何标签或权重列, 因为这些在运行时不可用&mdash;这跟在训练时使用 `input_fn()` 的解析说明正好相反。

结合起来，然后：

```py
feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}

def serving_input_receiver_fn():
  """需要一个已序列化的 tf.Example 的输入接收器"""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
```

@{tf.estimator.export.build_parsing_serving_input_receiver_fn} 功能函数为常见案例提供了输入接收器。

> 注意：当在本地服务器上使用预测 API 训练模型时, 不需要解析步骤, 因为模型将接收原始特征数据。

即使您不需要解析或其他输入处理 — 也就是说, 如果服务系统直接给出特征 `Tensor`, 您仍然必须提供一个 `serving_input_receiver_fn ()`, 先为特征 `Tensor` 创建占位符，然后再传入张量。@{tf.estimator.export.build_raw_serving_input_receiver_fn} 工具提供了此功能。

如果这些程序还不能满足您的需求，您可以编写自己的 `serving_input_receiver_fn()`。 一种应用场景是，您训练的 `input_fn()` 包含了一些必须在运行时执行的预处理逻辑。为了降低训练向生产状态倾斜的风险，建议将这些预处理的内容封装在 `input_fn()` 和 `serving_input_reveiver_fn()` 的函数中。

注意，`serving_input_receiver_fn()` 还确定了签名的*输入*部分。也就是说，在编写 `aserving_input_receiver_fn()` 时，您必须告诉解析器所期望的签名以及如何将它们映射到模型的预期输入。相反, 签名的*输出*部分由模型确定。


### 执行输出

通过输出基本路径和 `serving_input_receiver_fn` 来调用
@{tf.estimator.Estimator.export_savedmodel}，从而输出训练的Estimator。

```py
estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn)
```

这种方法在第一次调用 `serving_input_receiver_fn()` 时创建一个新的计算图，以获取特征 `Tensor`，然后调用 `Estimator` 的 `model_fn()` 去生成基于这些特征的模型图。它创建了一个新的会话，并将最近的快照文件恢复到会话里。（如果需要，可以传递不同的快照文件。）最后，它会在给定的`export_dir_base` (即 `export_dir_base/<timestamp>`)下创建一个有时间戳的输出目录，并将一个包含了会话中的 `MetaGraphDef` 的 SavedModel 写入其中。

> 注意：请及时清理旧的输出文件。
> 否则，持续输出的文件将堆积在 `export_dir_base` 目录下。

### 指定自定义模型的输出

编写一个自定义 `model_fn` 时，必须指定 @{tf.estimator.EstimatorSpec} 的返回值 `export_outputs`。这是一个 `{name: output}` 形式的数据字典，用来描述运行期间使用和导出的签名。

通常在预测单个值的时候，作为结果的数据字典仅包含一个元素，这时候`name` 就变得无关紧要。在多头部模型中, 每个头部由这个字典中的一个条目表示。在这种情况下, `name` 可以由你自行选择，并用于在运行时请求某个特定的头部。
每一个 `output` 值都必须是一个 `ExportOutput` 对象，如
@{tf.estimator.export.ClassificationOutput},
@{tf.estimator.export.RegressionOutput}, 或者
@{tf.estimator.export.PredictOutput}。

这些输出类型直接映射到
[TensorFlow 服务 API](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto), 以此来决定要执行哪个请求。

注意: 在多头部情况下, 从 model_fn 中返回的 `export_outputs` 字典中的每一个元素都会生成一个相同键名的 `SignatureDef`。这些 `SignatureDef` 仅在其输出中有所不同, 因为由相应的 `ExportOutput` 条目所生成。输入总是由 `serving_input_receiver_fn` 提供。推理请求可以按名称指定头部。头部必须使用  [`signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`](https://www.tensorflow.org/code/tensorflow/python/saved_model/signature_constants.py) 命名，在推理请求没有指定头部时来隐式地判断哪一个  `SignatureDef` 将会被执行。


### 在本地运行导出的模型

对于本地部署，您可以使用
[TensorFlow Serving](https://github.com/tensorflow/serving)（一个加载 SavedModel 并将其暴露为 [gRPC](https://www.grpc.io/) 服务的开源项目）来运行模型。

首先， [安装 TensorFlow Serving](https://github.com/tensorflow/serving)。

然后创建并运行本地模型服务器，用以上导出的 SavedModel 路径替换 `$export_dir_base`：

```sh
bazel build //tensorflow_serving/model_servers:tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_base_path=$export_dir_base
```

现在您就有了一台服务器，通过 gRPC 在端口 9000 来监听推理请求!


### 从本地服务器请求预测

服务器根据
[PredictionService](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_service.proto#L15)
gRPC API 服务定义来响应 gRPC 请求（ 嵌套的 Protocol Buffer 定义在不同的[neighboring files](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis)中）。

根据 API 服务的定义，gRPC 框架能够生成多种语言的客户端类库，提供对 API 的远程访问。在使用 Bzael 构建工具的项目中，这些库都是自动创建并通过如下依赖关系（以使用 Python 为例）提供的：

```build
  deps = [
    "//tensorflow_serving/apis:classification_proto_py_pb2",
    "//tensorflow_serving/apis:regression_proto_py_pb2",
    "//tensorflow_serving/apis:predict_proto_py_pb2",
    "//tensorflow_serving/apis:prediction_service_proto_py_pb2"
  ]
```

Python 客户端的代码中引入类库的方式如下：

```py
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
```

> 注意：`prediction_service_pb2` 将服务定义为一个整体, 因此始终需要引入它。但是，根据所做请求的类型，典型的客户端只需要引入  `classification_pb2`、`regression_pb2` 和 `predict_pb2` 中的一个。

然后, 将请求数据组装成 Protocol Buffer 格式，并传递给服务端，至此一个 gRPC 请求完成。请注意, 请注意 Protocol Buffer 的生成方式，先创建一个空的 Protocol Buffer 然后再通过[生成的协议缓冲区 API](https://developers.google.com/protocol-buffers/docs/reference/python-generated) 进行赋值。

```py
from grpc.beta import implementations

channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = classification_pb2.ClassificationRequest()
example = request.input.example_list.examples.add()
example.features.feature['x'].float_list.value.extend(image[0].astype(float))

result = stub.Classify(request, 10.0)  # 10 secs timeout
```

本例中返回的结果是一个 Protocol Buffer 格式的 `ClassificationResponse`。

这个言简意赅的例子；更多信息请查阅 @{$deploy$Tensorflow Serving}
文档和[示例](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/example)。

> 注意：`ClassificationRequest` 和 `RegressionRequest` 包含一个 `tensorflow.serving.Input` Protocol Buffer，其中包含了一个 `tensorflow.Example` 的 Protocol Buffer 列表。不同的是， `PredictRequest` 包含了一个从特征名到特征值的映射关系，其中特征值是通过 `TensorProto` 编码的。相同的是，当调用 `Classify` 和 `Regress` API 的时候， TensorFlow 运行时会将序列化的 `tf.Examples` 输入计算图，因此 `serving_input_receiver_fn ()` 应当包含一个 `tf. parse_example ()` 操作。当调用普通的 `Predict` API 时，TensorFlow 在运行中会将原始的特征数据输入计算图，因此应当通过 `serving_input_receiver_fn ()` 进行传递。


<!-- TODO(soergel): give examples of making requests against this server, using
the different Tensorflow Serving APIs, selecting the signature by key, etc. -->

<!-- TODO(soergel): document ExportStrategy here once Experiment moves
from contrib to core. -->




## 使用 CLI 检查和执行 SavedModel

您可以使用 SavedModel 命令行接口（CLI）来检查和执行 SavedModel。例如，使用 CLI 检查模型的 `SignatureDef`。CLI 可以让您迅速确认输入
的@{$tensors$张量类型和形状}和模型相匹配。此外，如果您想要测试模型的连通性，可以使用 CLI， 通过传入各种格式(例如, Python 表达式) 的样本输入, 然后获取输出来验证。


### 安装 SavedModel CLI

广义上讲，您可以通过以下两种方式安装 TensorFlow：

*  通过安装预先构建的 TensorFlow 二进制文件。
*  通过从源码创建 TensorFlow。

如果您通过预先构建的 TensorFlow 二进制文件来安装 TensorFlow，那么 SavedModel CLI 已经安装在您系统中名为 `bin\saved_model_cli` 的路径下。

如果您是从源码创建 TensorFlow，那么您必须要运行如下额外的命令来创建 `saved_model_cli`：

```
$ bazel build tensorflow/python/tools:saved_model_cli
```

### 命令概览

SavedModel CLI 支持如下两个命令来操作 SavedModel 中的 `MetaGraphDef`:

* `show`，展示 SavedModel 中 `MetaGraphDef` 上的计算。
* `run`，运行 `MetaGraphDef` 上的计算。


### `show` 命令

一个 SavedModel 包含一个或多个 `MetaGraphDef`，通过标签集区分。要运行一个模型，您可能想要知道每个模型中 `SignatureDef` 的类型以及它们的输入输出是什么。`show` 命令允许您按分层检查 SavedModel 的内容。语法如下：

```
usage: saved_model_cli show [-h] --dir DIR [--all]
[--tag_set TAG_SET] [--signature_def SIGNATURE_DEF_KEY]
```

例如，如下命令展示了 SavedModel 中所有可用的 MetaGraphDef 标签集：

```
$ saved_model_cli show --dir /tmp/saved_model_dir
The given SavedModel contains the following tag-sets:
serve
serve, gpu
```

如下命令展示了 `MetaGraphDef` 中所有可用的 `SignatureDef` 键：

```
$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve
The given SavedModel `MetaGraphDef` contains `SignatureDefs` with the
following keys:
SignatureDef key: "classify_x2_to_y3"
SignatureDef key: "classify_x_to_y"
SignatureDef key: "regress_x2_to_y3"
SignatureDef key: "regress_x_to_y"
SignatureDef key: "regress_x_to_y2"
SignatureDef key: "serving_default"
```

如果一个 `MetaGraphDef` 在标签集中包含了**多个**标签，那么您必须标识所有标签，每个标签需要用逗号隔开，如：

```none
$ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve,gpu
```

若要显示特定 `SignatureDef` 的所有输入和输出的张量信息，需将 `SignatureDef` 键名传递给 `signature_def` 选项。这对你了解计算图执行时输入张量的键值、类型和形状非常有帮助。例如:

```
$ saved_model_cli show --dir \
/tmp/saved_model_dir --tag_set serve --signature_def serving_default
The given SavedModel SignatureDef contains the following input(s):
  inputs['x'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: x:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['y'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1, 1)
      name: y:0
Method name is: tensorflow/serving/predict
```

使用 `--all` 选项展示 SavedModel 中所有可用的信息。如：

```none
$ saved_model_cli show --dir /tmp/saved_model_dir --all
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['classify_x2_to_y3']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x2
  The given SavedModel SignatureDef contains the following output(s):
    outputs['scores'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y3:0
  Method name is: tensorflow/serving/classify

...

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: x:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['y'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: y:0
  Method name is: tensorflow/serving/predict
```


### `run` 命令

调用 `run` 命令来运行计算图计算，传递输入值，然后显示（可选保存）输出。语法如下:

```
usage: saved_model_cli run [-h] --dir DIR --tag_set TAG_SET --signature_def
                           SIGNATURE_DEF_KEY [--inputs INPUTS]
                           [--input_exprs INPUT_EXPRS] [--outdir OUTDIR]
                           [--overwrite] [--tf_debug]
```

`run` 命令提供了如下两种方式将输入数据传递到模型：

* `--inputs` 选项允许您在文件中传递 numpy ndarray。
* `--input_exprs` 选项允许您传递 Python 表达式。
* `--input_examples` option enables you to pass `tf.train.Example`.


#### `--inputs`

要在文件中传递输入数据，需指定 `--inputs` 选项，它通常用如下格式：

```bsh
--inputs <INPUTS>
```

其中, *INPUTS* 是下列格式之一: 

*  `<input_key>=<filename>`
*  `<input_key>=<filename>[<variable_name>]`

你可以传递多个 **INPUT**。如果您确实传递了多个输入，请使用分号分隔每个 *INPUTS*。

`saved_model_cli` 使用 `numpy.load` 加载**文件名**。**文件名**可能是以下任一格式：

*  `.npy`
*  `.npz`
*  pickle 格式

`.npy` 文件总是包含一个 numpy ndarray。因此，从 `.npy` 文件加载内容时，文件内容将被直接赋值给指定的输入张量。如果您指定了包含此 `.npy` 文件的 **variable_name**，**variable_name** 将被忽略，且会发出警告。

从 `.npz` (zip) 文件加载时，您可以选择性的指定一个 **variable_name** 来标识 zip 文件中的变量，以此作为输入张量的值。如果不指定 **variable_name**，SavedModel CLI 将会检查 zip 文件中是否只包含一个文件，并将其赋值给指定的张量。

从 pickle 文件加载内容时，如果方括号内没有指定 `variable_name`，则无论 pickle 文件中内容是什么，都将被赋值给指定的张量。否则，SavedModel CLI 将假定 pickle 中保存了一个数据字典, 并且将使用与对应 **variable_name** 的值。


#### `--inputs_exprs`

若通过 Python 表达式传递输入, 请指定 `--input_exprs` 选项。这在你没有任何数据文件但仍想通过一些符合 `SignatureDef` 类型、形状定义的输入数据来检查模型的连通性时会很有用。例如：

```bsh
`<input_key>=[[1],[2],[3]]`
```

除了 Python 表达式外, 您还可以传递 numpy 函数。例如:

```bsh
`<input_key>=np.ones((32,32,3))`
```

(请注意，`numpy` 模块已经可以作为 `np` 使用。)

#### `--inputs_examples`

要将 `tf.train.Example` 当做输入传入，指定 `--input_examples` 选项。其中每个键值都是字典，这些字典都是一个 `tf.train.Example`的实例。字典中的键值是特性，对应特性的值列表。
例如：

```bsh
`<input_key>=[{"age":[22,24],"education":["BS","MS"]}]`
```

#### 保存输出

默认情况下，SavedModel CLI 将输出写入 stdout。如果传了一个目录给 `--outdir` 选项，输出内容将会以输出张量的键名保存在指定目录的 npy 文件中。

使用 `--overwrite` 覆盖现有输出文件。


#### TensorFlow 调试器 (tfdbg) 集成

如果设置了 `--tf_debug` 选项, 则 SavedModel CLI 将使用 TensorFlow 调试器 (tfdbg) 在运行 SavedModel 时监视过渡张量、运行的计算图或子图。


#### `run` 的完整示例

已知：

*  模型只是 `x1` 和 `x2` 相加获得输出 `y`。
*  模型中所有张量具有形状 `(-1, 1)`。
*  您有两个 `npy` 文件：
   *  `/tmp/my_data1.npy`, 包含一个 numpy ndarray `[[1], [2], [3]]`.
   *  `/tmp/my_data2.npy`, 包含另一个 numpy
      ndarray `[[0.5], [0.5], [0.5]]`.

通过模型运行两个 `npy` 文件以获取输出 `y`, 请使用以下命令:

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npy;x2=/tmp/my_data2.npy \
--outdir /tmp/out
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
```

让我们稍微改变一下前面的例子。这一次, 您有一个 `.npy` 文件和一个 pickle 文件，而不是两个 `.npy` 文件。此外, 还要覆盖任何现有的输出文件。命令如下：

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y \
--inputs x1=/tmp/my_data1.npz[x];x2=/tmp/my_data2.pkl --outdir /tmp/out \
--overwrite
Result for output key y:
[[ 1.5]
 [ 2.5]
 [ 3.5]]
```

您可以指定 python 表达式代替输入文件。例如，如下命令用一个 python 表达式替代了输入 `x2`:

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def x1_x2_to_y --inputs x1=/tmp/my_data1.npz[x] \
--input_exprs 'x2=np.ones((3,1))'
Result for output key y:
[[ 2]
 [ 3]
 [ 4]]
```

使用 TensorFlow 调试器运行模型, 请使用如下命令:

```
$ saved_model_cli run --dir /tmp/saved_model_dir --tag_set serve \
--signature_def serving_default --inputs x=/tmp/data.npz[x] --tf_debug
```


<a name="structure"></a>
## SavedModel 目录结构

当您以 SavedModel 格式保存模型时，TensorFlow 会创建一个包含如下子目录和文件的 SavedModel 目录：

```bsh
assets/
assets.extra/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb|saved_model.pbtxt
```

其中：

* `assets` 是包含辅助 (外部) 文件的子文件夹，如词汇表。 资源文件被复制到 SavedModel 目录, 并可在加载特定 `MetaGraphDef` 时被读取。
* `assets.extra` 是一个子文件夹，其中较高级库和用户可以添加与模型共存的自己的资源，但不由计算图加载。该子文件夹不由 SavedModel 库管理。
* `variables` 是一个包含 `tf.train.Saver` 输出的子文件夹。
* `saved_model.pb` 或 `saved_model.pbtxt` 是 SavedModel 的 Protocol Buffer 数据，包含了 `MetaGraphDef` Protocol Buffer 格式的计算图定义的内容。

单个 SavedModel 可以表示多个计算图。在这种情况下, SavedModel 中的所有计算图共享一组检查点 (变量) 和资源。例如, 下图显示了一个包含 3 个 `MetaGraphDef` 的 SavedModel, 三个计算图共享同一组快照和资源: 

![SavedModel represents checkpoints, assets, and one or more MetaGraphDefs](../images/SavedModel.svg)

每个计算图都与一组特定的标签相关联, 能够在加载或还原操作期间识别不同的计算图。
