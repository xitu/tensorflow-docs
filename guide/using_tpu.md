# 使用 TPU

这份文档说明了有效使用 [Cloud TPU](https://cloud.google.com/tpu/) 时必需使用的关键 TensorFlow API，并强调了常规的 TensorFlow 和在 TPU 上使用区别。

这份文档针对以下用户：

* 熟悉 TensorFlow 的 `Estimator` 和 `Dataset` API
* 使用一个已有模型[尝试使用过 Cloud TPU](https://cloud.google.com/tpu/docs/quickstart)
* 浏览过 TPU 模型的样例代码 [[1]](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_tpu.py) [[2]](https://github.com/tensorflow/tpu/tree/master/models)
* 对将一个现有的 `Estimator` 模型移植到 Cloud TPU 上运行感兴趣

## TPUEstimator

`tf.estimator.Estimator` 是 TensorFlow 的模型级抽象。标准的 `Estimators` 可以在 CPU 或者 GPU 上驱动模型。 你必须使用 `tf.contrib.tpu.TPUEstimator` 在 TPU 上驱动模型。

使用[预制的 `Estimator`](../guide/premade_estimators.md) 和[个性化 `Estimator`](../guide/custom_estimators.md) 的基础介绍可以参考 TensorFlow 的入门指南（Getting Started）部分，

`TPUEstimator` 类和 `Estimator` 之间多少有些不一样。

要使一个模型可以在 CPU/GPU 或 Cloud TPU 上运行的最简单方法是在 `model_fn` 外定义模型的推理过程（从输入到预测）。然后继续分离 `Estimator` 设置和 `model_fn`，都包含这个推理步骤。这种模式的一个样例是 [tensorflow/models](https://github.com/tensorflow/models/tree/master/official/mnist) 中比较 `mnist.py` 和 `mnist_tpu.py` 的实现。

### 本地运行 `TPUEstimator`

要创建一个标准的 `Estimator` 你可以调用构造函数，并将它传递给 `model_fn`，例如：

```
my_estimator = tf.estimator.Estimator(
  model_fn=my_model_fn)
```

在本地计算机上使用 `tf.contrib.tpu.TPUEstimator` 所需的更改相对较小。构造函数还需要另外两个参数。您应该将 `use_tpu` 参数设置为 `False`，并将 `tf.contrib.tpu.RunConfig` 作为 `config` 参数传入，如下所示：


``` python
my_tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=my_model_fn,
    config=tf.contrib.tpu.RunConfig()
    use_tpu=False)
```

这样简单的更改就能在本地运行 `TPUEstimator`。大多数 TPU 模型示例都可以以下面这种命令行设置标志参数，来在本地模式下运行：

```
$> python mnist_tpu.py --use_tpu=false --master=''
```

注意：`use_tpu=False` 参数对于尝试 `TPUEstimator` API 很有用。这也就意味着它不是个完整的 TPU 兼容测试。在 `TPUEstimator` 中成功地本地运行一个模型并不代表它能在 TPU 上运行。

### 构建一个 `tpu.RunConfig`

虽然默认的 `RunConfig` 足以进行本地训练，但在实际使用并不能忽略这些设置。

一种可以切换到 Cloud TPU 的更典型 `RunConfig` 设置，会如下所示：

``` python
import tempfile
import subprocess

class FLAGS(object):
  use_tpu=False
  tpu_name=None
  # 为 `model_dir` 设定本地临时路径
  model_dir = tempfile.mkdtemp()
  # 在返回控制之前在 Cloud TPU 上运行的训练步数
  iterations = 50
  # 一个包含 8 个分片的 Cloud TPU
  num_shards = 8

if FLAGS.use_tpu:
    my_project_name = subprocess.check_output([
        'gcloud','config','get-value','project'])
    my_zone = subprocess.check_output([
        'gcloud','config','get-value','compute/zone'])
    cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_names=[FLAGS.tpu_name],
            zone=my_zone,
            project=my_project)
    master = tpu_cluster_resolver.get_master()
else:
    master = ''

my_tpu_run_config = tf.contrib.tpu.RunConfig(
    master=master,
    evaluation_master=master,
    model_dir=FLAGS.model_dir,
    session_config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True),
    tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations,
                                        FLAGS.num_shards),
)
```

然后你必须将 `tf.contrib.tpu.RunConfig` 传入构造函数：

``` python
my_tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=my_model_fn,
    config = my_tpu_run_config,
    use_tpu=FLAGS.use_tpu)
```

通常，`FLAGS` 将由命令行参数设置。要从本地训练转换为 Cloud TPU 训练，你需要：

* 设置 `FLAGS.use_tpu` 为 `True`
* 设置 `FLAGS.tpu_name`，以便 `tf.contrib.cluster_resolver.TPUClusterResolver` 可以找到它
* 设置 `FLAGS.model_dir` 为一个 Google Cloud Storage 容器地址（`gs://`）。

## 优化器

在 Cloud TPU 上进行训练时，**必须**将优化器装饰在 `tf.contrib.tpu.CrossShardOptimizer` 中，该优化器使用 `allreduce` 聚合斜率结果并将结果广播到每个分片（每个 TPU 核心）。

`CrossShardOptimizer` 不兼容本地训练。因此，如果要在本地和 Cloud TPU 上运行相同的代码，请添加如下代码：

``` python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
if FLAGS.use_tpu:
  optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
```

如果想在模型代码中避免使用全局 `FLAGS` ，一种方法就是将优化器设置为 `Estimator` 的参数之一，如下所示：

``` python
my_tpu_estimator = tf.contrib.tpu.TPUEstimator(
    model_fn=my_model_fn,
    config = my_tpu_run_config,
    use_tpu=FLAGS.use_tpu,
    params={'optimizer':optimizer})
```

## 模型函数

本节详细介绍了使模型函数（`model_fn()`）能与 `TPUEstimator` 兼容所要做的必要更改。

### 静态形状

在常规使用过程中，TensorFlow 试图在流图构造过程中确定每个 `tf.Tensor` 的形状。在执行过程中，任何未知的形状维度都会被动态确定，更多内容请参阅 [Tensor Shapes](../guide/tensors.md#shape)。

要在 Cloud TPU 上运行，TensorFlow 模型需要是由 [XLA](../performance/xla/index.md) 编译的。XLA 在编译时使用类似的系统确定形状。XLA 要求在编译时所有张量维都被静态定义了。所有形状都必须转换为一个常量，且并不依赖外部数据或有状态操作（如变量或随机器数生成器）。

### 摘要

将模型中所有的 `tf.summary` 都删除。

[TensorBoard 摘要](../guide/summaries_and_tensorboard.md)是查看模型内部的一个好办法。`TPUEstimator` 会将一个基础摘要的最小摘要记录到 `model_dir` 的 `event` 文件中。然而，在 Cloud TPU 上进行训练时，目前不支持自定义摘要。因此，虽然 `TPUEstimator` 仍然可以包含摘要在本地运行，但在 TPU 上会运行失败。

### 评估标准

在一个独立的 `metric_fn` 中构建评估指标字典。

<!-- TODO(markdaoust) link to guide/metrics when it exists -->

评估指标是训练的重要部分。Cloud TPU完全支持这些功能，但语法略有不同。

一个标准的 `tf.metrics` 返回两个张量。第一个返回公制值的运行平均值，而第二个更新运行平均值并返回此批次的值：

```
running_average, current_batch = tf.metrics.accuracy(labels, predictions)
```

在标准的 `Estimator` 中，创建这些张量对的字典，并将其作为 `Estimator` 的一部分返回。

```python
my_metrics = {'accuracy': tf.metrics.accuracy(labels, predictions)}

return tf.estimator.EstimatorSpec(
  ...
  eval_metric_ops=my_metrics
)
```

相反，在 `TPUEstimator` 中，传递一个函数（返回一个度量词典）和一个参数张量列表，如下所示：


```python
def my_metric_fn(labels, predictions):
   return {'accuracy': tf.metrics.accuracy(labels, predictions)}

return tf.contrib.tpu.TPUEstimatorSpec(
  ...
  eval_metrics=(my_metric_fn, [labels, predictions])
)
```

### 使用 `TPUEstimatorSpec`

`TPUEstimatorSpec` 不支持钩子，并且某些字段需要函数装饰器。

`Estimator` 的 `model_fn` 必须返回 `EstimatorSpec`。`EstimatorSpec` 是一种简单结构的命名字段，它包含模型中可能需要与 `Estimator` 交互的所有 `tf.Tensors`。

`TPUEstimators` 使用一个 `tf.contrib.tpu.TPUEstimatorSpec`。它与标准的 `tf.estimator.EstimatorSpec` 会有一定的区别：

*  `eval_metric_ops` 必须被包装在 `metrics_fn` 中，这个字段会被重命名为 `eval_metrics`（[see above](#metrics)）。
*  `tf.train.SessionRunHook` 不受支持，因此省略这些字段。
*  如果使用 `tf.train.Scaffold`，必须被包装进一个函数中。这个字段会被重命名为 `scaffold_fn`。

`Scaffold` and `Hooks` 是高级用法，通常被忽略。

## 输入函数

因为输入函数是运行在主机上而不是 Cloud TPU 上的，所以它的运行方式没太大变化。本节主要解释了两项必要的调整。

### Params 参数

<!-- TODO(markdaoust) link to input_fn doc when it exists -->

标准 `Estimator` 的 `input_fn` **可以**包含一个 `params` 参数； `TPUEstimator` 的 `input_fn` **必须**包含一个 `params` 参数。这是允许估计器为输入流的每个副本设置批大小的必须参数。因此，`TPUEstimator` 的 `input_fn` 最简形式如下：

```
def my_input_fn(params):
  pass
```

`params['batch-size']` 包含了批次大小

### 静态形状和批次大小

由 `input_fn` 生成的输入管道在 CPU 上运行。因此，它并不需要遵循 XLA/TPU 环境下严格的静态形状要求。只有一个要求是，从输入管道输送到 TPU 的成批数据具有静态形状，由标准 TensorFlow 形状推断算法确定。中间张量可以随意，能具有动态形状。如果形状推断失败，但已知形状，则可以使用 `tf.set_shape()` 强制施加正确的形状。

在下面的示例中，形状推断算法失败，但使用了 `set_shape` 进行了更正：

```
>>> x = tf.zeros(tf.constant([1,2,3])+1)
>>> x.shape

TensorShape([Dimension(None), Dimension(None), Dimension(None)])

>>> x.set_shape([2,3,4])
```

在许多情况下，批次大小是唯一未知的维度。

使用 `tf.data` 的典型输入管道会产生固定大小的批次。不过，有限 `Dataset` 的最后一批次数据通常较小，只包含剩下的一些元素。由于 `Dataset` 不知道自己的长度或有限性，因此标准的 `f.data.Dataset.batch` 方法自己无法确定所有批次都有一个固定的批次大小：

```
>>> params = {'batch_size':32}
>>> ds = tf.data.Dataset.from_tensors([0, 1, 2])
>>> ds = ds.repeat().batch(params['batch-size'])
>>> ds

<BatchDataset shapes: (?, 3), types: tf.int32>
```
最直接的修复方法是按照以下方式使用 `f.data.Dataset.apply`，`.contrib.data.batch_and_drop_remainder`

```
>>> params = {'batch_size':32}
>>> ds = tf.data.Dataset.from_tensors([0, 1, 2])
>>> ds = ds.repeat().apply(
...     tf.contrib.data.batch_and_drop_remainder(params['batch-size']))
>>> ds

 <_RestructuredDataset shapes: (32, 3), types: tf.int32>
```

顾名思义，这种方法的一个缺点就是会在数据集的结尾丢弃任何的未满批次。于用于训练的无限重复数据集，这是可以接收的，但是你如果想要按一个具体的循环数训练，则会出现问题。

要进行一轮准确的运算，你可以通过手动填充批次的长度，并在创建 `tf.metrics` 时将条目设置为零权重来解决这一问题。

## 数据集

因为除非能够足够快地提供数据，否则不可能使用 Cloud TPU，所以在使用 Cloud TPU 时，如何高效使用 `tf.data.Dataset` API 是至关重要的。有关数据集性能的详细信息，请参见[输入管道性能指南(../performance/datasets_performance.md)。

对于最简单的实验（使用 `f.data.Dataset.from_tensor_slices` 或其他图中数据），需要将 `TPUEstimator` 中的 `Dataset` 读取的所有数据文件存储在  Google Cloud Storage Buckets 上。

<!--TODO(markdaoust): link to the `TFRecord` doc when it exists.-->

对于大多数使用情况，建议将数据转换为 `TFRecord` 格式，并使用 `tf.data.TFRecordDataset` 读取。但是，这不是硬性要求，你也可以根据喜好使用其他数据集读取器（`FixedLengthRecordDataset` 或 `TextLineDataset`）。

可以使用 `tf.data.Dataset.cache` 将小数据集完全加载到内存中。

不管使用何种数据格式，强烈建议[使用大型文件](../performance/performance_guide.md#use_large_files)，大小为 100MB。这在网络环境中尤为重要，因为打开文件的开销要大的多。

同样重要的是，无论使用哪种类型的读取器，都要使用构造函数的 `buffer_size` 参数启用缓冲。此参数以字节为单位指定。建议使用几 MB（`buffer_size=8*1024*1024`），以便在需要时提供数据。

TPU 示例仓库下包含一个用于下载 ImageNet 数据集并将其转换为适当格式的[脚本](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py)。

这与仓库中包含的 ImageNet [模型](https://github.com/tensorflow/tpu/tree/master/models)一起演示了所有的最佳实践。

## 下一步

* [Google Cloud TPU 文档](https://cloud.google.com/tpu/docs/) — 创建并运行一个 Google&nbsp;Cloud&nbsp;TPU。
* [TPU Demos Repository](https://github.com/tensorflow/tpu) — Cloud&nbsp;TPU 兼容 models 的示例。
* [Google Cloud TPU 性能指南](https://cloud.google.com/tpu/docs/performance-guide) — 通过调整应用程序的 Cloud TPU 配置参数，进一步增强 Cloud TPU 的性能。
