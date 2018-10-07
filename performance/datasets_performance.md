# 输入管道性能指南

GPU 和 TPU 可以从根本上缩短执行单个训练步骤所需的时间。为了实现最佳性能的目的，我们需要一个高效的输入管道用于在当前步骤完成前为下一步骤提供数据。`tf.data` API 有利于构建灵活高效的输入管道。本文档介绍了 `tf.data` 的特性以及在各种模型和加速器中构建高性能 TensorFlow 输入管道的最佳实践。

本指南主要有以下内容：

*   说明 TensorFlow 输入管道本质上是一个 ETL 进程。
*   介绍 `tf.data` API 在常见情景中的性能优化。
*   讨论您应用转换的顺序对训练性能所产生的影响。
*   总结设计高性能 TensorFlow 输入管道的最佳实践。

## 输入管道结构

一个典型的 TensorFlow 输入管道训练过程可以被设计为一个 ETL 进程：

1.  **抽取：** 从存储器中读取数据 —— 本地（例如 HDD 或 SSD）或云端（例如 GCS 或 HDFS）。
2.  **转化：** 使用 CPU 内核处理器解析及执行数据预处理操作，如图像解压缩，数据增强型转换（如随机裁剪，翻转和颜色失真），乱序化和批处理。
3.  **加载：** 将转换的数据加载到执行机器学习模型的加速器（例如 GPU 或 TPU）上。

这种模式在保证加速器来训练你的模型的同时有效利用 CPU。另外，将输入管道视为一个 ETL 流程提供了一种便于在性能优化中应用的结构。

当使用 `tf.estimator.Estimator` API 时，前两个阶段（抽取和转换）在传递到 `tf.estimator.Estimator.train` 时在 `input_fn` 中被捕获。
在代码中，这可能是如下自然顺序的实现：

```
def parse_fn(example):
  "Parse TFExample records and perform simple data augmentation."
  example_fmt = {
    "image": tf.FixedLengthFeature((), tf.string, ""),
    "label": tf.FixedLengthFeature((), tf.int64, -1)
  }
  parsed = tf.parse_single_example(example, example_fmt)
  image = tf.image.decode_image(parsed["image"])
  image = _augment_helper(image)  # augments image using slice, reshape, resize_bilinear
  return image, parsed["label"]

def input_fn():
  files = tf.data.Dataset.list_files("/path/to/dataset/train-*.tfrecord")
  dataset = files.interleave(tf.data.TFRecordDataset)
  dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
  dataset = dataset.map(map_func=parse_fn)
  dataset = dataset.batch(batch_size=FLAGS.batch_size)
  return dataset
```

下一部分构建是在输入管道上增加性能优化。

## 性能优化

随着新的计算设备（如 GPU 和 TPU）能够以越来越快的速度训练神经网络，CPU 训练开始容易成为训练时的瓶颈。`tf.data` API 为用户提供构建块以构建有效利用 CPU 的输入管道，优化 ETL 进程的每个步骤。

### 流水线机制

要执行训练步骤，您必须首先抽取并转换训练数据，然后将其输入到加速器运行的模型上。但是，在一般的同步实现过程中，当 CPU 正在准备数据时，加速器处于空闲状态。类似相反，在加速器正在训练模型时，CPU 处于闲置状态。所以训练步骤时间是 CPU 预处理时间和加速器训练时间的总和。

**流水线技术** 重叠训练的预处理和模型训练步骤。当加速器正在执行训练步骤 N 时，CPU 开始准备步骤 N + 1 的数据。这样做可以将步骤时间减少到模型训练与抽取转换数据二者所需的最大时间（而不是二者时间总和）。

没有流水线技术，CPU 和 GPU/TPU 大部分时间将处于闲置状态：

![without pipelining](https://www.tensorflow.org/images/datasets_without_pipelining.png)

通过流水线技术，空闲时间显著减少：

![with pipelining](https://www.tensorflow.org/images/datasets_with_pipelining.png)

`tf.data` API 通过 `tf.data.Dataset.prefetch` 转换提供了一种软件流水线机制，该转换可以用来分离从消耗时间生成的时间数据。特别地，转换使用后台线程和内部缓冲区来提前抽取输入数据集中的元素。因此，要实现上面说明的流水线效果，可以添加 `prefetch(1)` 对数据集管道的最终转换（或者如果单个训练步骤消耗 n 个元素，则预取 `prefetch(n)`）。

要将此应用于我们的运行样例，则需要更改：

```
dataset = dataset.batch(batch_size=FLAGS.batch_size)
return dataset
```

为：

```
dataset = dataset.batch(batch_size=FLAGS.batch_size)
dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)
return dataset
```

请注意，只要能够有机会将“训练”与“开销”重叠在一起，抽取转换就会带来好处。前面的建议是最常见的应用程序。

### 并行数据转换

当准备批处理时，可能需要预处理输入元素。为此，`tf.data ` API 提供了 `tf.data.Dataset.map` 转换，该转换将用户定义的函数（例如，来自运行示例的 `parse_fn`）应用于输入数据集的每个元素。由于输入元素彼此独立，因此可以在多个 CPU 内核之间对预处理进行并行处理。为了实现这一点，`map` 转换提供了`num_parallel_calls` 参数来指定并行性的级别。例如，下图说明了将 `num_parallel_calls=2`设置到 `map` 转换的效果：

![parallel map](https://www.tensorflow.org/images/datasets_parallel_map.png)

为 `num_parallel_calls` 参数选择最佳值取决于您的硬件情况，训练数据的特征（如大小和形状）及映射函数的消耗以及 CPU 上同时进行的其他处理进程；
一个简单的启发式就是所使用的可用 CPU 核心数量。例如，如果执行上述样例的机器具有 4 个内核，则设置 `num_parallel_calls=4` 会更高效。但另一方面，将 `num_parallel_calls` 设置为远大于可用 CPU 数量的值可能会导致调度效率低下，从而导致速度变慢。

要将此应用于我们的运行样例，则需要更改：

```
dataset = dataset.map(map_func=parse_fn)
```

为:

```
dataset = dataset.map(map_func=parse_fn, num_parallel_calls=FLAGS.num_parallel_calls)
```

此外，如果您的批大小为数百或数千，那么您的管道可能会从批创建的并行中受益。为此，`tf.data` API 提供了 `tf.contrib.data.map_and_batch` 转换，它有效地“融合”了映射和批量转换。

要将此应用于我们的运行样例，则需要更改：

```
dataset = dataset.map(map_func=parse_fn, num_parallel_calls=FLAGS.num_parallel_calls)
dataset = dataset.batch(batch_size=FLAGS.batch_size)
```

为:

```
dataset = dataset.apply(tf.contrib.data.map_and_batch(
    map_func=parse_fn, batch_size=FLAGS.batch_size))
```

### 并行数据抽取

在现实环境中，输入数据可能在云端存储（例如 GCS 或 HDFS），因为输入数据不适合存储在本地，或者因为训练是分布式的，在每台机器上复制输入数据是没有意义的。由于本地和远程存储之间存在以下差异，在本地抽取数据时运行良好的数据集管道可能在云端读取数据时反而成为 I/O 瓶颈：

*   **到首字节时间：** 从云端存储中抽取文件的第一个字节的时间可能比本地存储的时间要长。
*   **吞吐量：** 尽管云端存储通常提供较大的总带宽，但读取单个文件可能只能利用这一带宽的一小部分。

另外，一旦将原始字节读入内存中，也可能需要对数据进行反序列化或解密（例如 protobuf），这会增加额外的开销。尽管无论数据是本地存储还是远程存储，都会出现此开销，但如果数据未被有效预取，在云端情况下会变得更糟。

为了减轻各种数据抽取开销的影响，`tf.data`API 提供了 `tf.contrib.data.parallel_interleave` 转换。使用此转换来并行执行及交错其他数据集的内容（如数据文件读取器）。要重叠的数据集的数量可以由 `cycle_length` 参数指定。下图说明了为 `parallel_interleave` 转换提供 `cycle_length=2` 的效果：

![parallel io](https://www.tensorflow.org/images/datasets_parallel_io.png)

要将此应用于我们的运行样例，则需要更改：

```
dataset = files.interleave(tf.data.TFRecordDataset)
```

为：

```
dataset = files.apply(tf.contrib.data.parallel_interleave(
    tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
```

由于负载或网络事件的缘故，云端存储系统的吞吐量可能随时间而发生变化。为了解决这个差异，`parallel_interleave` 转换可以选择使用预读取。（有关详细信息，请参阅 `tf.contrib.data.parallel_interleave`）。

默认情况下，`parallel_interleave` 转换提供确定性元素顺序以实现重现性。作为预读取的替代方法（在某些情况下可能无效），`parallel_interleave` 转换还提供了一个选项以提高性能，但会牺牲顺序的确定性。别是，如果 `sloppy` 参数设置为 true，那么转换可能会偏离其确定性顺序，这是通过临时跳过请求下一个元素时元素不可用的文件导致的。

## 性能考虑

 `tf.data` API 是围绕可组合转换设计的，为用户提供了灵活性。虽然这些转换大多数是可交换的，但某些转换的排序会对性能产生影响。

### 映射和批处理

调用传递给 `map` 转换的用户定义函数具有与调度和执行用户定义函数有关的开销。通常情况下，这个开销比函数执行的计算量要小。然而在函数环境中，如果 `map` 工作量很小，这种开销可能会占用总成本。在这种情况下，我们建议矢量化用户定义的函数（即让它一次对一批输入进行操作），并在 `map` 转换之前应用 `batch` 转换。

### 映射和缓存

`tf.data.Dataset.cache` 转换可以将数据集缓存在内存或本地存储中。 如果传递给 `map` 转换的用户定义函数的开销很大，只要结果数据集仍然适合内存或本地存储，就可以在 `map` 转换之后应用高速缓存变换。 如果用户定义的函数增加了将数据集存储到缓存容量以外所需的空间，请考虑在您训练处理之前预处理您的数据以减少资源使用量。

### 映射和交错/预取/乱序

许多转换，包括 `interleave`， `prefetch`，和 `shuffle` 都维护着元素的内部缓冲区。如果传递给 `map` 转换的用户定义函数改变了元素的大小，那么 `map` 转换的顺序和缓冲元素的转换会影响内存使用。通常，我们建议选择导致内存占用较小的顺序，除非性能要求原因需要不同的排序（例如，启用 `map` 和批量转换的融合）。

### 重复和乱序

`tf.data.Dataset.repeat` 转换将输入数据重复有限（或无限）次数；数据的每次重复通常被称为**迭代次数**。`tf.data.Dataset.shuffle` 转换将乱序随机化数据集样例的顺序。

如果 `repeat` 转换在 `shuffle` 转换之前应用，则迭代次数边界将变的不确定。也就是说，某些元素可以在其他元素出现之前重复一次。另一方面，如果在 `repeat` 转换之前应用 `shuffle` 转换，则在涉及 `shuffle` 转换的内部状态初始化的每个迭代次数开始时性能可能会下降。换句话说，前者（在 `shuffle` 之前 `repeat`）提供了更好的性能，而后者（在 `repeat` 之前 `shuffle`）提供了更确定性的排序。

如果可能，我们建议使用融合的 `tf.contrib.data.shuffle_and_repeat` 转换，该转换具有两全其美的特性（良好的性能和明确的排序保证）。否则，我们建议在重复之前乱序随机化。

## 最佳实践摘要

以下是设计输入管道的最佳实践总结：

*   使用 `prefetch` 转换来合并训练和开销的工作。 特别是，我们建议在输入管道的末端添加 prefetch(n)（其中 n 是训练步骤消耗的元素/批次数），以将 CPU 上执行的转换与加速器上的训练合并。
*   通过设置 `num_parallel_calls` 参数来并行化 `map` 转换。我们建议使用可用的 CPU 内核数量作为其值。 
*   如果要使用 `batch` 转换将预处理元素组合到批处理中，我们建议使用融合的 `map_and_batch` 转换；特别是在你使用大批量数据的情况下。
*   如果您正在处理云端存储的数据和/或需要反序列化的数据，我们建议使用 `parallel_interleave` 转换来重叠读取（和反序列化）来自不同文件的数据。
*   向传递给 `map` 转换的轻量用户定义函数进行矢量化，以分摊与调度和执行函数相关的开销。
*   如果你的数据可以放入内存，在第一个迭代次数期间使用 `cache` 转换将其缓存在内存中，这样后续的迭代次数可以避免产生与读取，解析和转换相关的开销。
*   如果预处理增加了数据的大小，我们建议首先应用 `interleave`，`prefetch`，和 `shuffle` 如果可以的话）以减少内存占用量。
*   我们建议在 `repeat` 转换之前应用 `shuffle` 转换，理想情况下使用融合的 `shuffle_and_repeat` 转换。
