# 数据导入

`tf.data` API 能够快速的从简单可重用的片段中搭建输入总线。例如,一个图像处理模型的输入总线需要完成的任务可能是，从一个分布式文件系统中聚合数据，对每一张图像添加一些数据扰动，并能够将随机选中的图像合并进一个训练批次中。又如，一个文本处理模型的输入总线可能要从未加工的文本数据中提取出标识，并将它们按照一个查询表转化为嵌入标识符，再集合成不同长度的序列。`tf.data` API 能让处理大规模数据、不同数据格式兼容和复杂转换过程变得很简单。

`tf.data` API 为 TensorFlow 引入了两种新的抽象：

* `tf.data.Dataset` 是一个基本元素的序列，每一个基本元素中包含了一个或者多个 `Tensor` 对象。例如，在一个图像输入总线中，一个基本元素可能是一个单独的训练样本，它包含了一对 Tensor 对象，分别代表图像数据和一个标签。创建一个数据集有两种不同的方法：

  * 根据一个或者多个 `tf.Tensor` 对象创建一个 **source**（例如 `Dataset.from_tensor_slices()`）方法来构造一个数据集。
  
  * 根据一个或多个 `tf.data.Dataset` 对象应用一种 **transformation**（例如 `Dataset.batch()`）来构造一个数据集。

* `tf.data.Iterator` 提供了从一个数据集中提取基本元素的主要方法。由 `Iterator.get_next()` 返回的操作在执行时会给予 `DataSet` 的下一个基本元素，它通常作为输入总线代码和模型之间的接口。这个最简单的迭代器是 "one-shot iterator"，它关联到一个 `Dataset` 并遍历它一次。在更复杂的情况下，`Iterator.initializer` 操作能够将一个迭代器根据不同的数据集重新初始化并设定参数，所以举例来说，你可以在同样一个程序中多次迭代训练数据和验证数据。

## 基本结构

教程的这个部分描述了创建不同类型 `Dataset` 和 `Iterator` 对象的基本原理和怎样从它们之中提取数据。

要构建一个基本的输入总线，你必须先定义一个 **source**。例如，要从内存中的一些 tensor 对象中构造一个 `Dataset` ，你能够使用 `tf.data.Dataset.from_tensors()` 或者 `tf.data.Dataset.from_tensor_slices()`。或者，如果你的输入数据按照推荐的 TFRecord 格式存储在硬盘中的话，你可以构造一个 `tf.data.TFRecordDataset`。

一旦你有了一个 `Dataset` 对象，你能通过链接方法调用 `tf.data.Dataset` 对象将它**转换**为一个新的 `Dataset`。例如，你能够使用类似 `Dataset.map()`（用于对每一个元素使用一个函数）的方法转换每一个元素，或者使用 `Dataset.batch()` 处理多重元素的转换。要了解更多请查看关于 `tf.data.Dataset` 的文档中关于转换的完整内容。

要遍历来自一个 `Dataset` 中的内容一个最普通的方法是建立一个**迭代器**对象，它提供了每次获取数据集中一个元素的方法（比如，通过调用  `Dataset.make_one_shot_iterator()`）。`tf.data.Iterator` 提供了两种操作：`Iterator.initializer` 用作（重新）初始化迭代器的状态；`Iterator.get_next()` 返回符合下一元素标志的  `tf.Tensor` 对象。你可能会根据你的用例采用不同的迭代器，这些选项在下面会有概括。

### 数据集结构

一个数据集由具有相同结构的元素组成。一个元素包含一个或者多个 `tf.Tensor` 对象，它叫做**组件**。每一个组件都有表示 tensor 中元素类型的 `tf.DType` 和代表元素静态类型（可能是部分指定）的 `tf.TensorShape`。`Dataset.output_types` 和 `Dataset.output_shapes` 属性可以检查一个数据集元素中每个组件的推测类型。这些属性的**嵌套结构**映射到一个元素的结构中，可能是一个单一的 tensor，一组 tensor，或者一组嵌套结构的 tensor。例如：

```python
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"
```

给一个元素中的每一个组件起个名字是个很实用的方法，例如它们可以代表一个训练样本的不同特征。除了元组之外，你也能使用 `collections.namedtuple` 或者一个字符串映射到 tensor 的字典表示一个 `Dataset` 中的一个元素。

```python
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"
```

`Dataset` 转换支持任何结构的数据集，当使用 `Dataset.map()`, `Dataset.flat_map()` 和 `Dataset.filter()` 转换时，它们为每个元素请求一个函数，元素的结构决定这个函数的参数：

```python
dataset1 = dataset1.map(lambda x: ...)

dataset2 = dataset2.flat_map(lambda x, y: ...)

# 注意：Python 3 中不支持参数解构。
dataset3 = dataset3.filter(lambda x, (y, z): ...)
```

### 创建 iterator

一旦你创建了一个代表你输入数据的 `Dataset`，下一步就是创建一个 `Iterator` 来访问每个来自数据集中元素的。`tf.data` API 当前支持下面几种迭代器，它们的复杂度依次递增：

* **one-shot**,
* **initializable**,
* **reinitializable**, 和
* **feedable**.

**one-shot**迭代器是最简单的一种迭代器，它只支持在一个数据集中进行一次迭代，并且不需要显式初始化。它几乎处理了所有为基于队列的输入总线提供支持的情况，但是它不支持参数化。使用 `Dataset.range()` 的示例如下：

```python
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

注意：现在  one-shot 迭代器是唯一可以使用 `Estimator` 的迭代器类型。

**initializable** 迭代器在使用它之前需要使用一个 `iterator.initializer` 操作，虽然有些不便，但它允许你对数据集的定义进行**参数化**，在初始化迭代器时，通过使用一个或者多个 `tf.placeholder()` tensor 传入。下面依然是 `Dataset.range()` 的样例：

```python
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# 初始化一个处理十个元素的数据集的迭代器。
sess.run(iterator.initializer, feed_dict={max_value: 10})
for i in range(10):
  value = sess.run(next_element)
  assert i == value

# 初始化同样一个可以处理一百个元素的数据集的迭代器
sess.run(iterator.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = sess.run(next_element)
  assert i == value
```

**reinitializable** 迭代器能够从不同的 `Dataset` 对象中被初始化。例如，你可能会有一个使用训练输入总线，它对输入图像做随机扰动来提升泛化范围，还有一个用于验证的输入总线，它使用未经修改的数据对预测结果进行评估。这些输入总线通常会使用不同的 `Dataset` 对象，但它们拥有相同的结构（例如：每个组件都有相同的类型和兼容的形态）。

```python
# 将训练集和验证集设定成相同的结构
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)

# reinitializable 迭代器由它的结构定义。我们可以使用不论是在
# `training_dataset` 或者还是在 `validation_dataset` 中的
# `output_types` 和 `output_shapes`，因为它们是兼容的。
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# 在训练集中运行 20 轮，然后是训练集
for _ in range(20):
  # 初始化训练集的迭代器。
  sess.run(training_init_op)
  for _ in range(100):
    sess.run(next_element)

  # 初始化验证集的迭代器。
  sess.run(validation_init_op)
  for _ in range(50):
    sess.run(next_element)
```

**feedable**迭代器可以与 `tf.placeholder` 一起使用，用于在每次调用 `tf.Session.run` 时选取使用什么  `Iterator`，和 `feed_dict` 原理类似。它提供与了一个 **reinitializable** 迭代器一样的功能，但是它不需要你在切换迭代器时，再从一个数据集的起始处初始化。例如，使用与上述相同的训练和验证样例，你能使用 `tf.data.Iterator.from_string_handle` 来定义一个 **feedable** 迭代器，它允许你在两个数据集之间进行切换：

```python
# 将训练集和验证集设定成相同的结构
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# 一个 feedable 迭代器是由一个句柄占位符和它的结构定义的。
# 不论是在 training_dataset 还是在validation_dataset 中，
# 我们都可以使用 output_types 和 output_shapes 属性，
# 因为它们 (dataset) 有完全一致的结构。
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# 你能和 feeable 迭代器一起使用各种各样的迭代器
# （比如 one-shot 和 initializable 迭代器）。
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

# `Iterator.string_handle()` 方法返回一个可以被取值的 tensor，
# 它能被用于输入到“句柄”占位符。
training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

# 一直在训练集和验证集之间循环交替。
while True:
  # 使用训练集运行 200 轮。注意训练集是无限的，我们从前一个
  # `while` 循环迭代暂停的地方继续运行。
  for _ in range(200):
    sess.run(next_element, feed_dict={handle: training_handle})

  # 在验证数据集上运行一遍
  sess.run(validation_iterator.initializer)
  for _ in range(50):
    sess.run(next_element, feed_dict={handle: validation_handle})
```

### 如何使用迭代器

`Iterator.get_next()` 方法返回一个或多个 `tf.Tensor` 对象，它们对应一个迭代器中下一个元素的标志。每次这些 tensor 被评估时，它们会从底层的数据集中取出下一个元素的值。（注意：类似 TensorFlow 中其他有状态对象，调用 `Iterator.get_next()` 不能立即使迭代器增加向前运行。相反的，你需要使用一个 TensorFlow 表达式中返回的 `tf.Tensor` 对象，并将这个表达式的结果传递到 `tf.Session.run()` 中，得到下一个元素，使迭代器进入下一步。）

如果迭代器到达了数据的结尾，执行 `Iterator.get_next()` 操作会抛出一个 `tf.errors.OutOfRangeError`。在这之后，迭代器会是一个不可用的状态，如果还想使用它你需要再次将它初始化。

```python
dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# 通常 `result` 会输出一个模型，或者一个优化器的训练操作
result = tf.add(next_element, next_element)

sess.run(iterator.initializer)
print(sess.run(result))  # ==> "0"
print(sess.run(result))  # ==> "2"
print(sess.run(result))  # ==> "4"
print(sess.run(result))  # ==> "6"
print(sess.run(result))  # ==> "8"
try:
  sess.run(result)
except tf.errors.OutOfRangeError:
  print("End of dataset")  # ==> "数据集的结尾"
```

一个常用的模式是将"训练循环" 封装到一个  `try`-`except` 模块中：

```python
sess.run(iterator.initializer)
while True:
  try:
    sess.run(result)
  except tf.errors.OutOfRangeError:
    break
```

如果数据集中的每一个元素都有一个嵌套的结构，`Iterator.get_next()`的返回值会是一个或者多个具有相同嵌套结构的 `tf.Tensor` 对象

```python
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()

sess.run(iterator.initializer)
next1, (next2, next3) = iterator.get_next()
```

注意：`next1`、`next2` 和 `next3` 是由相同的 op/节点（由 `Iterator.get_next()` 创建）所产生的张量。因此，评估这些张量中的**任何**一个，都会提高所有组件的迭代器。一个典型的迭代器消费者将会在单独的表达式中包含所有的组件。

### 保存迭代器状态

`tf.contrib.data.make_saveable_from_iterator` 函数从一个迭代器中创建一个 `SaveableObject`，它可以用于保存和恢复迭代器（实际上，甚至可以是整个输入管道）。这样创建的可保存对象，可以添加到 `tf.train.Saver` 变量列表或者用于保存和恢复的 `tf.GraphKeys.SAVEABLE_OBJECTS` 集合中，方式与  `tf.Variable` 相同。关于如何保存和恢复变量的更多信息，请参阅[保存和恢复](../guide/saved_model.md)。

```python
# Create saveable object from iterator.
saveable = tf.contrib.data.make_saveable_from_iterator(iterator)

# Save the iterator state by adding it to the saveable objects collection.
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, saveable)
saver = tf.train.Saver()

with tf.Session() as sess:

  if should_checkpoint:
    saver.save(path_to_checkpoint)

# Restore the iterator state.
with tf.Session() as sess:
  saver.restore(sess, path_to_checkpoint)
```

## 读取输入数据

### 使用 NumPy 数组

如果你所有的数据都加载到内存中了，从中创建一个 `Dataset` 的最简单方法就是将它们转换为 `tf.Tensor` 对象再使用 `Dataset.from_tensor_slices()`。

```python
# 将训练数据加载进两个 NumPy 数组，例如使用  `np.load()`。
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# 假设 `features` 的每一行都对应与 `labels` 相同的行。
assert features.shape[0] == labels.shape[0]

dataset = tf.data.Dataset.from_tensor_slices((features, labels))
```

注意，以上代码片段会将 `features` 和 `labels` 数组当做 `tf.constant()` 操作嵌套进你的 Tensor 图中。这在小数据集中能够良好运行，但会浪费内存，因为数组中的内容会被多次复制，会达到 `tf.GraphDef` 协议缓冲区的 2 GB 限制。

另外，你可以按照 `tf.placeholder()` tensor 定义 `Dataset` ，并在使用这个数据集初始化一个`Iterator` 时**传入**这些 NumPy 数组。

```python
# 将训练数据加载进两个 NumPy 数组，例如使用  `np.load()`。
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]

# 假设 `features` 的每一行都对应与 `labels` 相同的行。
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [其他 `dataset` 中的转换]
dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
```

### 使用 TFRecord 数据

`tf.data` API 支持很多种文件格式，所以你能够处理无法放入内存的大数据集。例如，TFRecord 文件格式是一种简单的面向记录的二进制格式，许多 TensorFlow 应用都用它来存储训练数据。`tf.data.TFRecordDataset` 类允许你将一个或者多个 TFRecord 格式文件串流化，并将它们当做一个输入总线的一部分。

```python
# 创建一个从两个文件中读取所有例子的数据集
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
```

`TFRecordDataset` 初始化时的 `filenames` 参数可以是一个字符串，一个字符串列表，或者一个 `tf.Tensor` 的字符串。因此，如果你有两个文件集合，分别用作训练和验证，你可以使用一个 `tf.placeholder(tf.string)` 来代表文件名，并使用一个合适的文件名来初始化一个迭代器：

```python
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # 将记录解析为 tensor。
dataset = dataset.repeat()  # 无限重复输入。
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()

# 你可以向迭代器中传入适用于当前操作的文
# 件名，比如 training 或者 validation

# 初始化训练数据的 `iterator`。
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

# 初始化验证数据的  `iterator`。
validation_filenames = ["/var/data/validation1.tfrecord", ...]
sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
```

### 使用文本数据

许多数据集分布在一个或者多个文本文件中。`tf.data.TextLineDataset` 提供了一种从一个或多个文本文件抽取行的简易方法。传递一个或多个文件名，`TextLineDataset` 会为这些文件的每一行生成一个字符串格式的元素。例如 `TFRecordDataset` ， `TextLineDataset` 接收一个 `filenames` 当做 `tf.Tensor`，所以你能通过传递一个 `tf.placeholder(tf.string)` 对其参数化。

```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.data.TextLineDataset(filenames)
```

默认的 `TextLineDataset` 产出每个文件的**每一**行，这也许并不是我们期望的，例如如果这个文件开始于一个标题行或者包含了注释。这些行可以通过 `Dataset.skip()` 和 `Dataset.filter()` 转换来移除。为了单独的对每一个文件进行这样的转换，我们使用 `Dataset.flat_map()` 来为每一个文件创建一个嵌套的 `Dataset`。

```python
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

dataset = tf.data.Dataset.from_tensor_slices(filenames)

# 使用 `Dataset.flat_map()` 来将每个文件转换成一个单独的嵌套数据集，
# 然后连接他们的内容然后放入一个“扁平的”数据集。
# * 跳过第一行（标题行）。
# * 过滤以 "#" 开始的行（注释）。
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))
```

### 使用 CSV 数据

CSV 是一种以纯文本形式存储的常用表格数据格式。`tf.contrib.data.CsvDataset` 类提供了一种从一个或多个符合 [RFC 4180](https://tools.ietf.org/html/rfc4180) 规范的 CSV 文件中提取记录的方法。给定一个或多个文件名和一个默认列表，`CsvDataset` 将生成一个元组，其元素对应于每个 CSV 记录提供的默认类型。与 `TFRecordDataset` 和 `TextLineDataset` 一样，`CsvDataset` 接受 `filenames` 参数，然后以 `tf.Tensor` 的形式接受，所以你可以通过传递 `tf.placeholder(tf.string)` 的方式初始化参数。

```
# 创建一个数据集，该数据集读取两个 CSV 文件中的所有记录，每个文件包含八个浮点列
filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]
record_defaults = [tf.float32] * 8   # Eight required float columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
```

如果某些列为空，则可以提供默认值。

```
# 创建一个数据集，该数据集读取两个 CSV 文件中的所有记录，每个文件包含四个可能缺少值的浮点列
record_defaults = [[0.0]] * 8
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults)
```

默认情况下，`CsvDataset` 会读取文件的<b>每</b>行的<b>每</b>列，这是不可取的，例如，如果文件以应忽略的标题行开头，或者如果不需要某些列。这些行和字段可以分别用 `header` 和 `select_cols` 参数删除。

```
# 创建一个数据集，该数据集读取带有标题的两个 CSV 文件中的所有记录，从第 2 列和第 4 列中提取浮点数据。
record_defaults = [[0.0]] * 2  # Only provide defaults for the selected columns
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[2,4])
```

<!--
TODO(mrry): Add these sections.

### 处理来自 Python 生成器的数据
-->

## 使用 `Dataset.map()` 预处理数据

`Dataset.map(f)` 通过对输入数据集中每一个元素使用一个给定的 `f` 函数生成一个新的数据集。它基于 [`map()` 函数](https://en.wikipedia.org/wiki/Map_(higher-order_function))，这种函数通常在函数式编程语言中用于处理列表（和其他的一些数据结构）。函数 `f` 取用代表输入数据集中一个元素的 `tf.Tensor` 对象，并返回代表新数据集中一个元素的 `tf.Tensor` 对象。它的实现使用标准的 TensorFlow 操作来将一个元素转换成另一个。

这一节包含如何使用 `Dataset.map()` 的一般样例。

### 解析 tf.Example 协议缓冲区的消息

许多输入总线会从一个 TFRecord 格式文件中抽取 `tf.train.Example` 协议缓冲信息（例如，使用 `tf.python_io.TFRecordWriter`）。每一个 `tf.train.Example` 记录包含一个或者多个“特性”，并且输入总线通常会将这些特性转换为 tensors。

```python
# 将一个字符串标量 `example_proto` 转换到一个字符串标量和
# 一个整数标量，分别代表一个图像和它的标签。
def _parse_function(example_proto):
  features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
              "label": tf.FixedLenFeature((), tf.int64, default_value=0)}
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["image"], parsed_features["label"]

# 创建一个读取从两个文件中所有样例的数据集，并提取图像和特征标签。
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
```

### 解析图像数据并重设大小

在使用一个真实世界的图像数据来训练一个神经网络时，通常需要将不同尺寸的图像转换成同一尺寸，所以它们需要批量转换成一个固定的尺寸。

```python
# 从一个文件中读取图像，将它解码到一个稠密的 tensor 中，并且
# 将它重设到一个固定的尺寸。
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# 一个包含所有文件名的向量
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])

# `labels[i]` 是  `filenames[i]` 图像的标签。
labels = tf.constant([0, 37, ...])

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)
```

### 使用 `tf.py_func()` 调用 Python 库

出于性能的原因，我们推荐你尽可能的使用 TensorFlow 操作预处理数据。然而，在解析输入数据时，有时调用一些原生的 Python 库会很有效。我们可以通过在一个 `Dataset.map()` 转换中调用 `tf.py_func()` 操作来实现这一点。

```python
import cv2

# 使用一个自定义的 OpenCV 函数读取图像，来替代标准的
# TensorFlow 的 `tf.read_file()` 操作。
def _read_py_function(filename, label):
  image_decoded = cv2.imread(filename.decode(), cv2.IMREAD_GRAYSCALE)
  return image_decoded, label

# 使用一个标准的 TensorFlow 操作来重设图像到一个固定的大小。
def _resize_function(image_decoded, label):
  image_decoded.set_shape([None, None, None])
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
labels = [0, 37, 29, 1, ...]

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(
    lambda filename, label: tuple(tf.py_func(
        _read_py_function, [filename, label], [tf.uint8, label.dtype])))
dataset = dataset.map(_resize_function)
```

<!--
TODO(mrry): Add this section.

### 处理超常大小的文本数据
-->

## 打包元素成数据集

### 简单打包

打包的最简单方式是将一个数据集中 `n` 个连续元素堆叠进一个单一元素中。`Dataset.batch()` 函数就是用来做这样一个变换的 —— 它和 `tf.stack()` 操作有相同的约束条件，同时它作用于元素的每个组件上：例如，对于每个组件 i，所有的元素必须具有一个一致形态的 tensor。

```python
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

iterator = batched_dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])
```

### 使用填充打包 tensor

上面的方法需要所有的 tensor 都是相同的标准。但是，许多模型（比如，序列模型）的输入数据的标准变化多样（例如：序列有不同的长度）。为了处理这种情况，`Dataset.padded_batch()` 转换通过指定一个或多个它们可能的标准来将不同的形态的 tensor 填充。

```python
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                               #      [5, 5, 5, 5, 5, 0, 0],
                               #      [6, 6, 6, 6, 6, 6, 0],
                               #      [7, 7, 7, 7, 7, 7, 7]]
```

`Dataset.padded_batch()` 转换允许你为每一个组件的每一种尺寸设置不同的填充，并且它可以是可变长度的（在上面的样例中就被指定为 `None` ）或者恒定长度。它也可以重设默认值为 0 的填充值。

<!--
TODO(mrry): Add this section.

### 密集不规则 -> tf.SparseTensor
-->

## 训练工作流

### 多次循环处理

`tf.data` API 提供了两种主要方法来处理同一数据多次循环。

最简单的多次循环迭代一个数据集的方法是使用 `Dataset.repeat()` 转换。例如，创建一个重复输入数据10次的数据集：

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)
```

不设置参数使用 `Dataset.repeat()` 转换的话会导致无限次重复的输入。`Dataset.repeat()` 转换将它的参数连接，无需在一轮结束处与下一轮开始处发出信号。

如果你想要在每一轮的结尾处接收到一个信号，你可以编写一个训练循环，用于在数据集结尾处捕获 `tf.errors.OutOfRangeError`。在那个地方，你可以收集到该轮的一些统计信息（例如：验证错误）

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# 进行 100 轮的计算
for _ in range(100):
  sess.run(iterator.initializer)
  while True:
    try:
      sess.run(next_element)
    except tf.errors.OutOfRangeError:
      break

  # [这里展现一轮结束的操作]
```

### 随机混排输入数据

`Dataset.shuffle()` 转换使用一个与 `tf.RandomShuffleQueue` 相似的算法随机混排输入数据集：它会维护一个固定大小的缓冲区并从该缓冲区中随机挑选下一个元素。

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat()
```

### 使用高级 API

`tf.train.MonitoredTrainingSession` API 简化了分布式运行 TensorFlow 的许多设置。`MonitoredTrainingSession` 使用 `tf.errors.OutOfRangeError` 来标识训练已经完成，所以我们推荐使用 `tf.data` API。例如：

```python
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.repeat(num_epochs)
iterator = dataset.make_one_shot_iterator()

next_example, next_label = iterator.get_next()
loss = model_function(next_example, next_label)

training_op = tf.train.AdagradOptimizer(...).minimize(loss)

with tf.train.MonitoredTrainingSession(...) as sess:
  while not sess.should_stop():
    sess.run(training_op)
```

要使用 `tf.estimator.Estimator` 中 `input_fn` 的 `Dataset`，只需返回 `Dataset`，框架将负责创建迭代器并为帮你对其进行初始化。例如：

```python
def dataset_input_fn():
  filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
  dataset = tf.data.TFRecordDataset(filenames)

  # 使用 `tf.parse_single_example()` 来从一个 `tf.Example` 协议缓冲区
  # 提取数据，并额外完成每个记录的预处理。
   def parser(record):
    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
        "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # 完成在解析数据中的额外预处理
    image = tf.image.decode_jpeg(parsed["image_data"])
    image = tf.reshape(image, [299, 299, 1])
    label = tf.cast(parsed["label"], tf.int32)

    return {"image_data": image, "date_time": parsed["date_time"]}, label

  # 使用 `Dataset.map()` 为每个样例建立一个特征字典和
  # 一个标签 tensor 的数据组。
  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(num_epochs)

  # `dataset` 的每个元素都是包含特征字典的元组（其中每个值是该特征的一批值）和一批标签。
  return dataset
```
