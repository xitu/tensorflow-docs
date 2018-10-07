# 底层 API 编程介绍

这篇指南将帮助你使用底层的 TensorFlow 核心 API 进行编程，会涉及到以下几个部分：

  * 管理你的 TensorFlow 程序（一张 TensorFlow 计算图，`tf.Graph`）以及 TensorFlow 运行时（一个 TensorFlow 会话，`tf.Session`），而不是依赖 Estimators 来管理它们
  * 使用 `tf.Session` 运行 TensorFlow 操作。
  * 在底层环境中使用高级的组件（[datasets](#datasets)，[layers](#layers)，以及 [特征列](#特征列)）。
  * 创建你自己的训练循环，而不是使用 [Estimators 提供的](../guide/premade_estimators.md)。

我们推荐尽量使用更高层次的 API 来构建模型，但了解 TensorFlow 核心 API 有以下几个优点：

  * 使用低级 TensorFlow 的操作能帮你更加切中肯綮地进行实验和 debug 
  * 在使用高层次 API 的时候，你能够知道其内部是如何运作的

## 配置

在使用这篇指南之前，请先安装 TensorFlow [安装 TensorFlow](../install)。

理解本指南的大部分内容需要你有以下知识储备：

*   会使用 Python 进行编程
*   关于数组的一些知识
*   最好了解一些机器学习的内容

启动 `python` 然后跟着一起来实操吧!

运行下列代码来配置你的 Python 环境：

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
```

## 张量值

TensorFlow 核心的数据单元是**张量**（Tensor），一个张量由一个基本数据类型的多维数组来表示。张量的**秩**指的是它的维度数量，它的**形状**是一个整数元组，指定了各个维度的数组长度。下面一些张量值的例子：

```python
3. # 秩为 0 的张量，形为 [] 的标量
[1., 2., 3.] # 秩为 1 的张量，形为 [3] 的向量
[[1., 2., 3.], [4., 5., 6.]] # 秩为 2 的张量，形为 [2, 3] 的矩阵
[[[1., 2., 3.]], [[7., 8., 9.]]] # 秩为 3 的张量，形为 [2, 1, 3]
```

TensorFlow 使用 numpy 数组来表示张量的**值**。

## TensorFlow 核心 API 实战

使用 TensorFlow 核心 API 编程由以下两部分组成：

1.  构建计算图（一张 `tf.Graph`）。
2.  运行计算图（使用 `tf.Session`）。

### 计算图

**计算图**是一组定义在图中的 TensorFlow 操作，该图由两种类型的对象组成：

  * `tf.Operation`（或称为 "ops"）：操作图的节点，操作描述了如何创建和使用张量。
  * `tf.Tensor`：张量是图的边，代表着计算图中流动的值。大部分的 TensorFlow 函数都会返回一个 `tf.Tensors`。

重点：`tf.Tensors` 并不包含值，而只是操作图中各个部分的工具。

让我们来构建一张简单的计算图。图中最基础的操作就是常量操作，`tf.constant() `这个 Python 函数以一个张量值作为输入，其产生结果不需要额外的输入。当我们调用它时，它会输出传递给构造器的那个值。

我们可以创建两个浮点类型的常量 `a` 和 `b` 如下列代码所示：

```python
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # 隐式的创建一个浮点类型
total = a + b
print(a)
print(b)
print(total)
```

`print` 语句的结果如下：

```
Tensor("Const:0", shape=(), dtype=float32)
Tensor("Const_1:0", shape=(), dtype=float32)
Tensor("add:0", shape=(), dtype=float32)
```

注意，张量的打印结果并不是如你所想的 `3.0`，`4.0` 和 `7.0`。上述语句只是构建了计算图，这些 `tf.Tesor` 对象只代表了将要进行的操作的结果。

计算图中的每个操作都会被赋予一个唯一的名称，这个名称和 Python 对象的名称无关。张量的命名规则是`"操作名 + 输出序号"`，如上面的 `"add:0"`。

### TensorBoard

TensorFlow 提供了一个名为 TensorBoard 的工具。TensorBoard 的功能之一就是可视化一张计算图，几个简单的命令就能实现它。

首先，需要将计算图存为 TensorBoard 的总结文件，代码如下所示：

```
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
```

你会在当前目录下得到一个 `.event` 后缀的文件，其命名格式如下：

```
events.out.tfevents.{timestamp}.{hostname}
```

接下来，在一个新的终端里，使用下面的命令启动 TensorBoard：

```bsh
tensorboard --logdir .
```

你可以在你的浏览器里打开 TensorBoard 的[计算图页面](http://localhost:6006/#graphs)，你应该会看到和下面类似的一张图

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_add.png)

想要了解更多 TensorBoard 可视化的内容，请查看 [TensorBoard：图表可视化](../guide/graph_viz.md)。

### 会话

对张量求值，你需要实例化一个 `tf.Session` 对象，也叫做**会话**。一个会话封装了 TensorFlow 运行时的状态，并且在会话中运行 TensorFlow 操作。打一个比方，如果把 `tf.Graph` 看做是一个 `.py` 文件，那么一个 `tf.Session` 就像是一个 `python` 可执行文件。

下面的代码创建了一个 `tf.Session` 对象，并且调用了它的 `run` 方法来对我们先前创建的 `total` 张量求值：

```python
sess = tf.Session()
print(sess.run(total))
```

当你使用 `Session.run` 方法来获取一个节点的输出时，TensorFlow 回溯计算图，并先计算出所获取节点需要的输入（如果是某个节点的输出，则计算之），然后输出结果。所以打印的结果是 7.0：

```
7.0
```

你可以同时传递给 `tf.Session.run` 多个张量，`run` 方法会自动处理任意元组或者是字典的组合，如下面的例子所示：

```python
print(sess.run({'ab':(a, b), 'total':total}))
```

方法返回的结果和输入具有相同的结构：

``` None
{'total': 7.0, 'ab': (3.0, 4.0)}
```

在调用 `tf.Seeion.run` 的过程中，每一个 `tf.Tensor` 张量对象具有唯一的值。比如，下面的代码调用了 `tf.random_uniform` 函数来产生一个服从均一分布的 3 个元素的随机向量（值介于 `[0, 1]`）：

```python
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))
```

结果显示了每次调用 `run`，都会产生不同的结果，但是在一次调用中，`vec` 的值保持不变（`out1` 和 `out2` 接收同一个随机的输入）：

```
[ 0.52917576  0.64076328  0.68353939]
[ 0.66192627  0.89126778  0.06254101]
(
  array([ 1.88408756,  1.87149239,  1.84057522], dtype=float32),
  array([ 2.88408756,  2.87149239,  2.84057522], dtype=float32)
)
```

一些 TensorFlow 函数会返回 `tf.Operations` 对象而不是 `tf.Tensors`，在一个 Operation（操作） 上调用 `run` 的结果是 `None`。运行一个操作的结果不会获取值，但是会产生一系列的副作用，像[初始化](#Initializing Layers)以及[训练](#Training)操作，这些内容会在后面的部分谈到。

### 赋值

如果计算图始终只能产生常量结果的话，那就没什么意思了。计算图能够通过使用**占位符**（placeholder）来接收外部的输入，**占位符**像是一个函数的参数，能够在之后的操作中进行赋值。

```python
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
```

上面的三行代码有点像一个函数，我们首先定义了两个输入的参数（`x` 和`y`），然后对他们进行了操作。我们可以对这个计算图进行求值操作，通过使用 `feed_dict` 这个 `tf.Session.run` 的参数来为占位符赋值：

```python
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
```

结果的输出如下：

```
7.5
[ 3.  7.]
```

需要注意，`feed_dict` 参数可以用来覆盖计算图中的任何张量。占位符和 `tf.Tensors` 的唯一区别就在于如果占位符没有被赋值，则使用时会抛出错误。

## 数据集

简单的实验我们可以使用占位符， 在处理大量数据时，使用 `tf.data$Datasets` 来向模型传递数据是更好的方法。

从数据集（Dataset）中获取一个能够运行的 `tf.Tensor` 对象，你需要先将其转换为 `tf.data.Iterator` 类型，然后调用迭代器（Iterator）的 `tf.data.Iterator.get_next` 方法。

最简单的办法就是使用 `tf.data.Dataset.make_one_shot_iterator` 创建一个迭代器。比如下列的代码中，每次调用 `run`，`next_item` 会返回 `my_data` 数组中的一行。

``` python
my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()
```

当数据流到达末尾时，`Dataset` 会抛出 `tf.errors.OutOfRangeError`。下面的例子就展示了如何读取数据直到所有数据都被获取。

``` python
while True:
  try:
    print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
    break
```

如果 `Dataset` 依赖于有状态操作，那么使用它之前，你可能需要初始化迭代器，如下所示：

``` python
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break
```

更多关于数据集和迭代器的内容请参考：[导入数据](../guide/datasets.md)。

## 网络层

可训练的模型会不断地修改其参数值，从而对于相同的输入得到新的输出。向计算图中添加可训练的参数，使用 `tf.layers` 是我们推荐的方式。

网络层封装了变量和作用其上的操作。比如，[全连接层](https://developers.google.com/machine-learning/glossary/#fully_connected_layer))对输入进行一个加权求和的操作，并且可以作用一个可选择的[激活函数](https://developers.google.com/machine-learning/glossary/#activation_function)。连接的权重和偏置都由网络层对象来管理。

### 创建网络层

下面的代码创建了一个 `tf.layers.Dense` 层，它以一个向量批次（即多个向量）作为输入，并且对每个批次产生一个单值输出。你只需要像调用函数一样，就能够将该层作用在输入上：

```python
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)
```

网络层根据输入来决定内部的变量的形状大小，所以我们必须设置 `x` 这一占位符的形状，从而能够让网络层推断出正确的参数形状。

定义了输出的操作 `y` 之后，在我们运行之前，还有一点需要注意。

### 初始化网络层

网络层中的变量在使用之前，必须要**初始化**。你可以逐一的初始化每个变量，同样也能够和下面一样，简单地用一行代码初始化计算图中所有的变量：

```python
init = tf.global_variables_initializer()
sess.run(init)
```

重点：调用 `tf.global_variables_initializer` 创建并返回一个 TensorFlow 操作的句柄，当使用  `tf.Session.run` 来运行它时，这个操作将会初始化所有的全局变量。

还有一点需要注意，`global_variables_initializer` 只会初始化已经在计算图中定义的变量。所以初始化操作应该是构建计算图的最后一步操作。

### 执行网络层

网络层初始化之后，和其他张量一样，我们可以对 `linear_model` 的输出张量进行求值。如下面的代码所示：

```python
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))
```

结果是含有两个元素的向量：

```
[[-3.41378999]
 [-9.14999008]]
```

### 网络层函数快捷键

对于每个网络层类（像 `tf.layers.Dense`），TensorFlow 都提供了一个便捷函数（像 `tf.layers.dense`）。唯一的区别就是便捷函数能够只用一行就完成创建和调用操作，下面的代码和先前的是等价的：

```python
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
```

尽管很方便，但是这样的操作无法获取到 `tf.layers.Layer` 对象，这就会使得检查和 debug 变得困难，并且导致网络层无法重用。

## 特征列

体验特征列最简单的方法就是使用 `tf.feature_column.input_layer` 函数。这个函数接受将 [dense columns](../guide/feature_columns.md) 作为输入，所以你需要将输入通过 `tf.feature_column.indicator_column` 进行封装，才能查看类别列的结果。比如：

``` python
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)
```

运行 `inputs` 会把 `features` 变成一系列向量。

特征列和网络层一样具有内部状态，所以也需要被初始化。类别列在内部使用 `tf.contrib.lookup` 进行初始化，因而我们需要一个单独的初始化操作 `tf.tables_initializer`。

``` python
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))
```

内部状态初始化之后，就可以和其他张量一样运行 `inputs` ：

```python
print(sess.run(inputs))
```

这展示了特征列是如何对输入的向量进行编码：“部门”属性使用 one-hot 进行编码，占据每一个向量的前面两个索引（[1., 0.] 代表”sports"，[0., 1.] 代表“gardening”），“销量”属性则由第三个索引来表示：

```None
[[  1.   0.   5.]
 [  1.   0.  10.]
 [  0.   1.   8.]
 [  0.   1.   9.]]
```

## 训练

熟悉了 TensorFlow 的核心操作之后，一起来动手训练一个小的回归模型吧。

### 定义数据

先来定义一些输入，`x`，以及对应的输出 `y_true`：

```python
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)
```

### 定义模型

接下来，构建一个简单的线性模型，输出为一个值：

``` python
linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
```

你可以对模型的预测进行求值，如下面的代码所示：

``` python
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))
```

模型还没有经过训练，所以四个预测值和实际值相去甚远。下面是我们的结果，你的结果可能有些不一样：

``` None
[[ 0.02631879]
 [ 0.05263758]
 [ 0.07895637]
 [ 0.10527515]]
```

### 损失函数

为了优化一个模型，你需要先定义损失函数。我们使用经常应用在回归问题中的平方差作为损失函数。

你可以利用低级的数学操作来手动定义这个 loss，但是 `tf.losses` 已经提供了一系列常用的损失函数。你可以像下面的代码一样，使用它们来计算平方差：

``` python
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

print(sess.run(loss))
```
这会打印出损失函数的值，比如：

``` None
2.23962
```

### 训练

TensorFlow 提供了实现了标准的优化算法[**优化器**](https://developers.google.com/machine-learning/glossary/#optimizer)，这些优化器是 `tf.train.Optimizer` 的子类。它们通过对变量的微小改变来最小化损失函数，最简单的优化算法就是[梯度下降](https://developers.google.com/machine-learning/glossary/#gradient_descent)，`tf.train.GradientDescentOptimizer` 实现了这一算法。它通过损失函数对各个变量的微分大小来进行变量的更新，你可以这样使用它：

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

上面的代码构建了优化过程所需的所有计算图的组件，并且返回了一个训练操作。运行这一操作，计算图中的所有变量会被更新，你可以像下面一样运行：

```python
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)
```

因为 `train` 是一个操作，而不是一个张量，所以运行的结果并不会返回值。我们可以运行 `loss_value` 张量来获知训练的过程，结果如下：

``` None
1.35659
1.00412
0.759167
0.588829
0.470264
0.387626
0.329918
0.289511
0.261112
0.241046
...
```

### 完整程序

```python
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))
```

## 接下来

想要学习更多关于使用 TensorFlow 构建模型的内容，你可以参考一下内容：

* 阅读[个性化 Estimator](../guide/custom_estimators.md)，学习如何构建定制化的模型。你对 TensorFlow Core 的了解能够帮助你更好的理解和 debug 你的模型

如果你想要学习更多 TensorFlow 内部的工作原理，你可以参考下列文档，其中对我们涉及的话题有更深入的介绍：

* [图表和会话](../guide/graphs.md)
* [Tensors](../guide/tensors.md)
* [变量](../guide/variables.md)
