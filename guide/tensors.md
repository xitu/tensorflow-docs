# 张量（Tensors）

顾名思义，TensorFlow 是一个定义和运行张量计算的框架。**张量**是各种维度的向量和矩阵的统称。在内部，TensorFlow 用基本数据类型的多维数组来表示张量。

当你在写 TensorFlow 程序的时候，主要操作和传递的对象就是 `tf.Tensor`，即所谓的张量。一个 `tf.Tensor` 对象定义了计算的一部分，并最终会生成一个值。TensorFlow 程序的工作机制是先构建一张 `tf.Tensor` 对象的计算图，其中详细地展示了各个张量之间的运算关系，然后再通过运行这张计算图的各个部分来产生需要的结果。

一个 `tf.Tensor` 对象有以下属性：

 * 一种数据类型（比如`float32`，`int32`， 或者是 `string`）
 * 一个形状

张量中的每个元素有着相同的数据类型，并且是已知的。而张量的形状（也就是维度的数量和各个维度的大小）可能是部分已知。如果其输入形状是完全已知的，那么大部分操作会产生已知形状的张量，但有些时候只有在计算图执行时才能确定张量的形状。

有些张量的类型比较特殊，会在其他的 TensorFlow 指南章节中有所说明，主要有以下几种：

  * `tf.Variable`
  * `tf.constant`
  * `tf.placeholder`
  * `tf.SparseTensor`

除了 `tf.Variable` 以外，张量的值是不可变的，也就是说张量在单次执行的上下文中值是唯一的。但是，两次对同一个张量求值可能返回不同的值，比如，张量的值可能是从磁盘读取的数据，或者是一个随机数，那么每次产生的结果可能是不一样的。

## 秩

`tf.Tensor` 对象的**秩**就是它维度的数量。秩的也可以叫做**阶数**、**度数**或者是 **n 维**。注意：Tensorflow 里的秩和数学中矩阵的秩是不一样的。如下表所示，Tensorflow 中的不同的秩代表不同的数学实体：

| 秩    | 数学实体              |
| ---- | ----------------- |
| 0    | 标量（只有大小）          |
| 1    | 向量 (有大小和方向)       |
| 2    | 矩阵 (由数构成的表)       |
| 3    | 3 维张量 (由数构成的方体)  |
| n    | n 维张量 (你可以自行想象一下) |


### 秩为 0

下列代码片段展示了如何创建一些秩为 0 的变量：

```python
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)
```

注意：一个 string 对象在 Tensorflow 中作为一个单独的对象，而不是一个字符序列。字符串可以作为标量，也可以作为向量等类型出现。

### 秩为 1

你可通过传递一个列表作为初始值来创建一个秩为 1 的 `tf.Tensor` 对象，比如：

```python
mystr = tf.Variable(["Hello"], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)
```


### 更高的秩

一个秩为 2 的 `tf.Tensor` 对象至少由一行和一列（类似一个二维数组）组成：

```python
mymat = tf.Variable([[7],[11]], tf.int16)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)
```

类似地，具有更高秩的张量，由一个 n 维数组组成。比如，在图像处理过程中，我们会使用很多的秩为 4 的张量，其各个维度分别代表一个样本批次的大小、图像的宽度、图像的高度以及颜色通道数。

``` python
my_image = tf.zeros([10, 299, 299, 3])  # batch 大小 x 高度 x 宽度 x 颜色通道数
```

### 获取 `tf.Tensor` 对象的秩

我们可以通过调用 `tf.rank` 方法来获得一个 `tf.Tensor` 对象的秩。比如，下面的代码展示了如何获得先前定义的一个 `tf.Tensor` 对象的秩：

```python
r = tf.rank(my_image)
# 运行后，r 的值为 4
```

### 引用 `tf.Tensor` 切片

因为 `tf.Tensor` 对象是一个由各个单元组成的 n 维数组，你可以通过指定 n 维索引来获得其中的某个单元。

对于一个秩为 0 的张量（标量），因为它已经是一个数了，所以不需要使用索引。

对于一个秩为 1 的张量（向量），传递单个索引就可以获得其中的一个数值：

```python
my_scalar = my_vector[2]
```

注意，`[]` 传递的索引本身也可以是一个标量的 `tf.Tensor` 对象，这样你就可以动态地从向量中选取元素了。

对于秩为大于等于 2 的张量，就更有趣了。秩为 2 的 `tf.Tensor` 对象，传递 2 个索引值，不出所料，它会返回一个数：


```python
my_scalar = my_matrix[1, 2]
```

而看下面的代码，如果你只传递一个索引值，那么会返回矩阵的一个子向量，比如：


```python
my_row_vector = my_matrix[2]
my_column_vector = my_matrix[:, 3]
```

`:` 符号是 Python 的切片语法，代表“取这一维所有的对象”。这在高秩数的张量中非常的实用，你能够凭此来访问、操作它的子向量、子矩阵甚至子张量。


## 形状

张量的**形状**是其各个维度元素的数量。TensorFlow 能够在图的构建过程中自动推断张量的形状。这些推断出的形状的秩可能是已知的，也可能是未知的。即使张量的秩已知，其各个维度的大小也可能是未知的。

TensorFlow 开发者指南中使用三种传统的表示方法来描述向量的维度：秩、形状、和维数。下表展示了这几种表示方法之间的关系：

| 秩    | 形状                         | 维数  | 例子                                   |
| ---- | ---------------------------- | ---- | -------------------------------------- |
| 0    | []                           | 0 维 | 一个 0 维张量，一个标量。                  |
| 1    | [维度 0]                      | 1 维 | 一个形为 [5] 的 1-D 张量。                |
| 2    | [维度 0, 维度 1]               | 2 维 | 一个形为 [3, 4] 的 2-D 张量。             |
| 3    | [维度 0, 维度 1, 维度 2]        | 3 维 | 一个形为 [1, 4, 3] 的 3-D 张量。          |
| n    | [维度 0, 维度 1, ... 维度 n-1] | n 维 | 形为 [维度 0, 维度 1, ... 维度 n-1] 的张量。 |

形状可以通过 Python int 类型的列表或者是元组来表示，或者是 `tf.TensorShape`。

### 获取 `tf.Tensor` 对象的形状

获取 `tf.Tensor` 对象的形状有两种方式。在构建计算图的时候，检查一下张量的形状的已知部分是很有帮助的。我们可以通过读取 `tf.Tensor` 对象的 `shape` 属性来获知其已知部分。这种方法会返回一个 `TensorShape` 对象，在表示一些部分已知的形状时非常方便（因为在构建计算图的时候不是所有张量的形状都是完全已知的）。

运行时，我们也可以通过调用 `tf.shape` 操作，用 `tf.Tensor` 来表示另一个 `tf.Tensor` 的形状。这样，你就能够根据输入的 tf.Tensor 构建其他张量来操作张量的形状

下面的例子，展示了如何创建一个长度和已知矩阵的列数相同的零向量：

``` python
zeros = tf.zeros(my_matrix.shape[1])
```

### 改变 `tf.Tensor` 对象的形状

张量**元素的个数**是形状数组中各个维度大小的乘积。标量的元素个数始终是 `1`。因为很多不同的形状包含的元素数量是一样的，在保证元素的个数不变的前提下，我们可以很方便地改变其形状。我们通过使用 `tf.reshape` 来实现形状的改变。

下面的例子展示了如何改变张量的形状：

```python
rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # 把 rank_three_tensor 变成一个 
                                                 # 6x10 的矩阵 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  把 matrix 变成一个  3x20 的矩阵 matrixB
                                       #  -1 告诉 reshape 方法
                                       #  自动计算这一维度的大小
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # 把 matrixB 变成一个
                                             # 4x3x5 的张量 matrixAlt

# 注意，变形后的张量元素个数必须和原有张量中的元素个数相同。
# 如果不同，就会产生错误，因为无法确定某一个维度的元素的数量

yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # 错误
```

## 数据类型

除了维度以外，张量还有一个数据类型的属性。关于所有数据类型的完整列表，请参阅 `tf.DType` 页面。

`tf.Tensor` 对象只能有一种数据类型。但是，可以将任意数据结构作为字符串序列化并将其存储在 `tf.Tensor` 对象中。

我们可以使用 `tf.cast` 来进行数据类型的转换：

``` python
# Cast a constant integer tensor into floating point.
# 把一个常量整形张量转换为浮点数类型
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
```

To inspect a `tf.Tensor`'s data type use the `Tensor.dtype` property.

使用 `Tensor.dtype` 这个属性来获得 `tf.Tensor` 对象的数据类型。

从 Python 对象创建 `tf.Tensor` 时，是否指定数据类型是可选的。如果没有显式地指定，TensorFlow 会选择一个能够表示数据的数据类型。TensorFlow 会把 Python 中的整形转换为 `tf.int32` 并且把浮点数转换为 `tf.float32`。TensorFlow 其他的转换规则和 numpy 的相同。

## 对张量求值

计算图构建完成之后，你可以运行计算来产生特定的 `tf.Tensor` 对象并且获取它的值。这对程序的调试以及运行 TensorFlow 程序的时候是非常有帮助。

对一个张量求值最简单的方法就是使用 `Tensor.eval` 方法，比如：

```python
constant = tf.constant([1, 2, 3])
tensor = constant * constant
print(tensor.eval())
```

`eval` 方法只有当启用一个默认的 `tf.Session` 才能正常工作（你可以查看 Graphs 和 Sessions 开发者指南来了解更多）。

`Tensor.eval` 方法会返回一个和张量内容相同的 numpy 数组。

当 `tf.Tensor` 所需的动态信息不完全时，是无法对它求值的。比如，依赖 `placeholder` 的张量在没有给 `placeholder` 提供值之前是无法被评估的。

``` python
p = tf.placeholder(tf.float32)
t = p + 1.0
t.eval()  #  这会失败，因为 p 还没有被赋值
t.eval(feed_dict={p:2.0})  # 这就能够成功，因为我们通过 feed_dict 给 p 赋值了
```

注意，除了占位符（placeholder）以外，其他的 `tf.Tensor` 对象也是能够被赋值的。

一些模型的结构可能使得对 `tf.Tensor` 求值变得复杂。TensorFlow 无法直接对函数中或者是控制流结构中的 `tf.Tensor` 求值。如果一个 `tf.Tensor` 的值来自于一个队列，那么只有在某些东西入队后求值才能进行；否则，求值会被暂时挂起。在处理队列时，请记住在对任何 `tf.Tensor` 求值之前先调用  `tf.train.start_queue_runners` 。

## 打印张量

你可能需要在调试的时候打印一个 `tf.Tensor` 对象的值。尽管 [tfdbg](../guide/debugger.md) 提供了非常高级的调试支持，但 TensorFlow 依旧有着一些可以直接打印 `tf.Tensor` 值的操作。

注意，你可别用下面的这种方式打印 `tf.Tensor` ：

``` python
t = <<some tensorflow operation>>
print(t) # 当计算图构建完成后，将会打印出这个张量
         # 这个张量在这种情况下没有值
```

这段代码打印了 `tf.Tensor` 这个对象（代表着其所进行的运算）而不是它的值。相反，TensorFlow 提供了`tf.Print`操作，该操作会原封不动地返回第一个参数，并将第二个参数（`tf.Tensor`）打印出来。

正确使用 `tf.Print` 的方式是使用其返回值，如下面的例子所示：

``` python
t = <<some tensorflow operation>>
tf.Print(t, [t])  # 这行代码是不起作用的
t = tf.Print(t, [t])  # 我们使用 tf.Print 的返回值
result = t + 1  # 在求 result 的值的时候 `t` 的值将会被打印
```

当你在对 `result` 求值的时候，你将会求出所有 `result` 所依赖的张量的值。因为 `result` 依赖于 `t`，因此在对 `t` 求值的时候会同时打印 `t` 的值（`t` 原来的值，即 `t = <<某些 TensorFlow 操作>>`），这样一来 t 就被打印出来了
