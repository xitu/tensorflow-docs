# 深入 MNIST

TensorFlow 是一个非常强大的用来做大规模数值计算的库。其所擅长的任务之一就是实现以及训练深度神经网络。
在本教程中，我们将学到构建一个 TensorFlow 模型的基本步骤，并将通过这些步骤为 MNIST 构建一个深度卷积神经网络。

**这个教程假设你已经熟悉神经网络和 MNIST 数据集。如果你尚未了解，请查看
@{$beginners$introduction for beginners}。 并确认
@{$install$install TensorFlow} 已经在你的机器上安装。**


## 本教程的内容

第一部分将会讲解代码 [mnist_softmax.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_softmax.py) 是如何实现一个基本的 TensorFlow 模型的。
第二部分将会讲解我们提高精度用到的一些方法。


你可以跟随着教程拷贝代码在 Python 环境下运行，也可以在[这里下载](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_deep.py)
整份代码。

所以，这篇教程将主要包含下面四个部分：

- 通过学习图片上每一个像素值，创建一个 softmax 回归函数来识别 MNIST 图片里的数字
- 将上千张的手写图片作为样本输入到 TensorFlow 中，可以训练出识别手写数字的模型。（通过启动 TensorFlow session 来训练获得模型）
- 使用测试数据衡量模型的精确度
- 构建、训练并且测试一个具备多层卷积的神经网络，以获得更高精度的模型

## 准备工作

在创建我们的模型之前，首先需要加载 MNIST 数据库，然后启动一个 TensorFlow 的 Session。

### 加载 MNIST 数据库

如果你拷贝并粘贴的下面这两行代码，那么当你运行后他会自动地下载所需要的数据。


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

这里 `mnist` 变量是一个轻量级的类，里面包含了训练，验证，和测试的 NumPy 数组。他同时还提供了批量训练的迭代器，在下面的代码中会用到。


### 运行 TensorFlow 的 InteractiveSession

Tensorflow 依赖于一个高效的 C++ 后端来进行计算。与后端的这个连接叫做 session。一般而言，使用 TensorFlow 程序的流程是先创建一个图，然后在 session 中启动它。

这里，我们使用更加方便的 `InteractiveSession` 类。通过它，你可以更加灵活地构建你的代码。它能让你在运行图的时候，插入一些计算图@{$get_started/get_started#the_computational_graph$computation graph}，这些计算图是由某些操作 (operations) 构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用 IPython。如果你没有使用 `InteractiveSession`，那么你需要在启动 session 之前构建整个计算图，然后启动该计算图@{$get_started/get_started#the_computational_graph$launching the graph}。


```python
import tensorflow as tf
sess = tf.InteractiveSession()
```

### 计算图

为了在 Python 中进行高效的数值计算，我们通常会使用像 [NumPy](http://www.numpy.org) 一类的库，将一些诸如矩阵乘法的耗时操作在 Python 环境的外部来计算，这些计算通常会通过其它语言并用更为高效的代码来实现。

但遗憾的是，每一个操作切换回 Python 环境时仍需要不小的开销。如果你想在 GPUs 或者分布式环境中计算时，这一开销更加可怖，这一开销主要可能是用来进行数据迁移。

TensorFlow 也是在 Python 外部完成其主要工作，但是进行了改进以避免这种开销。其并没有采用在 Python 外部独立运行某个耗时操作的方式，而是先让我们描述一个交互操作图，然后完全将其运行在 Python 外部。这与 Theano 或 Torch 的做法类似。

因此 Python 代码的目的是用来构建这个可以在外部运行的计算图，以及安排计算图@{$get_started/get_started#the_computational_graph$Computation Graph}的哪一部分应该被运行。详情请查看基本用法@{$get_started/get_started}中的计算图一节。

## 构建 Softmax 回归模型

在这一节中我们将建立一个拥有一个线性层的 softmax 回归模型。在下一节，我们会将其扩展为一个拥有多层卷积网络的 softmax 回归模型。

### 占位符

我们通过为输入图像和目标输出类别创建节点，来开始构建计算图。


```python
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```

这里的 `x` 和 `y_` 并不是特定的值，相反，他们都只是一个占位符，可以在 TensorFlow 运行某一计算时通过该占位符输入具体的值。

输入图片 `x` 是一个 2 维的浮点数张量。这里，分配给它的 shape 为 [None, 784] ，其中 784 是一张展平的 MNIST 28 * 28 尺寸图片的维度。None 表示其值大小不定，在这里作为第一个维度值，用以指代 batch 的大小，意即每次输入的图片的数量不定。输出类别值 `y_` 也是一个2维张量，其中每一行为一个 10 维的 one-hot 向量,用于代表对应某一 MNIST 图片的类别（从一到九）。

虽然 placeholder 的 shape 参数是可选的，但有了它，TensorFlow 能够自动捕捉因数据维度不一致导致的错误。

### 变量

我们现在为模型定义权重 `W` 和偏置 `b`。可以将它们当作额外的输入量，但是 TensorFlow 有一个更好的处理方式：变量。一个变量代表着 TensorFlow 计算图中的一个值，能够在计算过程中使用，甚至进行修改。在机器学习的应用过程中，模型参数一般用 `Variable` 来表示。

```python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

我们在调用 `tf.Variable` 的时候为参数传入初始值。在这个例子里，我们把 `W` 和 `b` 都初始化为零向量。W 是一个 784x10 的矩阵（因为我们有 784 个输入特征和 10 个输出值）。b 是一个 10 维的向量（因为我们有 10 个分类）。

变量需要通过 seesion 初始化后，才能在 session 中使用。这一初始化步骤为，为初始值指定具体值（本例当中是全为零），并将其分配给每个变量,可以一次性为所有变量完成此操作。

```python
sess.run(tf.global_variables_initializer())
```

### 类别预测与损失函数

现在我们可以实现我们的回归模型了。这只需要一行！我们把向量化后的图片 `x` 和权重矩阵 `W` 相乘，加上偏置 `b`。

```python
y = tf.matmul(x,W) + b
```

可以很容易地为训练过程指定最小化误差的损失函数，损失表示模型在对样本类别预测时误差的程度。我们尝试在训练过程中调整参数来尽可能地缩小误差。因此，我们的损失函数就是目标类别和预测类别之间的交叉熵。正如下面代码所定义的那样。

```python
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
```

注意，`tf.nn.softmax_cross_entropy_with_logits` 应用在没有归一化的预测结果值下，然后累加每个类别对应的预测值，`tf.reduce_mean` 则对累加值进行求和。

## 训练模型

我们已经定义好模型和训练用的损失函数，那么用 TensorFlow 进行训练就很简单了。因为 TensorFlow 知道整个计算图，它可以使用自动微分法找到对于各个变量的损失的梯度值。TensorFlow 有大量内置的优化算法@{$python/train#optimizers$built-in optimization algorithms}，这个例子中，我们用随机梯度下降法让交叉熵值下降，学习率为 0.5。

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

TensorFlow 执行上述那一行代码后会添加一个新的操作到计算图中，其中包括计算梯度，计算每个参数的步长变化，并且计算出新的参数值。

返回的 `train_step` 操作对象，在运行时会使用梯度下降来更新参数。因此，整个模型的训练可以通过反复地运行 `train_step` 来完成。

```python
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
```

每一步迭代，我们都会加载 100 个训练样本，然后执行一次 `train_step`，并通过 `feed_dict` 将 `x` 和 `y_` 张量占位符用训练数据替代。

注意，在计算图中，你可以用 `feed_dict` 来替代任何张量，并不仅限于替换占位符。

### 评估模型

那么我们的模型性能如何呢？

首先让我们找出那些预测正确的标签。`tf.argmax` 是一个非常有用的函数，它能给出张量在某一维上最大值所在的索引值。由于标签向量是由 0, 1 组成，因此最大值1所在的索引位置就是类别标签，比如 `tf.argmax(y,1)` 返回的是模型对于任一输入 x 预测到的标签值，而 `tf.argmax(y_,1)` 代表正确的标签，我们可以用 `tf.equal` 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

这里返回一个布尔数组。为了计算我们分类的正确率，我们将布尔值转换为浮点数来表示，然后取平均值。例如：`[True, False, True, True]` 变为 `[1,0,1,1]`，计算出平均值为 `0.75`。

```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

最后，我们可以计算出在测试数据上的准确率，大概是 92%。


```python
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

## 构建一个多层卷积网络

在 MNIST 上只有 92% 正确率，实在太糟糕。在这个小节里，我们用一个稍微复杂的模型：卷积神经网络来改善效果。这会达到大概 99.2% 的准确率。虽然不是最高，但是还是比较让人满意。

下面是用 TensorBoard 创建的可视化的计算图，也是我们接下来要构建的模型：

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img src="https://www.tensorflow.org/images/mnist_deep.png">
</div>

### 权重初始化

为了创建这个模型，我们需要创建大量的权重和偏置项。这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免 0 梯度。由于我们使用的是 [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) 激活函数，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为 0 的问题（dead neurons）。为了不在建立模型的时候反复做初始化操作，我们定义两个函数用于初始化。

```python
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```

### 卷积和池化

TensorFlow 在卷积和池化上有很强的灵活性。我们怎么处理边界？步长应该设多大？在这个实例里，我们会一直使用 vanilla 版本。我们的卷积使用 1 步长（stride size），0 边距（padding size）的模板，保证输出和输入是同一个大小。我们的池化用简单传统的 2x2 大小的模板做最大值（max pooling）。为了代码更简洁，我们把这部分抽象成一个函数。

```python
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```

### 第一层卷积

现在我们可以开始实现第一层了。它由一个卷积接一个嘴大池（max pooling）完成。卷积在每个 5x5 的 patch 中算出 32 个特征。卷积的权重张量形状是 [5, 5, 1, 32] ，前两个维度是 patch 的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。

```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

为了用这一层，我们把 `x` 变成一个 4d 向量，其第 2、第 3 维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为 1，如果是 rgb 彩色图，则为 3)。

```python
x_image = tf.reshape(x, [-1, 28, 28, 1])
```

We then convolve `x_image` with the weight tensor, add the
bias, apply the ReLU function, and finally max pool. The `max_pool_2x2` method will
reduce the image size to 14x14.

我们把 `x_image` 和权值张量进行卷积，加上偏置项，然后应用 ReLU 激活函数，最后执行 `max_pool_2x2`。

```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```

### 第二层卷积

为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个 5x5 的 patch 会得到 64 个特征。

```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

### 全连接层

现在，图片尺寸减小到 7x7，我们加入一个有 1024 个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量 reshape 成一些向量，乘上权重矩阵，加上偏置，然后对其使用 ReLU。

```python
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

#### Dropout

为了减少过拟合，我们在输出层之前加入 [dropout](
https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)。我们用一个 `placeholder` 来代表一个神经元的输出在 `dropout` 中保持不变的概率。这样我们可以在训练过程中启用 `dropout`，在测试过程中关闭 `dropout`。 TensorFlow 的 `tf.nn.dropout` 操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的 `scale`。

```python
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

### 输出层

最后，我们添加一个 softmax 层，就像前面的单层 softmax 回归一样。

```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
```

### 训练和验证模型

这个模型的效果如何呢？我们训练和评价模型的方式大体与前面评价单层 SoftMax 网络的差不多。

区别主要有以下三点：

- 我们将使用性能更好的 ADAM 优化器，而不是梯度下降

- 我们在 `feed_dict` 中加入额外的参数 `keep_prob` 来控制 `dropout` 比例

- 在训练的步骤中，每 100 次迭代的时候打印训练的精确度

我们还使用了 tf.Session 而不是 tf.InteractiveSession。这样可以更好的隔离构建图和评价模型的过程，从而让代码显得简洁。tf.Session 是创建在 [`with`](https://docs.python.org/3/whatsnew/2.6.html#pep-343-the-with-statement) 代码块里的，所以块内代码执行完后会自动销毁相关变量。

你可以随意运行这段代码。需要注意的是代码中包含了 2000 次的迭代，所以可能会花费一些时间来执行（可能半个小时），这取决于你机器的配置好坏。

```python
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

跑完测试集后精确度应该接近 99.2%。

目前为止，我们已经学会了用 TensorFlow 快捷地搭建、训练和评估一个复杂一点儿的深度学习模型。

<b id="f1">1</b>: 对于这个复杂度并不高的神经网络，加不加 Dropout 对性能几乎没有影响。但 Dropout 在训练大型神经网络的时候能够很有效的防止过拟合。[↩](#a1)