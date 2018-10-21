# 用 Estimators 构建卷积神经网络（CNN）

`tf.layers` 模块提供的高层 API 能让构建神经网络变得简单。它提供了一些便利的方法来创建全连接层和卷积层，添加激活函数，以及应用 dropout 正则化。在这篇教程中，你将会学习到如何使用 `layers` 来创建一个识别手写数字图片（来自于 MNIST 数据集）的卷积神经网络模型。

![MNIST 数据集中 0-9 的手写数字](../images/mnist_0-9.png)

**MNIST 数据集由 60,000 张训练样本和 10,000 张测试样本组成，这些样本表示 0-9 的手写数字，都被处理为 28x28 像素大小的灰度图片。**

## 准备开始

新建一个名为 `cnn_mnist.py` 的文件，然后在里面编写 TensorFlow 程序的框架代码。

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 导入依赖模块
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# 程序的逻辑将会被添加到这里

if __name__ == "__main__":
  tf.app.run()
```

阅读本教程，你将学会如何编写代码，来构建、训练并运行一个卷积神经网络。该教程完整的代码可以[查看这里](https://www.tensorflow.org/code/tensorflow/examples/tutorials/layers/cnn_mnist.py)。

## 卷积神经网络的简介

卷积神经网络（CNNs）是当前用户图像分类任务中最前沿的模型。CNNs 对图像的原始像素数据应用了一系列的过滤器，以提取和学习更高层次的特征，然后模型利用这些特征对图像进行分类。CNNs 主要包含下面三个组件：


*   **卷积层**，它表示应用在图像中卷积核的数量。对于图片的子区域，卷积层会执行一系列的数学变换，从而输出特征映射的值。卷积层一般情况下会使用 [ReLU 做为激活函数](https://en.wikipedia.org/wiki/Rectifier_\(neural_networks\))来让模型引入非线性变换。

*   **池化层**，它是对卷积层提取出的图像数据进行[下采样](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer)，作用是可以减少特征映射的维度，从而减少计算的时间。常用的池化算法是最大池化算法，它提取的像素值是池化窗口（e.g., 2x2-像素块）中值最大的，而子区域中其他的像素值则被抛弃。

*   **稠密（全连接）层**，在经过卷积层和池化层的采样后，全连接层就可以对特征进行分类了。具体来说，在全连接层中，层中的每个节点都与上一层的结点相连。

一般来说，CNN 是通过多层卷积模块来提取特征的。每一个模块都包含一个卷积层，后面跟着一个池化层。最后一个卷积模块后面跟着一层或者多层的全连接层来获得分类结果。CNN 中的最后一个全连接层结点的数量等于预测任务所有可能类别的数量，而这些结点的值通过 softmax 激活函数后会产生一个 0~1 的值（该层所有的结点值之和为 1）。这些 softmax 值可以解释为输入图片最有可能是属于哪个类别的概率。

> 注意：想要更深入了解 CNN 的架构，请看斯坦福大学的 <a href="https://cs231n.github.io/convolutional-networks/">卷积神经网络课程资料</a>

## 构建基于卷积神经网络的 MNIST 分类器

基于 CNN 架构，让我们构建一个模型来对 MNIST 数据集中的图像进行分类：


1.  **第一个卷积层**：应用 32 个 5x5 窗口大小的卷积核（提取 5x5-像素的子区域）和 ReLU 激活函数。
2.  **第一个池化层**：使用 2x2 窗口大小的最大池化过滤器来做采样，且窗口每次滑动的步长为 2（步长的作用是设置窗口采样时的重叠程度）。
3.  **第二个卷积层**：应用 64 个 5x5 窗口大小的卷积核，和 ReLU 激活函数。
4.  **第二个池化层**：和第一个池化层的操作一样，2x2 的采样窗口，步长 2。
5.  **第一个全连接层**：1,024 个神经元和 dropout 的正则化率为 0.4（训练时随机屏蔽的神经元占比）。
6.  **第二个全连接层（逻辑层）**：10 个神经元，每个神经元代表着 0~9 中的一个类别。

`tf.layers` 模块中包含创建上述卷积神经网络三种类型的层的方法：

*   `conv2d()`：构建一个两维的卷积层。输入的参数是卷积的核数，大小，边缘填充方式和选择的激活函数。
*   `max_pooling2d()`：使用 max-pooling 池化算法构建一个二维的池化层。输入参数是池化的大小和步长。
*   `dense()`：构建稠密全连接层。输入参数是神经元数目和激活函数。

每一个方法都是接受一个张量然后再将转换后的张量作为输出。这使得层与层之间的连接变得简单：即上一层的输出可以直接作为下一层的输入。

打开 `cnn_mnist.py` 文件，然后添加 `cnn_model_fn` 函数，该函数是符合 TensorFlow 评估器 API 接口的要求的（更详细的指南可以查阅后续的[创建评估器](#create-the-estimator)的相关文档）。`cnn_minst.py` 文件将以 MNIST 的特征数据，标签数据和模型（来自 `tf.estimator.ModeKeys`：`TRAIN`、`EVAL` 和 `PREDICT`）作为参数来配置 CNN 架构，然后返回模型的预测，损失，和训练的操作。


```python
def cnn_model_fn(features, labels, mode):
  """CNN 的模型函数"""
  # 输入层
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # 第一个卷积层
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # 第一个池化层
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # 第二个卷积层和池化层
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # 全连接层
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits 层
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # (为 PREDICT 和 EVAL 模式)生成预测值
      "classes": tf.argmax(input=logits, axis=1),
      # 将 `softmax_tensor` 添加至计算图。用于 PREDICT 模式下的 `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # 计算损失（可用于`训练`和`评价`中）
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # 配置训练操作（用于 TRAIN 模式）
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # 添加评价指标（用于评估）
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```

接下来会分章节解释上述 `tf.layers` 的代码，内容包括每一层是如何创建的，损失值是如何计算出来的，训练操作是如何配置的，预测是如何生成的。如果你已经使用过 CNNs 和 [TensorFlow `Estimator`](../../guide/custom_estimators.md)，觉得上述的代码已经很直观明了了，你可以跳过这些章节直接看["训练和评估 CNN MNIST 分类器"](#train_eval_mnist)。

### 输入层

在 `layer` 模块中，用于二维图像数据的卷积层和池化层期望输入的张量维度默认为`[batch_size, image_height, image_width, channels]`。可以通过修改 `data_format` 的参数改变这种行为，定义如下：

*   **`batch_size`**：在训练过中，每次执行梯度下降时使用的样本子集大小。
*   **`image_height`**：样本图片的高度。
*   **`image_width`**：样本图片的宽度。
*   **`channels`**：样本图片的通道数。对于彩色图片，通道数为 3（红，绿，蓝）。对于灰度图片，就只有一个通道（黑）。
*   **`data_format`**：字符串，`channels_last`（default）或 `channels_first`。`channels_last` 对应于具有 `(batch, ..., channels)` 形状的输入，而 `channels_first` 对应于 *具有 `(batch, channels, ...)` 形状的输入。

在这里，我们的 MNIST 数据集图片是灰度图片，每张图片的大小是 28x28 像素，因此我们输入层数据的维度为`[batch_size, 28, 28, 1]`

如果输入的特征不能满足这个维度，我们可以使用下面的 `reshape` 操作来进行转换。

```python
input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
```

注意，这里的 batch_size 值为 `-1`，该值会根据输入 `features[x]` 和另外 3 个维度的值自动计算出来。这使我们可以将 `batch_size` 当成一个超参数来进行调整。举个例子，如果我们输入到模型的样本子集大小为 5，那么 `features["x"]` 会包含 3,920 个值（每个值对应每张图像像素的一个值，也即 5x28x28 = 3920），也就是说 `input_layer` 的形状为 `[5, 28, 28, 1]`，同样的，如果我们的输入样本子集大小为 100，`features["x"]` 就会包含 78,400 个值，也就是说 `input_layer` 的形状为 `[100, 28, 28, 1]`

### 第一个卷积层

第一个卷积层中，我们对输入层应用了 32 个 5x5 的卷积核和 ReLU 激活函数。我们用到了 `layer` 模块中的 `conv2d` 方法，如下所示：

```python
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
```

`inputs` 参数指定了我们的输入张量，这个张量的形状必须为 `[batch_size, image_height, inage_width, channels]`。在这里，我们将 `input_layer` 连接到第一个卷积层，它的形状是 `[batch_size, 28, 28, 1]`。

> 注意：如果你传入了参数 `data_format=channels_first`，那么 `conv2d()` 所接受的维数是 `[batch_size, channels, image_height, image_width]`。

参数 `filters` 指定的是具体应用的卷积核的数量（在这里，数量为 32），`kernel_size` 指定的是卷积核的尺寸 `[height, width]`（在这里，尺寸为 `[5, 5]`）

**小建议：**如果卷积核的高度和宽度一致的话，你可以传递一个单独整数给参数 `kernel_size`，譬如 `kernel_size=5`。

参数 `padding` 的输入值是两个枚举值中的一个（值不区分大小写）：`valid` （默认值）或 `same`。当你设置 `padding=same` 的时候，TensorFlow 将会在边界填充 0 值从而让输出的张量和输入的张量有相同的宽高，也即 28x28。（如果没有填充，那么 5x5 的卷积核会产生一个 24x24 形状的张量）

参数 `activation` 指定在每层输出时要应用的激活函数。在这里，我们使用的是 ReLU 激活函数 `tf.nn.relu`。

函数 `conv2d()` 的输出张量的形状为 `[batch_size, 28, 28, 32]`：以相同的高度和宽度作为输入，但是有 32 个通道，每个通道对应着一个卷积核的输出。

### 第一个池化层

接下来，我们将第一个池化层连接到我们刚创建的卷积层上去。我们使用 `layers` 中的 `max_pooling2d()` 方法来创建一个 2x2 大小，步长为 2 的最大池化过滤器。

```python
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
```

再次说明，`inputs` 指定了输入的张量，它的形状为 `[batch_size, image_height, image_width, channels]`。在这里，我们的输入的张量是 `conv1`，也就是第一个卷积层的输出，它的形状是 `[batch_size, 28, 28, 32]`。

> 注意：如果你传入了参数 `data_format=channels_first`，那么 `conv2d()` 所接受的形状是`[batch_size, channels, image_height, image_width]`。

参数 `pool_size` 指定了最大池化过滤器的维度 `[height, width]`（在这里维度值为 [2, 2]），该参数也可以接受一个单独的数字（譬如 `pool_size=2`）

参数 `strides` 指定了滑动步长的大小。在这里，我们设置步长的值为 2，它的含义是过滤器提取的子区域在高度和宽度上都间隔有 2 个像素（对于 2x2 的过滤器，我们所提取的子区域都不会重叠）。如果你要为高度和宽度设置不同的步长值，你可以传入一个类型为元组或列表的值（e.g., `stride=[3, 6]`）。

方法 `max_pooling2d()` 输出的张量（`pool1`）的形状为 `[batch_size, 14, 14, 32]`：2x2 的过滤器让高和宽分别减少了 50%。

### 第二个卷积层和池化层

如前所述，我们使用 `conv2d()` 和 `max_pooling2d()` 方法就可以连接和创建我们 CNN 的第二个卷积层和池化层。对于第二个卷积层，我们配置了 64 个窗口大小为 5x5 的卷积核，使用了 ReLU 激活函数，对于第二个池化层，我们使用了和第一个池化层一样的设置（大小为 2x2 且步长为 2 的最大池化过滤器）：

```python
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
```

注意第二个卷积层将第一个池化层的输出（`pool1`）作为输入，然后得到的输出张量为 `conv2`。张量 `conv2` 的形状为 `[batch_size, 14, 14, 62]`，高和宽与第一个池化层（`pool1`）相同，64 个通道表示应用的 64 个卷积核。

第二个池化层拿 `conv2` 作为输入，然后得到的 `pool2` 作为输出。`pool2` 的形状为 `[batch_size, 7, 7, 64]`（将高和宽的长度分别减少了 50%）

### 全连接层

接下来，我们将要为 CNN 添加全连接层（拥有 1,024 个神经元和 ReLU 激活函数），以用来对我们前面的卷积层和池化层所提取到的特征来做分类。在我们连接该层时，我们需要拉平 `pool2` 的形状为 `[batch_size, features]`，这时张量只有两维：

```python
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
```

在上面 `reshape()` 操作中，`-1` 表示 **`batch_size`** 的维数，它会根据输入的数据样本数动态的计算出来。每一个样本有 7 (`pool2` 的 height) * 7 (`pool2` 的 width) * 64 (`pool2` 的通道数) 个特征，因此我们的特征维数为 7 * 7 * 64（总共 3136 个）。输出的张量 `pool2_flat` 的形状是 `[batch_size, 3136]`

现在，我们可以使用 `layers` 模块中的 `dense()` 方法连接全连接层了。

```python
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
```

参数 `inputs` 指定了输入的张量：也就是拉平后的特征映射 `pool2_flat`。参数 `units` 指定了全连接层的神经元数（1,024）。参数 `activation` 指定了激活函数；同样，我们使用了 ReLU 激活函数，也即传入了 `tf.nn.relu` 值。

为了提高模型的效果，我们还在全连接层中应用了 dropout 正则化，使用 `layers` 模块中的 `dropout` 方法来定义：

```python
dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
```

同样，参数 `inputs` 指定了输入张量，它是上一个全连接层（`dense`）的输出张量。

参数 `rate` 指定了 dropout 的比率；在这里，我们的值是 `0.4`，意味着 40% 的神经元在训练期间会被随机的屏蔽。

参数 `training` 接受一个布尔值，它指定模型当前是否正在训练模式下运行；dropout 操作只会在此布尔值为 `True` 的时候执行。在这里，在这里，我们检查传递到我们的模型函数 `cnn_model_fn` 的 `mode` 是否是 `TRAIN` 模式。

我们的输出张量 `dropout` 的形状是 `[batch_size, 1024]`。

### Logits 层

神经网络中的最后一层是 logits 层，它将返回我们预测的原始值。逻辑层是一个有 10 个神经元，且默认有线性激活函数的全连接层（每个神经元对应 0~9 中的一个类别）。

```python
logits = tf.layers.dense(inputs=dropout, units=10)
```

CNN 最终张量由 `logits` 层输出，它的形状是 `[batch_size, 10]`

### 生成预测

我们的模型的 `logits` 层将我们的原始预测值作为一维张量返回，形状为 `[batch_size, 10]`。让我们将这些原始值转换成模型函数所支持的两种不同格式：

*   每个样本的**预测的类别**：0~9 的数字。
*   每个样本在不同类别下的**概率**：样本是 0 的概率，样本是 1 的概率，样本是 2 的概率，等等。

对于一个给定的例子，模型输出的张量中数值最大的值对应的下标为预测的类别，使用 `tf.argmax` 函数找到该数值对应的索引：

```python
tf.argmax(input=logits, axis=1)
```

参数 `input` 指定了提取最大值的张量，这里传入的张量是 `logits`，用于提取最大值。参数 `axis` 指定了应该沿着 `input` 的哪个轴找最大值，这里传入的值是 1，它意味着我们沿着第二个维度来找最大值，这对应我们输出的预测张量的形状 `[batch_size, 10]` 中的 10。

使用 softmax `tf.nn.softmax` 激活函数可以从 logits 层获得类别对应的概率值。

```python
tf.nn.softmax(logits, name="softmax_tensor")
```

> 注意：我们使用参数 `name` 给这个操作命名为 `softmax_tensor`，这样的话我们就可以在后面引用他。（我们将在[“设置日志钩”](#set-up-a-logging-hook)中为 softmax 值设置日志记录）。

我们用一个字典数据结构来表示预测，然后生成一个 `EstimatorSpec` 对象：

```python
predictions = {
    "classes": tf.argmax(input=logits, axis=1),
    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
}
if mode == tf.estimator.ModeKeys.PREDICT:
  return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
```

### 计算损失

对于训练（`TRAIN`）和评价（`EVAL`）环节，我们需要定义[损失函数](https://en.wikipedia.org/wiki/Loss_function)来衡量预测类别和真实类别之间的差距。对于像 MNIST 这样的多分类问题，我们常用[交叉熵](https://en.wikipedia.org/wiki/Cross_entropy)作为损失的度量。下面的代码将会在训练或者验证模式下计算对应的交叉熵。

```python
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
```

让我们清楚地了解一下上面的代码做了什么。

`labels` 张量中包含了我们样例的预测值的索引值，比如 `[1, 9, ...]`。`logits` 中则包含了我们最后一层线性层得到的输出。

`tf.losses.sparse_softmax_cross_entropy` 将高效、稳定地计算两个输入值的 softmax 交叉熵（又名分类交叉熵、负对数似然）。

### 配置训练操作

在上面的章节，我们定义了交叉熵损失函数。接下来让我们在训练中配置我们的模型来最优化这个损失值。我们使用的最优化算法是[随机梯度下降法](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)，对应的学习率为 0.001 。

```python
if mode == tf.estimator.ModeKeys.TRAIN:
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  train_op = optimizer.minimize(
      loss=loss,
      global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
```

> 请在 ["Creating Estimations in tf.estimator"](../../guide/custom_estimators.md) 教程中阅读 ["Defining the training op for the model"](../../guide/custom_estimators.md#defining-the-training-op-for-the-model) 一节，了解更多关于训练函数的内容。

### 添加评价指标

通过在 EVAL 模式中定义 `eval_metric_ops` 字典，我们可以给模型添加准确度评价指标：

```python
eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
```

<a id="train_eval_mnist"></a>

## 训练并评价此 CNN MNIST 分类器

我们已经完成了 CNN 模型的代码工作；现在我们准备训练和评价它。

### 加载训练和测试数据

首先，我们需要加载训练和测试数据。在 `cnn_mnist.py` 文件中的 `main()` 函数添加下面的代码：

```python
def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
```

我们将训练特征数据（55, 000 张手写数字图片数据的原始像素值）和标注数据（每张图片对应的 0~9 的值）分别存储为 `train_data` 和 `train_labels` 中，格式为 [numpy 数组](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)。类似地，我们将用于评价的特征数据（10,000 张图片）和相应的标注数据分别存储在 `eval_data` 和 `eval_labels` 中。

### 创建评估器（Estimator）

接下来，在 `main()` 函数添加下面的代码，它的作用是为我们的模型创建 `Estimator`（一个用于执行模型训练，评价和推断的 TensorFlow 类）：

```python
# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
```

参数 `model_fn` 指定了用于训练，评价和预测的模型函数；我们传入的 `cnn_model_fn` 函数是在[构建 CNN MNIST 分类器](#building-the-cnn-mnist-classifier)中创建的。参数 `model_dir` 指定了模型数据（检查点）保存的目录（这里我们传入的目录是 `/tmp/mnist_convnet_model`，这个目录是可以更改的）。

> 注意：如果要更深入的了解 TensorFlow 的 `Estimator` API，请查阅 ["Creating Estimators in tf.estimator."](../../guide/custom_estimators.md) 教程。

### 建立一个日志钩子

训练 CNNs 需要耗费一定的时间，在这个过程中记录一些日志可以方便我们追踪在训练的进展。如果我们想在训练过程中输出 CNN 的 softmax 层的值，我们可以在 `main()` 函数中使用 TensorFlow 的 tf.train.SessionRunHook` 创建一个 `tf.train.LoggingTensorHook`：

```python
# 为预测过程设置日志
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)
```

我们可以用字典储存想要打印的张量 `tensors_to_log`。每个键值只不过是用于日志输出的一个别名，它的值则是 TensorFlow 计算图中的某个张量的名称。这里的的 `softmax_tensor` 是前面 `cnn_model_fn` 中创建的一个用于生成概率的张量的名称，而 `probabilities` 是这里给它取的别名。

> 注意：如果你没有通过 `name` 参数显式的给操作分配一个名称，那么 TensorFlow 将分配一个默认名称。有两种简单的方式可以查看到操作的名称，第一种方法是用 [TensorBoard](../../guide/graph_viz.md) 可视化计算图，另外一种方法是启用 [TensorFlow Debugger (tfdbg)](../../guide/debugger.md)。

接下来，通过给 `tensors` 参数传递 `tensor_to_log` 变量来创建 `LoggingTensorHook` 对象，并且设置 `every_n_iter` 的值为 50，每训练 50 步后在日志中输出概率。

### 训练模型

准备完成后，在 `main()` 函数中调用 `train_input_fn` 中的 `train()` 方法就可以训练我们的模型了： 

```python
# 模型训练
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])
```

在 numpy_input_fn 函数调用中，我们训练的特征数据和标注值分别传递给参数 `x` （字典类型）和 `y`。参数 `batch_size` 的值为 `100`（意味着模型每一步训练都会用到 100 个样本）。参数 `num_epochs=None` 指定训练迭代的次数。参数 `shuffle` 值为 True 表示训练时的样本是乱序的。在 `train` 调用中，`steps=20000` 表示模型总共会训练 20000 步。`hooks` 参数指定为 `logging_hook`，表示训练过程中会触发日志打印。

### 评估模型

训练完成后，我们可以调用 `evaluate` 方法来评价模型，它会根据我们定义在 `model_fn` 上的 `eval_metrics_ops` 的指标来评价模型在测试集上的准确度。

```python
# 评估模型并输出结果
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
```

在创建 `eval_input_fn` 时，我们设置 `num_epochs=1`，意味着迭代一次数据来得到模型的评价指标。我们同时也设置 `shuffle` 参数为 False 从而顺序的迭代数据。

### 运行模型

我们已经编写了 CNN 模型的函数，`Estimator`，以及训练/评价的逻辑；现在来运行 `cnn_mnist.py` 来看看结果。

> 注意：训练 CNNs 是一个计算密集型任务。`cnn_mnist.py` 的运行时长取决于你的处理器的性能，很有可能会耗费一个小时的时间来训练。当然为了加快训练的速度，你可以调低在 `train()` 函数中参数 `steps` 的取值，但注意这会影响到模型的准确性。

在模型训练过程中，你将会看到下面的输出日志：

```python
INFO:tensorflow:loss = 2.36026, step = 1
INFO:tensorflow:probabilities = [[ 0.07722801  0.08618255  0.09256398, ...]]
...
INFO:tensorflow:loss = 2.13119, step = 101
INFO:tensorflow:global_step/sec: 5.44132
...
INFO:tensorflow:Loss for final step: 0.553216.

INFO:tensorflow:Restored model from /tmp/mnist_convnet_model
INFO:tensorflow:Eval steps [0,inf) for training step 20000.
INFO:tensorflow:Input iterator is exhausted.
INFO:tensorflow:Saving evaluation summary for step 20000: accuracy = 0.9733, loss = 0.0902271
{'loss': 0.090227105, 'global_step': 20000, 'accuracy': 0.97329998}
```

在这里，我们最后在测试集上的准确度是 97.3%。

## 其他的资料

如果你想了解更多有关于 TensorFlow 中评估器（Estimators）和 CNNs 的内容，请查阅下面的资料：

*   [Creating Estimators in tf.estimator](../../guide/custom_estimators.md) 提供了一个 TensorFlow 估计器 API 的介绍。这篇教程将向你介绍如何配置一个估计器，编写一个模型函数，估计损失以及定义训练运算符。
*   [先进的卷积神经网络](../../tutorials/images/deep_cnn.md)则介绍了如何构建一个使用底层 TensorFlow 运算符而**不使用估计器**的 MNIST CNN 分类模型。
