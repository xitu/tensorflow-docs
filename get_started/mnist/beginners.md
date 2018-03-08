# 给机器学习初学者看的 MNIST

*本教程适用于在机器学习和 TensorFlow 方面是新手的读者。 如果您已经知道 MNIST 是什么，以及 softmax（多元逻辑）回归是什么，那么您可能更喜欢这个快节奏教程。 在开始任一教程之前，请确保安装了 TensorFlow。

当我们学习编程时，做的第一件事情通常是打印“Hello World”。就像编程有 Hello World 一样，机器学习有 MNIST。

MNIST 是一个简单的计算机视觉数据集。它由如下手写数字的图像组成：

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/MNIST.png">
</div>

它还包括每个图像的标签，告诉我们该图像是哪个数字。 例如，上述图像的标签是 5,0,4 和 1。

在本教程中，我们将训练一个模型来查看图像并预测它们上面是什么数字。我们的目标不是训练一个能够达到顶尖性能的十分精细的模型（尽管之后我们会提供这样的代码） - 而是简单看一下怎样使用 TensorFlow。 因此，我们将从一个非常简单的模型开始，称之为 Softmax 回归。

本教程的实际代码非常短，所有有趣的内容仅仅需要三行代码。然而，了解其背后的理念非常重要：包括 TensorFlow 的工作原理和机器学习的核心概念。因此，我们会非常仔细地过一遍代码。

## 关于这个教程

本教程会逐行解释 [mnist_softmax.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_softmax.py) 中的代码发了生什么。

您可以通过几种不同的方式使用本教程，其中包括：

- 通过阅读每行的解释，将每个代码段逐行复制并粘贴到 Python 环境中。

- 在阅读解释之前或之后运行整个`mnist_softmax.py` Python 文件，并使用本教程来理解不清楚的代码行。

我们将在本教程中完成的任务：

- 了解 MNIST 数据和 softmax 回归

- 基于查看图像中的每个像素，创建一个识别数字的模型

- 使用 TensorFlow 来训练模型，通过“看”数千个示例来识别数字（运行我们的第一个 TensorFlow 会话来完成）

- 用我们的测试数据检查模型的准确性

## MNIST 数据

MNIST 数据托管在[Yann LeCun 的网站](http://yann.lecun.com/exdb/mnist/)上。如果您正在从本教程中复制代码，请从这两行代码开始，这些代码将自动下载并读取数据：

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

MNIST 数据分为三个部分：55,000 个训练数据的数据集（`mnist.train`），10,000 个测试数据集（`mnist.test`）和 5,000 个验证数据点（`mnist.validation`）。这种划分是非常重要的：机器学习中必须具备我们不能从中学习的独立数据，这样我们才能确保我们学到的东西实际上是通用的。

如前所述，每个 MNIST 数据点有两部分：一个手写数字的图像和一个对应的标签。我们将用“x”代表图像，“y”代表标签。训练集和测试集都包含图像及其相应的标签; 例如训练图像是`mnist.train.images`，训练标签是`mnist.train.labels`。

每个图像是 28×28 像素。我们可以把它解释为一大堆数字：

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/MNIST-Matrix.png">
</div>

我们可以把这个数组展开成 28x28=784 个数字组成的向量。只要我们和原图像之间保持一致，那么如何展开这个数组并不重要。从这个角度来看，MNIST 图像只是 784 维向量空间中的一束点，具有[非常丰富的结构](https://colah.github.io/posts/2014-10-Visualizing-MNIST/)
（警告：计算密集的可视化）。

展平数据会丢弃有关图像二维结构的信息。这不好吗？嗯..最好的计算机视觉方法，以及我们在后面的教程中都需要利用这个二位结构。但是我们在这里使用的简单方法，也就是 softmax 回归（将在下面定义）不会利用这个结构。

所以， mnist.train.images 就是一个张量（一个n维数组），形式为`[55000,784]`。第一维是图像列表的索引，第二维是每个图像中每个像素的索引。张量中的每个条目是针对特定图像中的特定像素的介于0和1之间的像素强度。

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/mnist-train-xs.png">
</div>

MNIST 中每个图像都有相应的标签：一个 0 到 9 之间的数字，代表了图像中绘制的数字。。

为了本教程的目标，我们希望将我们的标签变成“独热向量”。“独热向量”是一个在大多数维度上为 0，在单维上为 1 的向量。在这个例子中，第 n 个数字将被表示为在第 n 维中为 1 的向量。例如，3 将是([0,0,0,1,0,0,0,0,0,0])。因此，mnist.train.labels 是一个`[55000, 10]`形式的浮点数数组。

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/mnist-train-ys.png">
</div>

现在我们准备好开始制作我们的模型了！

## Softmax 回归

我们知道 MNIST 中的每个图像都是一个 0 到 9 之间的手写数字。所以一个给定的图像只有十种可能的情况。我们希望模型能够查看一个图像，并给出它是每个数字的概率。例如，我们的模型可能会查看一张写着 9 的图片，并且 80％ 地确定这是一张 9，但是也给出一个 5％ 的概率认为它是一个8（因为数字 9 上部的圈），并且对所有其他情况也给出一点概率，因为它不是 100％ 确定。

这是一个经典的例子，在这里 softmax 回归是一个自然的，简单的模型。如果你想把一个对象取值的概率分配给几个不同的东西之一，softmax 回归就是要做的事情，因为 softmax 给了我们一个 0 到 1 之间的数值列表，并且他们加起来就是 1。甚至以后当我们训练更复杂的模型时，最后一步也会是一层 softmax 回归。

softmax 回归有两个步骤：第一步统计输入属于某些类别的证据，第二步将证据转换成概率。

为了收集给定图像在特定类别中的置信度，我们进行像素强度的加权求和。如果具有高强度的像素显示出该图像与某类别中的图像不一致，则权重是负的;如果显示出是一致的，则权重是正的。

下图显示了一个模型为每个类学习的权重。红色代表负面权重，蓝色代表正面权重。

<div style="width:40%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/softmax-weights.png">
</div>

我们也加入了一个额外的偏置量（bias），因为输入往往会带有一些无关的干扰量。因此对于给定的输入图片`x`它代表的是数字`i`的置信度可以表示为：

$$\text{evidence}_i = \sum_j W_{i,~ j} x_j + b_i$$

这里\\(W_i\\)代表权重， \\(b_i\\) 代表类别\\(i\\)的偏置量，  \\(j\\)代表输入图像 \\(x\\) 的用于按像素求和的索引，然后我们可以用 "softmax" 函数把这些置信度转换为预测的概率\\(y\\)

$$y = \text{softmax}(\text{evidence})$$

这里 softmax 起一个激励（activation）函数或者链接（link）函数的作用，把我们定义的线性函数的输出转换成我们想要的格式，在这个例子里是 10 个数字类的概率分布。你可以认为，对输入的一张图片，它对于每一个数字的吻合度被 softmax 函数转换成了一个概率值。softmax 函数被定义为：

$$\text{softmax}(evidence) = \text{normalize}(\exp(evidence))$$

如果你展开这个式子，会得到：

$$\text{softmax}(evidence)_i = \frac{\exp(evidence_i)}{\sum_j \exp(evidence_j)}$$

但是更多的时候按第一个公式理解 softmax 模型函数更有帮助：把输入值当成幂指数求值，再正则化这些结果值。这个幂运算表示，更多的置信度会成倍地增加模型里假设的权重值。反之，拥有更少的置信度意味着假设拥有更小的权重。假设的权值不可以是 0 值或者负值。之后 softmax 会正则化这些权重值，使它们的总和等于 1，以此构造一个有效的概率分布。（更多的关于 softmax 函数的信息，可以参考 Michael Nieslen 的书里面的[这一节](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)，其中有交互式的可视化解释。）

你可以用下面的图理解 softmax 回归模型，虽然有很多的\\(x\\)s，对于每一个输入，先计算出一个权重和，再分别加上一个偏置量，最后再应用 softmax 函数：

<div style="width:55%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/softmax-regression-scalargraph.png">
</div>

如果把它写成一个等式，我们可以得到：

<div style="width:52%; margin-left:25%; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/softmax-regression-scalarequation.png"
   alt="[y1, y2, y3] = softmax(W11*x1 + W12*x2 + W13*x3 + b1,  W21*x1 + W22*x2 + W23*x3 + b2,  W31*x1 + W32*x2 + W33*x3 + b3)">
</div>

我们也可以用向量表示这个计算过程，把它变成矩阵乘法和向量加法。这有助于提高计算效率。（也是一种更有效的思考方式）

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/softmax-regression-vectorequation.png"
 alt="[y1, y2, y3] = softmax([[W11, W12, W13], [W21, W22, W23], [W31, W32, W33]]*[x1, x2, x3] + [b1, b2, b3])">
</div>

更进一步，可以写成更加紧凑的方式：

$$y = \text{softmax}(Wx + b)$$

现在让我们把它转换成 TensorFlow 可以使用的东西。

## 实现回归

为了用 python 进行高效的数值计算，我们通常会使用 [NumPy](http://www.numpy.org) 这样的函数库，NumPy 会把类似矩阵乘法这样的复杂运算使用其他外部语言更高效率地实现。不幸的是，从外部计算切换回 Python 的每一个操作，仍然是一个很大的开销。如果你想用 GPU 来进行外部计算，这样的开销会更大。用分布式的计算方式，也会花费更多的资源用来传输数据。

TensorFlow 也把复杂的计算放在 Python 之外完成，但是为了避免那些开销，它做了进一步完善。Tensorflow 不单独地运行单一的复杂计算，而是让我们可以先用图描述一系列可交互的计算操作，然后全部一起在 Python 之外运行。（这样的运行方式可以在不少的机器学习库中看到。）

使用 TensorFlow 之前，首先需要导入它：

```python
import tensorflow as tf
```

我们通过操作符号变量来描述这些可交互的操作单元，用下面的方式创建一个：

```python
x = tf.placeholder(tf.float32, [None, 784])
```

`x`不是一个特定的值，而是一个占位符`placeholder`，我们在 TensorFlow 运行计算时输入这个值。我们希望能够输入任意数量的 MNIST 图像，每一张图展平成 784 维的向量。我们用 2 维的浮点数张量来表示这些图，这个张量的形状是`[None,784 ]`。（这里的 None 表示此张量的第一个维度可以是任何长度的。）

我们的模型也需要权重值和偏置量，当然我们可以把它们当做是另外的输入，但 TensorFlow 有一个更好的方法来表示它们：`Variable` 。 一个`Variable`代表一个可修改的张量，存在于 TensorFlow 的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。对于各种机器学习应用，一般都会有模型参数，可以用`Variable`表示。

```python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```

我们通过给 tf.Variable 初始值来创建 Variable：在这个例子里，我们都用全为零的张量来初始化 `W` 和 `b`。因为我们要学习`W`和`b`的值，它们的初值可以随意设置。

注意，`W`的维度是[784, 10]，因为我们想要用 784 维的图片向量乘以它以得到一个 10 维的置信度值向量，每一维对应不同数字类。`b`的形状是[10]，所以我们可以直接把它加到输出上面。

现在，我们可以实现我们的模型啦。只需要一行代码！

```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
```

首先，用 `tf.matmul(​​X，W)` 表示`x`乘以`W`，对应之前等式里面的，这里`x`是一个2维张量，拥有多个输入。然后再加上b，最后输入到`tf.nn.softmax`函数里面。

至此，我们先用了几行简短的代码来设置变量，然后只用了一行代码来定义我们的模型。TensorFlow 不仅仅可以使 softmax 回归模型计算变得特别简单，它也用这种非常灵活的方式来描述其他各种数值计算，从机器学习模型对物理学模拟仿真模型。一旦被定义好之后，我们的模型就可以在不同的设备上运行：你的电脑的CPU上，GPU上，甚至是手机上！


## 训练

为了训练我们的模型，我们需要为模型定义什么是好。那么实际上，在机器学习中，我们通常定义什么对于一个模型是坏的。我们称之为成本或损失，它表示我们的模型离我们期望的结果有多远。我们尽量减少这个损失，损失越小，我们的模型就越好。

一个来确定模型的损失的非常常见的，非常好的函数被称为“交叉熵”。交叉熵来源于信息论中信息压缩编码的思想，但后来它演变到从博弈论到机器学习等许多领域的一个重要的思想。它被定义为：

$$H_{y'}(y) = -\sum_i y'_i \log(y_i)$$

\\(y\\) 是我们预测的概率分布, \\(y'\\)是实际的分布（代表标签的独热向量)。比较粗糙的理解是，交叉熵是用来衡量我们的预测对于描述真相的低效性。更详细的关于交叉熵的解释超出本教程的范畴，但是你很有必要好好[理解它](https://colah.github.io/posts/2015-09-Visual-Information)。

为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：

```python
y_ = tf.placeholder(tf.float32, [None, 10])
```

然后我们可以用这个公式来计算交叉熵： \\(-\sum y'\log(y)\\):

```python
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```

首先，用 `tf.log` 计算 `y` 的每个元素的对数。接下来，我们把 `y_` 的每一个元素和 tf.log(y) 的对应元素相乘。最后，用 `tf.reduce_sum` 通过`reduction_indices=[1]` 参数来把`y`第二维的元素相加。最后，`tf.reduce_mean` 计算这组例子的平均值。

请注意，在源代码中，我们不使用这个公式，因为它在数值上是不稳定的。相反，我们应用`tf.nn.softmax_cross_entropy_with_logits`到没有正交化的 logits（例如，我们在`tf.matmul(x, W) + b`上应用`softmax_cross_entropy_with_logits`），因为这个数值上更稳定的函数内部计算了 softmax。在你的代码中，请考虑使用`tf.nn.softmax_cross_entropy_with_logits`。

现在我们知道我们需要我们的模型做什么了，用 TensorFlow 来训练它是非常容易的。因为 TensorFlow 拥有一张描述你各个计算单元的图，它可以自动地使用[反向传播算法](https://colah.github.io/posts/2015-08-Backprop)来有效地确定你的变量是如何影响你想要最小化的那个成本值的。然后，TensorFlow 会用你选择的优化算法来不断地修改变量以降低损失值。

```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

在这里，我们要求 TensorFlow 用[梯度下降算法](https://en.wikipedia.org/wiki/Gradient_descent) 以 0.5 的学习速率最小化交叉熵。梯度下降算法是一个简单的学习过程，TensorFlow 只需将每个变量一点点地往使成本不断降低的方向移动。当然 TensorFlow 也提供了其他许多优化算法，只要简单地调整一行代码就可以使用其他的算法。

TensorFlow 在这里实际上所做的是，它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少成本。

我们现在可以通过以下方式启动模型`InteractiveSession`：

```python
sess = tf.InteractiveSession()
```

我们首先必须创建一个操作来初始化我们创建的变量：

```python
tf.global_variables_initializer().run()
```


然后开始训练模型，这里我们让模型循环训练1000次！

```python
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

该循环的每个步骤中，我们都会随机抓取训练数据中的 100 个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行`train_step`。

使用一小部分的随机数据来进行训练被称为随机训练 - 在这里更确切的说是随机梯度下降训练。在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。


## 评估我们的模型

我们的模型性能如何呢？

首先让我们找出那些预测正确的标签。`tf.argmax`是一个非常有用的函数，它能给出某个张量对象在某一维上的其数据最大值所在的索引值。比如`tf.argmax(y,1)`返回的是模型对于任一输入预测到的标签值，而 `tf.argmax(y_,1)` 代表正确的标签，我们可以用 `tf.equal` 来检测我们的预测是否与真实标签匹配。

```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
```

这行代码会给我们一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。例如`·[True, False, True, True]` 会变成 `[1,0,1,1]` ，取平均值后得到 0.75.

```python
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

最后，我们计算所学习到的模型在测试数据集上面的正确率。

```python
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

这个最终结果值应该大约是92%

这个结果好吗？嗯，并不太好。事实上，这个结果是很差的。这是因为我们仅仅使用了一个非常简单的模型。不过，做一些小小的改进，我们就可以得到97％的正确率。最好的模型甚至可以获得超过99.7％的准确率！（想了解更多信息，可以看看这个关于各种模型的性能对比[列表](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results)。)

What matters is that we learned from this model. Still, if you're feeling a bit
down about these results, check out
@{$pros$the next tutorial} where we do a lot
better, and learn how to build more sophisticated models using TensorFlow!
比结果更重要的是，我们从这个模型中学习到的东西。不过，如果你仍然对这里的结果有点失望，可以查看下一个教程，在那里你可以学习如何用 TensorFlow 构建更加复杂的模型，并获得更好的性能！
