# TensorFlow 原理入门

教程源码下载：[tensorflow/examples/tutorials/mnist/](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/)

这篇文章的目的是为了展示如何使用 Tensorflow，并且用训练一个简单的以输入为导向来判断
手写数字的神经网络来举例说明。这个例子会使用 MNIST 来进行训练。这篇文章的是写给那些有
机械学习经验的用户们并对使用 Tensorflow 有兴趣的人而设计的。

注意: 这篇文章意图并不是教读者如何做机械学习的开发的。

在阅读时，请务必配合 @{$install$install TensorFlow} 中的教程。

## 教程文件

这篇教程的需要以下这些文件

文件名 | 用途
--- | ---
[`mnist.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist.py) | The code to build a fully-connected MNIST model.
[`fully_connected_feed.py`](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/fully_connected_feed.py) | The main code to train the built MNIST model against the downloaded dataset using a feed dictionary.

我们可以通过直接运行 `fully_connected_feed.py` 以开始我们的训练：

```bash
python fully_connected_feed.py
```

## 准备数据

MINST 是一个在机械学习问题中非常典型的一个数据集。该案例是通过分析一个 28 ＊ 28 像素点的
手写数字图像来区分该图像是属于 0 到 9 中那一个数字的。

![MNIST Digits](https://www.tensorflow.org/images/mnist_digits.png "MNIST Digits")

关于更多信息，可以访问 [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/)
或者 [Chris Olah's visualizations of MNIST](http://colah.github.io/posts/2014-10-Visualizing-MNIST/) 这两个页面。

### 下载

在 `run_training()` 这个 function 的开头，`input_data.read_data_sets()`这个 funtion
会确保你已经成功的在本地下载了 MINST 的数据集，并且解压该数据集然后返回一个叫做 `DataSet` 
的实例。

```python
data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)
```

**注意**: `fake_data` 这个 flag 是用于单元测试的，如果你在读这篇文章的话你应该不会用到。

数据集 | 用途
--- | ---
`data_sets.train` | 55000 个图片个对应标签，用于主要训练。
`data_sets.validation` | 5000 个图片和对应标签，用于迭代性的调整模型以提高正确率。
`data_sets.test` | 10000 个图片和标签，用于最终的模型正确率测试。

### 输入值和占位符

`placeholder_inputs()` 这个 function 用于创建 2 个定义输入值形状的变量 @{tf.placeholder}，
其中包括 `batch_size` 用于描述剩余图像，以及用于描述真正的会被用于训练的图像。

```python
images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                       mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
```

此外，在训练时，整个图像和标签的数据集在每一步中会根据 `batch_size`  以及对应的占位符被
分成多份，并且通过 `feed_dict` 作为参数传入 `sess.run()` 这个function中。

## 建立图像

在数组中创立占位符之后, `mnist.py` 这个文件会根据3种模式来生成图像. 它们分别是:

1.  `inference()` - 通过神经网络生成初级的预测模型 
2.  `loss()` - 在基础预测模型的基础上加上噪点的影响
3.  `training()` - 在加上噪点的影响之后再应用梯度下降的算法提高精准性

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/mnist_subgraph.png">
</div>

### 推理

`inference()` 这个 function 可以根据张量（tensor）来生成最初包含输出结果的预测模型。

它需要使用图像的占位符来作为输入，然后在这个基础上使用 [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) 
来激活并用十个节点的 logits 来构建一层完全连接层。
It takes the images placeholder as input and builds on top
of it a pair of fully connected layers with [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) 
activation followed by a ten
node linear layer specifying the output logits.

每一层都创建于一个唯一的 @{tf.name_scope} 之下，创建于该作用域之下的所有元素都将带有其前缀。

```python
with tf.name_scope('hidden1'):
```

在定义域中，每一层所使用的权重和偏差都会在带有期望形状下在 @{tf.Variable} 中生成。


```python
weights = tf.Variable(
    tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                        stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
    name='weights')
biases = tf.Variable(tf.zeros([hidden1_units]),
                     name='biases')
```

举个例子，当这些层是在 `hidden1` 这个作用域之下生成的时候，这个权重变量的名字就会变成 "`hidden1/weights`" 。

每个变量在生成的时候都会执行初始化操作。

在这种最常见的情况下，通过 @{tf.truncated_normal} 函数初始化权重变量，给赋予的 shape 则是一个二维 tensor，
其中第一个维度代表该层中权重变量所连接（connect from）的单元数量，第二个维度代表该层中权重变量所连接到的（connect to）单元数量。
对于名叫 hidden1 的第一层，相应的维度则是 `[IMAGE_PIXELS, hidden1_units]`，因为权重变量将图像输入连接到了 hidden1 层。
tf.truncated_normal 初始函数会根据所得到的均值和标准差，生成一个随机分布。

初始偏差量则是由 @{tf.zeros} 这个函数来生成以确保所有初始值都是 0。而它们的形状则直接是其在该层中所接到的单元数量。

图表的有三个主要操作，分别是两个 @{tf.nn.relu} 操作 和一个 @{tf.matmul}。
前面两个中嵌入了隐藏层所需的 `tf.matmul` 。第三个会用于生成 logits 模型。
三者依次生成，各自的 tf.Variable 实例则与输入占位符或下一层的输出 tensor 所连接。

```python
hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
```

```python
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
```

```python
logits = tf.matmul(hidden2, weights) + biases
```

最终，包含了 `logits` 的 tensor 会作为结果被返回。

### 损失

`loss()` 这个函数会添加所需的损失操作来进一步生成图表。

首先，从 `labels_placeholder` 内取得的值会被转换成一个 64-bit 的整数。然后，一个 @{tf.nn.sparse_softmax_cross_entropy_with_logits} 
会从 `labels_placeholder` 自动产生一个 1-hot 标签， 并且比较 `inference()` 函数与 1-hot 标签所输出的 logits Tensor。

```python
labels = tf.to_int64(labels)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels, logits=logits, name='xentropy')
```

然后使用 @{tf.reduce_mean} 函数来计算 batch 维度（第一维度）下交叉熵（cross entropy）的平均值，将将该值作为总损失。

```python
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
```

最后，程序会返回包含了损失值的Tensor。

> 注意：交叉熵是信息理论中的概念，可以让我们描述如果基于已有事实，相信神经网络所做的推测最坏会导致什么结果。
> 更多详情，请查阅博文[《可视化信息理论》](http://colah.github.io/posts/2015-09-Visual-Information/)

### 训练

`training()` 这个函数会通过用[梯度下降算法](https://en.wikipedia.org/wiki/Gradient_descent)执行各种操作来减小损失。

首先，该函数从 loss() 函数中获取损失的 Tensor ，将其交给 @{tf.summary.scalar}，后者在与 SummaryWriter（见下文）配合使用时，
可以向事件文件（events file）中生成汇总值（summary values）。在本篇教程中，每次写入汇总值时，它都会释放损失 Tensor 的当前值（snapshot value）。

```python
tf.summary.scalar('loss', loss)
```

接下来，我们实例化一个 @{tf.train.GradientDescentOptimizer}，负责按照所要求的学习效率（learning rate）应用梯度下降法（gradients）。

```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
```

之后，我们生成一个变量用于保存全局训练步骤（global training step）的数值，并使用 @{tf.train.Optimizer.minimize} 函数更新系统中的三角权重（triangle weights）、
增加全局步骤的操作。根据惯例，这个操作被称为 train_op，是 TensorFlow 会话（session）诱发一个完整训练步骤所必须运行的操作（见下文）。

```python
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
```

## 训练模型

一旦图表构建完毕，就通过 fully_connected_feed.py 文件中的用户代码进行循环地迭代式训练和评估。

### 图表

在 `run_training()` 这个函数的开始是一个 Python 语言中的 with 命令，
这个命令表明所有已经构建的操作都要与默认的 @{tf.Graph} 全局实例关联起来。

```python
with tf.Graph().as_default():
```

`tf.Graph` 实例是一系列可以作为整体执行的操作。TensorFlow 的大部分场景只需要依赖默认图表一个实例即可。

利用多个图表的更加复杂的使用场景也是可能的，但是超出了本教程的范围。

### 会话

一旦所有的构建准备完成之后，我们可以创建一个 @{tf.Session} 用于运行图表。

```python
sess = tf.Session()
```

另外一个选项就是用 `with` 来当前作用域下生成会话:

```python
with tf.Session() as sess:
```
当 `session` 函数中没有传入参数当情况下，默认该代码将会依附于（如果还没有创建会话，则会创建新的会话）默认的本地会话。

在生成会话之后，所有 `tf.Variable` 实例都会立即通过调用各自初始化操作中的 sess.run() 函数进行初始化。

```python
init = tf.global_variables_initializer()
sess.run(init)
```

 @{tf.Session.run} 方法将会运行图表中与作为参数传入的操作相对应的完整子集。在初次调用时，init 操作只包含了变量初始化
 程序 @{tf.group}。图表的其他部分不会在这里，而是在下面的训练循环运行。



### 训练循环

完成会话中变量的初始化之后，就可以开始训练了。

训练的每一步都是通过用户代码控制，而能实现有效训练的最简单循环就是：

```python
for step in xrange(FLAGS.max_steps):
    sess.run(train_op)
```

但是，本教程中的例子要更为复杂一点，原因是我们必须把输入的数据根据每一步的情况进行切分，以匹配之前生成的占位符。

#### 向图表提供反馈

执行每一步时，我们的代码会生成一个反馈字典，其中包含对应步骤中训练所要使用的例子，这些例子的哈希键就是其所代表的占位符操作。

`fill_feed_dict()` 函数会查询给定的 DataSet，索要下一批次batch_size 的图像和标签，
与占位符相匹配的 Tensor 则会包含下一批次的图像和标签。

```python
images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                               FLAGS.fake_data)
```

然后，以占位符为哈希键，创建一个 Python 字典对象，键值则是其代表的反馈 Tensor。

```python
feed_dict = {
    images_placeholder: images_feed,
    labels_placeholder: labels_feed,
}
```

这个字典之后会作为参数传入 `sess.run()` 这个函数，为这一步的训练提供输入样例。


#### 检查状态

在运行 sess.run 函数时，要在代码中明确其需要获取的两个值：`[train_op, loss]`。

```python
for step in xrange(FLAGS.max_steps):
    feed_dict = fill_feed_dict(data_sets.train,
                               images_placeholder,
                               labels_placeholder)
    _, loss_value = sess.run([train_op, loss],
                             feed_dict=feed_dict)
```

因为要获取这两个值，`sess.run()`会返回一个有两个元素的元组。其中每一个 Tensor 对象，对应了返回的元组中的 numpy 数组，
而这些数组中包含了当前这步训练中对应 Tensor 的值。由于 train_op 并不会产生输出，其在返回的元祖中的对应元素就是 None，
所以会被抛弃。但是，如果模型在训练中出现偏差，loss Tensor 的值可能会变成 NaN，所以我们要获取它的值，并记录下来。

假设训练一切正常，没有出现NaN，训练循环会每隔100个训练步骤，就打印一行简单的状态文本，告知用户当前的训练状态。

```python
if step % 100 == 0:
    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
```

#### 状态可视化

为了释放 @{$summaries_and_tensorboard$TensorBoard} 所使用的事件文件，所有的即时数据（在这里只有一个）都要在图表构建阶段合并至一个操作中。


```python
summary = tf.summary.merge_all()
```

在创建好会话（session）之后，可以实例化一个 @{tf.summary.FileWriter} ，用于写入包含了图表本身和即时数据具体值的事件文件。

```python
summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
```

最终，每次运行 summary 操作时，都会往事件文件中写入最新的即时数据，函数的输出会传入事件文件读写器的 `add_summary()`函数。。

```python
summary_str = sess.run(summary, feed_dict=feed_dict)
summary_writer.add_summary(summary_str, step)
```

事件文件写入完毕之后，可以就训练文件夹打开一个 TensorBoard，查看即时数据的情况。

![MNIST TensorBoard](https://www.tensorflow.org/images/mnist_tensorboard.png "MNIST TensorBoard")

**NOTE**：了解更多如何构建并运行 TensorBoard 的信息，请查看相关教程 Tensorboard：@{$summaries_and_tensorboard$Tensorboard: Visualizing Learning} 。

#### 保存检查点

为了得到可以用来后续恢复模型以进一步训练或评估的检查点文件，我们会实例化一个 @{tf.train.Saver}。

```python
saver = tf.train.Saver()
```

在训练循环中，将定期调用 @{tf.train.Saver.save} 方法，向训练文件夹中写入包含了当前所有可训练变量值得检查点文件。

```python
saver.save(sess, FLAGS.train_dir, global_step=step)
```

之后，训练就可以在使用 @{tf.train.Saver.restore} 的情况下多次进行来重载模型.

```python
saver.restore(sess, FLAGS.train_dir)
```

## 评估模型

每隔一千个训练步骤，我们的代码会尝试使用训练数据集与测试数据集，对模型进行评估。
`do_eval` 函数会被调用三次，分别使用训练数据集、验证数据集合测试数据集。

```python
print('Training Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.train)
print('Validation Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.validation)
print('Test Data Eval:')
do_eval(sess,
        eval_correct,
        images_placeholder,
        labels_placeholder,
        data_sets.test)
```

> 注意，更复杂的使用场景通常是，先隔绝 `data_sets.test` 测试数据集，
> 只有在大量的超参数优化调整（hyperparameter tuning）之后才进行检查。
> 但是，由于 MNIST 问题比较简单，我们在这里一次性评估所有的数据。

### 构建评估图表

在进入训练循环之前，我们应该先调用 mnist.py 文件中的 evaluation 函数，传入的 logits 和标签参数要与 loss 函数的一致。
这样做事为了先构建评估操作。

```python
eval_correct = mnist.evaluation(logits, labels_placeholder)
```

函数会生成 @{tf.nn.in_top_k} 操作，如果在 K 个最有可能的预测中可以发现真的标签，那么这个操作就会将模型输出标记为正确。
在本文中，我们把 K 的值设置为 1，也就是只有在预测是真的标签时，才判定它是正确的。


```python
eval_correct = tf.nn.in_top_k(logits, labels, 1)
```

### 评估图表输出

之后，我们可以创建一个循环，往其中添加 `feed_dict` ，并在调用 `sess.run()` 函数时传入 `eval_correct` 操作，目的就是用给定的数据集评估模型。

```python
for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
```

The `true_count` 变量会累加所有 `in_top_k`操作判定为正确的预测之和。
接下来，只需要将正确测试的总数，除以例子总数，就可以得出准确率了。

```python
precision = true_count / num_examples
print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
      (num_examples, true_count, precision))
```
