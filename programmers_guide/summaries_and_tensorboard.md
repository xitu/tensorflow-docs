# Tensorboard：可视化学习面板

Tensorflow 做的一些计算是复杂和混乱，就像训练深度神经网络。为了更易于理解、调试和优化 Tensorflow 程序，我们发布了一套可视化工具称为 Tensorboard。你可以使用 TensorBoard 图形化 Tensorflow 图，绘制图计算过程的定量指标，以及显示图的额外数据。TensorBoard 完成配置后，是这样的：

![](https://camo.githubusercontent.com/f0f03739a6b2a0e312f929759fab857856b7cf0c/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f6d6e6973745f74656e736f72626f6172642e706e67)

这个只有 30 分钟的教程旨在让你可以通过简单的 TensorBoard 用法入门。我们假设你对于 TensorFlow 已经有了一定的基础。

当然还有其他可用的资源！[TensorBoard GitHub](https://github.com/tensorflow/tensorboard) 有更多关于在 TensorBoard 中使用单个面板所需要的信息，包括提示、技巧和调试信息。

## 建立（Setup）

[安装 TensorFlow](../install/)。通过 pip 方式安装 TensorFlow 时，会自动安装 TensorBoard。

## 数据序列化

TensorBoard 是通过读取 Tensorflow 事件文件来操作的，Tensorflow 事件包含 Tensorflow 运行所生成的汇总数据。下面是 TensorBoard 汇总数据的完整生命周期。

首先，创建用于收集汇总数据的 TensorFlow 图，并决定你想哪种节点来记录[操作摘要](../api_guides/python/summary.md)。例如，假设你正在训练用于识别 MNIST 数字的卷积神经网络。你希望记录学习速率是如何随时间变化的，以及目标函数是如何变化的。通过将 `tf.summary.scalar` 操作附加在输出学习速率和损耗的节点来收集这些信息。然后，给每一个 scalar_summary 标记一个有意义的标签，比如 `learning rate` 或 `loss function`。

也许你也希望可视化来自特定层的激活分布，或梯度或权重的分布。收集这些分布数据，可以在梯度输出值和权重变量上通过执行 `tf.summary.histogram` 操作来得到。你可以查阅[操作摘要](../api_guides/python/summary.md)文档来找到所有可用的汇总d的操作信息。Tensorflow 上的操作不会做任何事情直到你运行它们，或一个运算取决于这些操作的输出。我们刚刚创建的汇总节点对于你的图来说是次要的：目前没有一个操作是依赖与它们的。因此，为了生成汇总数据，我们需要运行所有这些汇总节点。手工管理他们是乏味的，所以用 `tf.summary.merge_all` 来将它们组合成一个简单的运算，生成汇总数据。

然后，你就可以运行合并汇总命令，它会依据特定步骤将所有数据生成一个序列化的汇总 protobuf 对象。最后，通过汇总 protobuf 对象传递给 `tf.summary.FileWriter`，可以将汇总数据写到磁盘上。

FileWriter 的构造函数需要包含 logdir，logdir 目录是非常重要的，所有的事件都会写到它所指的目录下。另外，FileWrite 可以方便地携带一个图对象在它的构造函数中。如果它接收到一个图对象，Tensorboard 会将张量形状信息和你的图像一并显示出来。这会让你更加直观地感受到图的生产过程：查看[Tensor 形状信息](../guide/graph_viz.md#tensor-shape-information)。

现在已经修改了你的图并且有一个 FileWriter，准备开始你的神经网络吧！如果你愿意，你可以单步运行合并汇总操作，并记录大量的训练数据。不过，这可能比你需要的数据更多。你可以每 n 步执行一次汇总。

下面的代码示例是基于[简单 MNIST 教程](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py)，更改的，其中我们额外每十步执行一次总结。如果你运行这个然后启动 Tensorboard—logdir=/tmp/tensorflow/mnist，你将能够可视化统计数据，例如在训练过程中权重或精确度是如何变化的。下面的代码是一个摘录，完整的资料在[这里](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)。

```python
def variable_summaries(var):
  """为了 TensorBoard 可视化，给Tensor添加一些汇总"""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """用于生成简单的神经网络层的可复用代码。
  它进行矩阵乘法，偏置加法，然后使用 relu 进行非线性化。
  它还设置了名称范围，使得生成的图形易于阅读，
  并增加了一些汇总操作。
  """
  # 添加一个名称范围以确保图层的逻辑分组。
  with tf.name_scope(layer_name):
  # 这个变量将保存图层权重的状态
  with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
  keep_prob = tf.placeholder(tf.float32)
  tf.summary.scalar('dropout_keep_probability', keep_prob)
  dropped = tf.nn.dropout(hidden1, keep_prob)

  # 不要使用 softmax 激活，请参阅下文。
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
  # 交叉熵的原始公式,
  #
  # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),reduction_indices=[1]))
  #
  # 可能在数值上不稳定。
  #
  # 所以这里，我们使用 tf.losses.sparse_softmax_cross_entropy 处理前面 nn_layer 输出的原始 logit
  with tf.name_scope('total'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
  train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
      cross_entropy)

with tf.name_scope('accuracy'):
  with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

  # 合并所有的汇总信息，并把它们写到 /tmp/mnist_logs（ 默认路径 ）
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')
tf.global_variables_initializer().run()

```

在初始化 FileWriters 后，必须要添加汇总到 FileWriters 中作为我们训练和测试的模型。

```python
  # 训练模型，并写入汇总信息。
  # 每 10 步，测量一次测试集的准确度，并写出测试汇总信息
  # 所有其他步骤，在训练数据上运行 train_step，并添加训练汇总

def feed_dict(train):
  """做一个 TensorFlow feed_dict：将数据映射到张量占位符上。"""
  if train or FLAGS.fake_data:
    xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
    k = FLAGS.dropout
  else:
    xs, ys = mnist.test.images, mnist.test.labels
    k = 1.0
  return {x: xs, y_: ys, keep_prob: k}

for i in range(FLAGS.max_steps):
  if i % 10 == 0:  # 记录汇总和测试集精度
    summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
    test_writer.add_summary(summary, i)
    print('Accuracy at step %s: %s' % (i, acc))
  else:  # 记录训练得到的汇总并且训练
    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
    train_writer.add_summary(summary, i)

```

## 启动 TensorBoard

运行下面命令(二者选一)，运行 TensorBoard。

```python
tensorboard —logdir=path/to/log-directory

（python -m tensorboard.main）

```

logdir 就是 FileWriter 序列化数据的目录。如果 logdir 目录包含子目录，并且子目录具有来自单独线程的序列化数据，那么 Tensorboard 将统一展示这些可视化数据。一旦 TensorBoard 运行，你可以通过你的 Web 浏览器 localhost:6006，查看 Tensorboard。你会在 Tensorboard 右上角看到导航标签。每个选项代表一组可以可视化的序列化数据。

关于如何使用 **“graph”** 来显示你的图的详细信息，你可以查看 [TensorBoard：图表可视化](../guide/graph_viz.md)。

更多关于 TensorBoard 的信息，请通过 [TensorBoard 的 GitHub](https://github.com/tensorflow/tensorboard) 地址查看。
