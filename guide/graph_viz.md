# TensorBoard: 图形可视化

TensorFlow 的计算图功能强大但却复杂。而图表可视化功能可以帮助您了解和调试它们。以下是一个可视化工作的例子。

![Visualization of a TensorFlow graph](https://www.tensorflow.org/images/graph_vis_animation.gif "Visualization of a TensorFlow graph")
**Tensorflow 图形可视化**

要查看您自己的图形，请运行 TensorBoard 并将其指向工作的日志目录，单击顶部窗格上的图形选项卡，然后使用左上角的菜单选择相关的运行。如果想获得有关于如何运行 TensorBoard 并且确保记录了所有必要信息的更多信息，请参阅 [TensorBoard：可视化学习](../guide/summaries_and_tensorboard.md)。

## 命名范围和节点

典型的 TensorFlow 图可能有成千上万的节点——太多了，很难一次看到，甚至无法使用标准图形工具进行布局。为方便起见，变量名可以作用于域，可视化使用这些信息来定义图中节点上的层次结构。默认情况下，只显示该层次结构的顶部。以下是在一个在 `hidden` 名字域下使用 `tf.name_scope` 名称的范围定义三个操作的示例 ：

```python
import tensorflow as tf

with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')
```

这导致了以下三个 op 的名称：

* `hidden/alpha`
* `hidden/weights`
* `hidden/biases`

在默认情况下，可视化将把所有三个 op 都折叠成标记为 `hidden` 的节点。至于额外的细节是不会丢失的。您可以双击，或者点击右上角橙色 `+` 号展开节点，然后你便会看到三个子节点 `alpha`，`weights` 和 `biases`。

以下是一个更复杂的节点在其初始状态和扩展状态的真实例子。

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="https://www.tensorflow.org/images/pool1_collapsed.png" alt="Unexpanded name scope" title="Unexpanded name scope" />
    </td>
    <td style="width: 50%;">
      <img src="https://www.tensorflow.org/images/pool1_expanded.png" alt="Expanded name scope" title="Expanded name scope" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      顶级名称域  <code>pool_1</code> 的初始视图。点击右上角橙色 + 号按钮或双击节点本身将会展开它。
    </td>
    <td style="width: 50%;">
      名称域 pool_1 的扩展视图。点击右上角的橙色 - 按钮或双击节点本身将折叠名称范围。
    </td>
  </tr>
</table>

按名称域对节点进行分组的方法对于制作清晰的图形至关重要。如果您正在构建模型，则名称域可以让你更好的控制生成的可视化图形。**你命名的名字域越好，你的可视化图形就越好。**

上图说明了可视化的第二个方面。TensorFlow 图有两种连接方式：数据相关性和控制相关性。数据相关性显示两个操作符之间的张量流，并显示为实线箭头，而控制相关性使用虚线。在展开视图（上面两幅图中的右侧图片）除了连接 `CheckNumerics` 和 `control_dependency` 的虚线外，所有连接都是数据依赖关系。

这儿还有一个简化布局的技巧。大多数 TensorFlow 图有几个与其他节点连接的节点。比如说，许多节点可能对初始化步骤具有控制相关性。绘制 init 节点及其相关关系之间的所有边将创建一个非常混乱的视图。

为了减少混乱，可视化将所有高位节点分隔到右侧的辅助区域，并通过不画线来表示其边缘。我们绘制小节点图标来代替连线。分离辅助节点通常不会去除关键的信息，因为这些节点通常与记账功能有关。请参阅 [Interaction](#interaction) 来了解如何在主图和辅助区之间移动节点。

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="https://www.tensorflow.org/images/conv_1.png" alt="conv_1 is part of the main graph" title="conv_1 is part of the main graph" />
    </td>
    <td style="width: 50%;">
      <img src="https://www.tensorflow.org/images/save.png" alt="save is extracted as auxiliary node" title="save is extracted as auxiliary node" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      结点 <code>conv_1</code> 连接到 <code>save</code>。请注意结点 <code>save</code> 右侧的小节点图标。
    </td>
    <td style="width: 50%;">
      <code>save</code> 具有很高的等级，并会出现作为辅助节点。连接 <code>conv_1</code> 显示为其左侧的节点图标。为了进一步减少混乱，由于 <code>save</code> 有很多的连接，我们显示前 5 个并缩写其他为 <code>... 12 more</code>.
    </td>
  </tr>
</table>

最后一个结构简化方法是**系列崩溃**。连续图案——也就是说，名称相差最后一个数字并具有同构结构的节点——会折叠成**栈**节点，就如下图所示。对于拥有长序列的网络结构，这大大简化了视图。与分层节点一样，双击将扩展该部分。请参阅   [交互](https://github.com/xitu/tensorflow/blob/zh-hans/tensorflow/docs_src/get_started/graph_viz.md#interaction)以了解如何为特定节点集禁用/启用折叠。

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="https://www.tensorflow.org/images/series.png" alt="Sequence of nodes" title="Sequence of nodes" />
    </td>
    <td style="width: 50%;">
      <img src="https://www.tensorflow.org/images/series_expanded.png" alt="Expanded sequence of nodes" title="Expanded sequence of nodes" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      节点序列折叠后的视图。
    </td>
    <td style="width: 50%;">
      双击后展开的一小块视图。
    </td>
  </tr>
</table>

最后，作为对于易读性的最后一个帮助，可视化将对于常量和总节点使用特殊的图标。在此总结一下，下面是一个节点的符号表：

| Symbol                                   | Meaning                    |
| ---------------------------------------- | -------------------------- |
| ![Name scope](https://www.tensorflow.org/images/namespace_node.png "Name scope") | 表示名称域的*高级*节点。鼠标双击用以展开高级节点。 |
| ![Sequence of unconnected nodes](https://www.tensorflow.org/images/horizontal_stack.png "Sequence of unconnected nodes") | 没有相互连接的编号节点序列。             |
| ![Sequence of connected nodes](https://www.tensorflow.org/images/vertical_stack.png "Sequence of connected nodes") | 相互连接的编号节点序列。               |
| ![Operation node](https://www.tensorflow.org/images/op_node.png "Operation node") | 一个单独的操作节点。                 |
| ![Constant node](https://www.tensorflow.org/images/constant.png "Constant node") | 一个常数。                      |
| ![Summary node](https://www.tensorflow.org/images/summary.png "Summary node") | 摘要节点。                      |
| ![Data flow edge](https://www.tensorflow.org/images/dataflow_edge.png "Data flow edge") | 显示互相操作过程中的数据流。             |
| ![Control dependency edge](https://www.tensorflow.org/images/control_edge.png "Control dependency edge") | 显示互相操作过程中的控制相关性。           |
| ![Reference edge](https://www.tensorflow.org/images/reference_edge.png "Reference edge") | 显示传出操作节点可以改变传入张量。          |

## 互动 {#interaction}

通过平移和缩放导航图形。可以通过点击并拖动来平移，并使用滚动手势进行缩放。双击某个节点，或者单击其 `+` 按钮，展开代表一组操作的名称域。为了在放大和平移时轻松跟踪当前视点，右下角会有一个小地图。

要关闭打开的节点，请再次双击它或单击其 `-` 按钮。您也可以单击一次来选择一个节点。它会变成一个较深的颜色，并且关于它的详细信息以及它所连接的节点将出现在可视化对象右上角的信息卡中。

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="https://www.tensorflow.org/images/infocard.png" alt="Info card of a name scope" title="Info card of a name scope" />
    </td>
    <td style="width: 50%;">
      <img src="https://www.tensorflow.org/images/infocard_op.png" alt="Info card of operation node" title="Info card of operation node" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      信息卡会显示 <code>conv2</code> 名称域的详细信息。输入和输出通过名称域内操作节点的输入和输出，组合而成。对于名称域，不显示任何属性。
    </td>
    <td style="width: 50%;">
      信息卡会显示 <code>DecodeRaw</code> 操作节点的详细信息。除输入和输出外，卡片还显示设备与当前操作相关的属性。
    </td>
  </tr>
</table>

TensorBoard 提供了几种方法来改变图形的视觉布局。这不会改变图的计算语义，但是它可以让网络的结构更加清晰。通过右键单击某个节点或按该节点信息卡底部的按钮，可以对其布局进行以下更改：

- 节点可以在主图表和辅助区域之间移动。
- 可以将一组节点取消分组，使得该组的节点不会出现在一起。未分组的节点也可以重新组合。

选择也可以帮助理解高级节点。选择任何高度节点，其他连接的相应节点图标也将被选中。例如，这可以很容易地看到哪些节点正在保存 —— 哪些不是。

点击信息卡中的节点名称将选择它。如有必要，视点将自动平移，以便于节点可见。

最后，您可以使用图例上方的颜色菜单为图形选择两种配色方案。默认的 **结构视图** 会显示以下结构：当两个高级节点具有相同的结构时，它们以彩虹的相同颜色出现。结构独特的节点是灰色的。然后还有第二个视图，它显示了不同操作运行的设备。名称域与其内部操作的设备比例成比例。

下面的图片我们给出了一张真实生活图的插图。

<table width="100%;">
  <tr>
    <td style="width: 50%;">
      <img src="https://www.tensorflow.org/images/colorby_structure.png" alt="Color by structure" title="Color by structure" />
    </td>
    <td style="width: 50%;">
      <img src="https://www.tensorflow.org/images/colorby_device.png" alt="Color by device" title="Color by device" />
    </td>
  </tr>
  <tr>
    <td style="width: 50%;">
      结构视图：灰色节点具有独特的结构。橙色的 <code>conv1</code> 和 <code>conv2</code> 节点具有相同的结构，至于其他颜色的节点类似。
    </td>
    <td style="width: 50%;">
      设备视图：名称域与其内部操作节点的设备比例成比例。在这里，紫色是指 GPU，而绿色是 CPU。
    </td>
  </tr>
</table>

## 张量的形状信息

当序列化 `GraphDef` 包括张量的形状时，图形可视化工具用张量维度标注边缘，边缘厚度反映总张量的大小。在 `GraphDef` 的传递中包含张量形状的实际图形对象（如在 `sess.graph` ）到 `FileWriter` 序列化图形的时候。下面的图片就显示了具有张量形状信息的 CIFAR-10 模型：
<table width="100%;">
  <tr>

    <td style="width: 100%;">
      <img src="https://www.tensorflow.org/images/tensor_shapes.png" alt="CIFAR-10 model with tensor shape information" title="CIFAR-10 model with tensor shape information" />
    </td>
  </tr>
  <tr>
    <td style="width: 100%;">
      张量形状信息的 CIFAR-10 模型。
    </td>
  </tr>
</table>

## 运行时间统计

通常收集运行时的元数据是非常有用的，例如总内存使用量，总计算时间和节点的张量形状。下面的代码示例是修改自 [Estimators MNIST 教程](../tutorials/estimators/cnn.md) 中的一个片段，其中我们记录了摘要法和运行时统计的信息。有关如何记录摘要的详细信息，请参阅[摘要教程](../guide/summaries_and_tensorboard.md#serializing-the-data)。完整的源代码在[这里](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py)。

```python
  # 训练模型，并记录日志
  # 每 10 步，评估测试数据集准确性，并记录测试日志
  # 所有其他步骤中，在训练集上运行训练步骤，并记录训练日志

  def feed_dict(train):
    """创建一个 TensorFlow feed_dict：将数据映射到 Tensor 占位符上。"""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # 记录日志
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
```

此代码将从步骤 99 开始每 100 步发出运行时统计信息。

当启动 tensorboard 并转到图表选项卡时，您现在将在“会话运行”下看到与添加运行元数据的步骤和与其相对应的选项。选择其中一个运行过程将显示在该步骤的网络快照，淡出未使用的节点。在左侧的控件中，您可以通过总内存或总计算时间对节点着色。此外，单击节点将显示确切的总内存，计算时间和张量输出大小。


<table width="100%;">
  <tr style="height: 380px">
    <td>
      <img src="https://www.tensorflow.org/images/colorby_compute_time.png" alt="Color by compute time" title="Color by compute time"/>
    </td>
    <td>
      <img src="https://www.tensorflow.org/images/run_metadata_graph.png" alt="Run metadata graph" title="Run metadata graph" />
    </td>
    <td>
      <img src="https://www.tensorflow.org/images/run_metadata_infocard.png" alt="Run metadata info card" title="Run metadata info card" />
    </td>
  </tr>
</table> 
