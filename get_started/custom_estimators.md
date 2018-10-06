
# 创建定制化 Estimator

本文档介绍定制化 Estimator。特别是，本文档将演示如何创建定制化 `tf.estimator.Estimator`，`tf.estimator.DNNClassifier` 在解决虹膜问题的行为中，它可以模拟预制的 Estimator。有关虹膜问题的详细信息，请参阅 [Pre-Made Estimators 章节](../guide/premade_estimators.md)。

下载及访问示例代码，请调用以下两个命令：

```shell
git clone https://github.com/tensorflow/models/
cd models/samples/core/get_started
```

在本文档中，我们将查看 [`custom_estimator.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py)。您可以使用以下命令运行它：

```bsh
python custom_estimator.py
```

如果你感到不耐烦，可随时将 [`custom_estimator.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py) 与 [`premade_estimator.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/premade_estimator.py) 进行比较（对比），它们在同一目录中。

## 预制 vs. 定制化

如下图所示，预制 Estimator 是 `tf.estimator.Estimator` 基类的子类，而定制化 Estimators 则是 tf.estimator.Estimator 的实例：

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="Premade estimators are sub-classes of `Estimator`. Custom Estimators are usually (direct) instances of `Estimator`"
  src="https://www.tensorflow.org/images/custom_estimators/estimator_types.png">
</div>
<div style="text-align: center">
预制和定制化 Estimator 都是 Estimator。
</div>

预制 Estimator 更加成熟。有时候您需要更多地控制 Estimator 的行为。这就是定制化 Estimator 出现的场景。您可以创建一个定制化 Estimator 来做任何事情。如果希望以某种不寻常的方式连接隐藏层，请编写定制化 Estimator。如果想要计算模型的唯一[度量](https://developers.google.com/machine-learning/glossary/#metric)，请编写定制化 Estimator。基本上，如果想要针对特定问题优化的 Estimator，请编写定制化 Estimator。

一个函数模型（或者 `model_fn`）实现了 ML 算法，使用预制和定制化 Estimator 的唯一区别是：

* 预制 Estimator，已经有人为您编写了函数模型。 
* 定制化 Estimator，您必须自己写函数模型。

您的函数模型可以实现范围更广的算法，定义各种隐藏层和度量。与输入函数一样，所有函数模型都必须接受一组标准的输入参数，并返回一组标准的输出值。就像输入函数可以利用数据集 API 一样，函数模型可以利用层 API 和 度量 API。

让我们看看如何使用定制化 Estimator 解决 Iris 问题。快速提醒 —— 这是我们尝试模仿虹膜模型的组织结构：

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="网络体系结构的图表：输入、两个隐藏层和输出"
  src="https://www.tensorflow.org/images/custom_estimators/full_network.png">
</div>
<div style="text-align: center">
我们的虹膜实施包含四个特征，两个隐藏层，和一个 logits 输出层。
</div>

## 写一个输入函数

我们定制化 Estimator 的实现使用的与来自 [`iris_data.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py) 的 [pre-made Estimator 实现](../guide/premade_estimators.md)的输入函数相同。即：

```python
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # 将输入转换为数据集。
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # 随机播放，重复和批处理示例。
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # 返回管道读取的结束端。
    return dataset.make_one_shot_iterator().get_next()
```

这个输入函数建立一个输入流水线，产生一批 `(features, labels)` 对，其中 `features` 是字典特征。

## 创建功能列

详见 [Premade Estimators](../guide/premade_estimators.md) 和[特征列](../guide/feature_columns.md)章节，你必须定义您的模型特征列来指定模型应该如何使用每个特征。无论是使用预制 Estimator 还是定制化 Estimator,您都可以用相同的方式定义列。

下面的代码为每个输入特征创建一个简单的 `numeric_column` ，表明输入特征值应该直接作为模型的输入：

```python
# 特征列描述如何使用输入。
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

## 写一个模型函数

我们将使用的函数模型具有以下调用签名：

```python
def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration
```

前两个参数是输入函数返回的功能部件和标签的批次：也就是说，`features` 和 `labels` 是您的模型将使用的数据的句柄。`mode` 参数指向调用方是否请求训练、预测或评估。

调用者可以将 `params` 传递给 Estimator 的构造函数。任何传递给构造函数的 `params` 接着都会传递给 `model_fn`。在 [`custom_estimator.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py) 中以下几行会创建 estimator 并设置参数来配置该参数模型。此配置步骤与我们在 [Premade Estimators](../guide/premade_estimators.md) 中如何配置 `tf.estimator.DNNClassifier` 相似。

```python
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'n_classes': 3,
    })
```

实现一个典型的函数模型，你必须做如下操作：

* [定义模型](#define_the_model)
* 为[三种不同模式](#modes)指定附加计算：
  * [预测](#预测)
  * [评估](#评估)
  * [训练](#训练)

##  定义模型

基本的深度神经网络模型必须定义如下三个部分：

* 一个[输入层](https://developers.google.com/machine-learning/glossary/#input_layer)
* 一个或者更多的[隐藏层](https://developers.google.com/machine-learning/glossary/#hidden_layer)
* 一个[输出层](https://developers.google.com/machine-learning/glossary/#output_layer)

### 定义输入层

 `model_fn` 的第一行调用 `tf.feature_column.input_layer` 将特征字典和 `feature_columns` 转换为模型的输入。如下：

```python
    # 使用 `input_layer` 来应用特征列。
    net = tf.feature_column.input_layer(features, params['feature_columns'])
```

上一行应用由特征列定义的转换，创建模型输入层。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="输入层的关系图，在本例中是从原始输入到特性的 1：1 映射。"
  src="https://www.tensorflow.org/images/custom_estimators/input_layer.png">
</div>


### 隐藏层

如果要创建深度神经网络，则必须定义一个或多个隐藏层，层 API 提供一组丰富的函数来定义所有类型的隐藏层，包括卷积层、池层和丢弃层。对于 Iris，我们只需要简单调用 `tf.layers.dense` 来创建隐藏层，其维度由 `params['hidden_layers']` 定义。在每个节点的 `dense` 层连接到上一层中的每个节点。以下是相关代码：

``` python
    # 根据 'hidden_units' 参数构建隐藏层。
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
```

*  `units`  参数定义了给定层中输出神经元的数量。
*  `activation` 参数定义[激活函数](https://developers.google.com/machine-learning/glossary/#activation_function) — 在本例中为 [Relu](https://developers.google.com/machine-learning/glossary/#ReLU)。

这里的变量 `net` 表示网络中当前的顶层。第一次迭代时，`net` 表示输入层。在每次迭代循环中，`tf.layers.dense` 创建一个新层，它使用 `net` 将上一层的输出作为输入。

创建两个隐藏层后，我们的网络如下所示。简而言之，该图不显示每层的所有单元。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="输入层添加了两个隐藏层。"
  src="https://www.tensorflow.org/images/custom_estimators/add_hidden_layer.png">
</div>

请注意 `tf.layers.dense` 提供了许多附加功能，包括设置众多正则化参数的能力。不过，为了简单起见，我们将简单地接受其他参数的默认值。

### 输出层

我们将再次调用 `tf.layers.dense` 来定义输出层，这次没有激活函数：

```python
    # 计算 logits (每个 class 一个)。
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
```

这里 `net` 表示最后的隐藏层。因此，现在这个图层连接如下： 

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="连接到顶层隐藏层的 logit 输出层。"
  src="https://www.tensorflow.org/images/custom_estimators/add_logits.png">
</div>
<div style="text-align: center">
最后的隐藏层输入到输出层。
</div>

定义输出层时，`units` 参数指定输出的数量。因此，将 `units` 设置为 `params['n_classes']`，该模型会为每个类生成一个输出值。输出向量的每个元素都将包含得分或 logit，为关联的 Iris 类计算：Setosa、Versicolor 或 Virginica。

之后，这些 logits 将通过 `tf.nn.softmax` 函数转化为概率。

## 实现训练、评估、预测 {#modes}

创建函数模型的最后一步是编写分支代码实现预测、评估和训练。

当有人调用 Estimator 的 `train` 时，模型函数就会调用 `evaluate` 或者 `predict` 方法。就像这样回调模型的签名函数：

``` python
def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # An instance of tf.estimator.ModeKeys, see below
   params):  # Additional configuration
```

关注第三个论点 — 模式。如下表所示，当某人调用 `train`，`evaluate`，或者 `predict`，Estimator 框架调用您的模型。函数模式的参数设置如下：

| Estimator 方法 | Estimator 模块 |
|:---------------------------------|:------------------|
|`tf.estimator.Estimator.train`|`tf.estimator.ModeKeys.TRAIN`|
|`tf.estimator.Estimator.evaluate`|`tf.estimator.ModeKeys.EVAL`|
|`tf.estimator.Estimator.predict`|`tf.estimator.ModeKeys.PREDICT`|

例如，假设您实例化一个自定义 Estimator 来生成一个 叫做 `classifier` 的对象。然后您执行以下调用： 

``` python
classifier = tf.estimator.Estimator(...)
classifier.train(input_fn=lambda: my_input_fn(FILE_TRAIN, True, 500))
```
Estimator 框架调用您的函数模型，模式设置为 `ModeKeys.TRAIN`。

您的函数模型必须提供处理所有三个模式值的代码。对于每个模式值，您的代码必须返回 `tf.estimator.EstimatorSpec`，包含调用者的请求信息。让我们检查每一种模式。

### 预测

当 Estimator 的 `predict` 方法被调用时，`model_fn` 会接收到 `mode = ModeKeys.PREDICT`。在这种情况下，函数模型必须返回一个包含预测的 `tf.estimator.EstimatorSpec`。 

在进行预测之前，模型必须经过训练。经过训练的模型存储在实例化 Estimator 时建立的 `model_dir` 目录中的磁盘上

为此模型生成预测的代码如下所示：

```python
#  预测计算
predicted_classes = tf.argmax(logits, 1)
if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
```
在预测模式下，预测字典包含模型运行时返回的所有内容。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="额外输出添加到输出层。"
  src="https://www.tensorflow.org/images/custom_estimators/add_predictions.png">
</div>

`predictions` 包含以下三个键值对：

*   `class_ids` 保存表示模型类 的 id  (0, 1, 或者 2)，这个例子中最有可能出现的物种的预测。
*   `probabilities` 保存三个概率 (在本例中 0.02、0.95 和 0.03)。
*   `logit` 保存原始 logit 值 (在本例中 -1.3、2.6 和 -0.9)。

我们通过 `tf.estimator.EstimatorSpec` 的 `predictions`。Estimator 的 `tf.estimator.Estimator.predict` 方法将生成这些字典。

### 计算损失

对于 [training](#训练) 和 [evaluation](#评估) 我们都需要计算模型的损失。这是将要被优化的[目标](https://developers.google.com/machine-learning/glossary/#objective)。

我们可以通过调用 `tf.losses.sparse_softmax_cross_entropy` 来计算损失。当正确类（在索引“label”处）的概率接近 1.0 时，此函数的返回的值将达最低约为 0。随着正确类别的概率降低，返回的损失值逐渐增大。

此函数返回整个批处理的平均值。

```python
# 损失计算。
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
```

### 评估

当 Estimator 的 `evaluate` 方法被调用时，`model_fn` 接收到 `mode = ModeKeys.EVAL`。在这种情况下，模型函数必须返回一 `tf.estimator.EstimatorSpec` 包含模型损失和可选的一个或者更多的指标。

虽然返回度量是可选的，但大多数定制化 Estimator 至少返回一个度量。TensorFlow 提供了一个 Metrics 模块 `tf.metrics` 来计算通用度量。为了简洁起见，我们只返回准确性。`tf.metrics.accuracy` 函数将我们的预测与真值进行比较，即与输入函数提供的标签进行比较。`tf.metrics.accuracy` 函数要求标签和预测具有相同的形状。下面是对 `tf.metrics.accuracy` 的调用：

``` python
#  metrics 指标计算。
accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
```

返回求值的 `tf.estimator.EstimatorSpec`。通常包含以下信息：

* `loss`，这是模型损失。
* `eval_metric_ops`，这是一个可选的度量字典。

所以我们将创建一个包含我们唯一度量标准的字典。如果我们计算了其他度量，我们会将它们添加为与其他键值对相同的字典。然后我们将在 `eval_metric_ops` 中传递该字典 `tf.estimator.EstimatorSpec` 的参数。代码如下：

```python
metrics = {'accuracy': accuracy}
tf.summary.scalar('accuracy', accuracy[1])

if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=metrics)
```

`tf.summary.scalar` 将为 TensorBoard 提供准确性。在 `TRAIN` 和 `EVAL` 两种模式下。（稍后将详细介绍）。

### 训练

当调用 Estimator 的 `train` 方法时，使用 `mode = ModeKeys.TRAIN` 调用 `model_fn`。在这种情况下，模型函数必须返回包含损失和培训操作的 `EstimatorSpec`。

构建训练操作将需要一个优化器。我们将使用 `tf.tra.AdagradOptimizer`，因为我们正在模`DNNClassifier`，默认情况下也使用 `Adagrad`。`tf.train` 包提供了许多其他优化器 — 可以随意使用它们。

以下是构建优化器的代码：

``` python
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
```

接下来，我们使用优化器的 `tf.train.Optimizer.minimize` 方法构建之前计算损失的训练操作。

`minimize` 方法还接受一个 `global_step` 参数。TensorFlow 使用此参数来计算已处理的培训步骤数（以知道何时结束训练运行）。此外，`global_step` 对于 TensorBoard 图的正确工作是必不可少的。只需调用 `tf.train.get_global_step`，并将结果传递给 `minimize` 的 `global_step` 参数。

以下是训练模型的代码：

``` python
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
```

返回用于训练的 `tf.estimator.EstimatorSpec` 必须设置以下字段：

* `loss` 包含损失的函数值。
* `train_op` 执行一个训练步骤。

下面是我们调用 `EstimatorSpec` 的代码：

```python
return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
```

模型功能现在已经完成了。

## 定制化 Estimator

通过 Estimator 基类指定定制化 Estimator，如下所示：

```python
    # 用两个 10 单元建立 2 个隐藏层的 DNN。
    classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })
```
在这里 `params` 字典的作用与关键字相同。`DNNClassifier` 的参数，也就是说，`params` 字典允许您在不修改 `model_fn` 中代码的情况下配置您的 Estimator。

使用我们的 Estimator 进行训练，评估和生成预测的其余代码与 [Premade Estimators](../guide/premade_estimators.md) 章节相同。例如，以下行将训练模型：

```python
# 训练模型
classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps)
```

## TensorBoard

您可以在 TensorBoard 中查看定制化 Estimator 的训练结果。查看此报告，请从命令行启动 TensorBoard，如下所示：

```bsh
# 将 PATH 替换为以 model_dir 形式传递的实际路径
tensorboard --logdir=PATH
```

然后在浏览器输入 [http://localhost:6006](http://localhost:6006) 来打开 TensorBoard。

所有预制 Estimator 都会自动将大量信息记录到 TensorBoard 中。而对于定制化的 Estimators，TensorBoard 只提供一个默认日志（损失图）以及显式告诉 TensorBoard 进行日志记录的信息。对于您刚刚创建的定制化 Estimator，TensorBoard 生成以下内容：

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">

<img style="display:block; margin: 0 auto"
  alt="精度，来自 tensorboard 的'标量'图形"
  src="https://www.tensorflow.org/images/custom_estimators/accuracy.png">

<img style="display:block; margin: 0 auto"
  alt="损失 '来自 tensorboard 的'标量'图形"
  src="https://www.tensorflow.org/images/custom_estimators/loss.png">

<img style="display:block; margin: 0 auto"
  alt="steps/second '来自 tensorboard 的'标量'图形"
  src="https://www.tensorflow.org/images/custom_estimators/steps_per_second.png">
</div>

<div style="text-align: center">
TensorBoard 显示三个图形。
</div>


简而言之，下面三张图将告诉您：

* global_step/sec: 显示多少批的性能指标（更新），我们每秒处理的作为训练模型)。

* 损失：损失报告。  

* 准确性：准确性由以下两行记录： 

  * `eval_metric_ops={'my_accuracy': accuracy}`，在评估期间
  * `tf.summary.scalar('accuracy', accuracy[1])`，在训练期间

这些 tensorboard 是向优化的 `minimize` 方法传递 `global_step` 的主要原因之一。没有它，模型就不能记录这些图的 x 坐标。

注意以下 `my_accuracy` 和 `loss` 图表：

* 橙色线代表训练。
* 蓝色点代表评估。

在训练期间，随着批次的处理，会定期记录摘要（橙色线），这就是为什么它会跨越 x 轴范围的图形。

相比之下，对于每个 `evaluate` 调用，评估只会在图形上产生一个点。这个点包含整个评估调用的平均值。这在图上没有宽度，因为它完全是从特定训练步骤的模型状态（从单个检查点）计算的。

如下图所示，您可以有选择地看到。使用左侧的控件禁用/启用报告。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="display:block; margin: 0 auto"
  alt="复选框允许用户选择显示哪些运行。"
  src="https://www.tensorflow.org/images/custom_estimators/select_run.jpg">
</div>
<div style="text-align: center">
启用或暂停报告。
</div>


## 总结

尽管预制 Estimator 是快速创建新模型的高效方式，通常您需要提供定制化 Estimator 额外的灵活性。幸运的是，预制和定制化遵循相同的编程模型。唯一的实际区别是您必须写一个模型用于自定义 Estimators 的函数，其他的所有内容都是相同的。 

了解更多细节，请务必查看：

* 使用定制化 Estimator [MINIST 的官方 TensorFlow 实现 ](https://github.com/tensorflow/models/tree/master/official/mnist),
 
* TensorFlow [官方模型库](https://github.com/tensorflow/models/tree/master/official),其中包含了更多使用定制化 Estimator 的示例。
* [TensorBoard 视频](https://youtu.be/eBbEDRsCmv4)
* 介绍 TensorBoard。
* [底层介绍](../guide/low_level_intro.md)演示了如何直接使用 TensorFlow 低级 API 进行实验，从而让调试更加简单。
