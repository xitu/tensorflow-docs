# 在 tf.estimator 中创建评估器

tf.estimator 框架中的高层 API 评估器（Estimator）让构建和训练模型变得更加简单。通过在`估计器`中提供的可以实例化的类，我们可以快速配置常见的模型类型，譬如回归模型和分类器：


*   @{tf.estimator.LinearClassifier}:
    可用于构建线性分类模型。
*   @{tf.estimator.LinearRegressor}:
    可用于构建线性回归模型。
*   @{tf.estimator.DNNClassifier}:
    可用于构建神经网络分类模型。
*   @{tf.estimator.DNNRegressor}:
    可用于构建神经网络回归模型。
*   @{tf.estimator.DNNLinearCombinedClassifier}:
    可用于构建神经网络和线性结合的分类模型。
*   @{tf.estimator.DNNLinearCombinedRegressor}:
    可用于构建神经网络和线性结合的回归模型。

但是万一 `tf.estimator` 中预先定义的模型类型不能满足您的要求呢？您想要细粒度的调整模型的配置，譬如自定义损失函数优化器，给每一层神经网络指定不同的激活函数。又或者实现一个打分或者推荐系统，而分类器和回归器都不适应于产生预测。

所以，这篇文章还包含了如何使用 `tf.estiamtor` 中模块来构建一个自定义的`评估器`，它可以根据物理测量的结果来预测[鲍鱼](https://en.wikipedia.org/wiki/Abalone)的年龄，通过这个例子您将学会做如下操作：

*   实例化一个`评估器`
*   构建一个自定义的模型函数
*   使用 `tf.feature_column` 和 `tf.layers` 配置一个神经网络
*   从 `tf.losses` 中选择一个合适的损失函数
*   为您的模型定义一个训练操作
*   生成和返回预测结果

## 预备条件

这篇文章假定你已经知道基本的 tf.estimator 操作，入定义特征列，输入函数，和添加 `train()`/`evaluate()`/`predict()` 操作。如果您之前没有使用过 tf.estimator，或者需要温习，下面的文章或许可以帮到你：

*   @{$estimator$tf.estimator Quickstart}: 如何使用 tf.estmator 快速搭建一个神经网络。
*   @{$wide$TensorFlow Linear Model Tutorial}: 如何使用 tf.estimator 定义特征列，进而搭建线性回归器。
*   @{$input_fn$Building Input Functions with tf.estimator}: 如何构建 input_fn 函数来为您的模型添加数据预处理操作。
    
## 一个鲍鱼年龄的预测器 {#abalone-predictor}

我们可以通过壳的环数来估计[鲍鱼](https://en.wikipedia.org/wiki/Abalone)的年龄。但是这种方法需要对壳进行切割，染色后置于显微镜下观察后才能估计，所以我们想找到预测年龄的其他测量值。

这里有一份鲍鱼的[数据集](https://archive.ics.uci.edu/ml/datasets/Abalone)，
里面包含的[特征数据](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names)
如下：

| 特征名          | 描述                                                      |
| -------------- | --------------------------------------------------------- |
| 长度         | 鲍鱼的长度 (最长的方向; 单位为毫米)           |
| 直径       | 鲍鱼的直径（根据正交的方向来测量：单位为毫米）|
| 高度         | 鲍鱼的高度（鲍鱼肉在壳里，单位为毫米）     |
| 重量   | 整个鲍鱼的重量（单位为克）                      |
| 肉的重量 | 鲍鱼肉的重量（单位为克）                    |
| 脏器的重量 | 晒干后，鲍鱼肉的重量（单位为克）          |
| 壳的重量   | 晒干后，壳的重量（单位为克）                  |

预测的标签是环的数量，代表着鲍鱼的年龄。

![Abalone shell](https://www.tensorflow.org/images/abalone_shell.jpg)
**[“鲍鱼壳”](https://www.flickr.com/photos/thenickster/16641048623/) (by [Nicki Dugan
Pogue](https://www.flickr.com/photos/thenickster/), CC BY-SA 2.0)**

## 准备工作

本教程使用了三个数据集。
[`abalone_train.csv`](http://download.tensorflow.org/data/abalone_train.csv)
 包含了 3320 个标注的训练样本
 [`abalone_test.csv`](http://download.tensorflow.org/data/abalone_test.csv)
 包含了 850 个标注的测试样本
[`abalone_predict`](http://download.tensorflow.org/data/abalone_predict.csv)
 包含了 7 个用于预测的样本

接下来的章节将逐步完成 `评估器` 代码的编写，完整的代码请查看[这里](https://www.tensorflow.org/code/tensorflow/examples/tutorials/estimators/abalone.py)。

## 加载鲍鱼的 CSV 数据到 TensorFlow Datasets

为了给模型供给鲍鱼的数据集，我们需要将数据集下载下来，然后将 CSVs 加载到 TensorFlow 的 `Dataset`s 中。首先，我们需要导入 TensorFlow 和 Python 中的一些库，然后设置 FLAGS：

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

# 导入 urllib
from six.moves import urllib

import numpy as np
import tensorflow as tf

FLAGS = None
```

开启打印日志：

```python
tf.logging.set_verbosity(tf.logging.INFO)
```

然后定义函数来加载 CSVs（接收命令行参数从本地加载或从[官网](https://www.tensorflow.org/)下载）

```python
def maybe_download(train_data, test_data, predict_data):
  """可能会下载训练数据并返回训练和测试文件的名字"""
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_train.csv",
        train_file.name)
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_test.csv", test_file.name)
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)

  if predict_data:
    predict_file_name = predict_data
  else:
    predict_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_predict.csv",
        predict_file.name)
    predict_file_name = predict_file.name
    predict_file.close()
    print("Prediction data is downloaded to %s" % predict_file_name)

  return train_file_name, test_file_name, predict_file_name
```

最后，创建 `main()` 函数来加载鲍鱼的 CSVs 到 `Datasets` 中去，定义标记让用户可以通过命令行来指定训练，测试和预测数据集的 CSV 文件（默认情况下，文件将会从[官网](https://www.tensorflow.org/)上下载）：

```python
def main(unused_argv):
  # 加载数据集
  abalone_train, abalone_test, abalone_predict = maybe_download(
    FLAGS.train_data, FLAGS.test_data, FLAGS.predict_data)

  # 训练样本
  training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_train, target_dtype=np.int, features_dtype=np.float64)

  # 测试样本
  test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)

  # 包含 7 个样本的集合，用于预测鲍鱼的年龄
  prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--train_data", type=str, default="", help="Path to the training data.")
  parser.add_argument(
      "--test_data", type=str, default="", help="Path to the test data.")
  parser.add_argument(
      "--predict_data",
      type=str,
      default="",
      help="Path to the prediction data.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```

## 实例化一个评估器

当实例化 tf.estimator's 中提供的一个类时，譬如说 `DNNClassifier`，你可以提供所有的配置参数给到构造函数，如下所示：

```python
my_nn = tf.estimator.DNNClassifier(feature_columns=[age, height, weight],
                                   hidden_units=[10, 10, 10],
                                   activation_fn=tf.nn.relu,
                                   dropout=0.2,
                                   n_classes=3,
                                   optimizer="Adam")
```

我们不再需要编写任何代码来指导 TensorFlow 训练模型，计算损失，或者返回预测值；因为相关逻辑的实现已经包含在 `DNNClassifier` 中了。

相比之下，当你从零开始创建自己的评估器时，构造器只接受模型配置的两个高级参数 `model_fn` 和 `params`：

```python
nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
```

*   `model_fn`：一个方法对象，里面包含了上面所提到的训练，评价，和预测逻辑。您需要负责实现该函数。下一个章节，[构建 `model_fn`](#constructing-modelfn) 包含了创建模型函数的细节。

*   `params`: 超参数值（譬如：学习率，dropout），类型为字典，将会被传递给 `model_fn`。

注意：类似 `tf.estimator` 模块中预先定义好的回归器和分类器实例化那样，`Estimator` 的实例化也要接受 `model_dir` 和 `config` 参数。

训练鲍鱼年龄预测器模型需要配置一个超参数：学习率。在代码中（下面高亮的部分），`学习率` 被定义为一个常量。

<pre class="prettyprint"><code class="lang-python">tf.logging.set_verbosity(tf.logging.INFO)

<strong># Learning rate for the model
LEARNING_RATE = 0.001</strong></code></pre>

注意：这里我们将 `LEARNING_RATE` 值设为 `0.001`，但你可以根据自己的需求来进行微调，进而让训练得到的模型拥有更好的效果。

然后，添加下面的代码到 `main()` 中，它将包含学习率的 `model_params` 字典作为参数来实例化 `Estimator`：

```python
# 配置模型参数
model_params = {"learning_rate": LEARNING_RATE}

# 实例化评估器
nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)
```

## 构建模型函数（`model_fn`） {#constructing-modelfn}

模型函数是构建`评估器`需要传入的参数，它的基本架构如下：

```python
def model_fn(features, labels, mode, params):
   # 相关逻辑的逻辑如下：
   # 1. 通过使用 TensorFlow 的一些操作函数配置模型。
   # 2. 定义训练 / 评价的损失函数。
   # 3. 定义训练操作所使用的优化器。
   # 4. 生成预测。
   # 5. 返回在 EstimatorSpec 对象中定义的预测 / 损失 / 训练操作 / 评价指标
   return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops) 
```

`model_fn` 必须传入下面三个参数

*   `features`: 一个包含特征的字典类型变量，它将会通过 `input_fn` 传递给模型。
*   `labels`: 一个包含标签数据的`张量`类型变量，它将会通过 `input_fn` 传递给模型。在`预测`的时候，该字段值为空，因为模型将会推断其对应的值。
*   `mode`: 下面的字符串值 @{tf.estimator.ModeKeys} 代表 model_fn 在哪一个场景中调用：
    *   `tf.estimator.ModeKeys.TRAIN`：意味着 `model_fn` 在训练场景中被调用了，即调用了 `train()` 方法。
    *   `tf.estimator.ModeKeys.EVAL`：意味着 `model_fn` 在评价场景中被调用了，即调用了 `evaluate()` 方法。

    *   `tf.estimator.ModeKeys.PREDICT`：意味着 `model_fn` 在预测场景中被调用了，即调用了 `train()` 方法。

在 `model_fn` 中也可以通过传入类型为字典的 `params` 变量来设置训练的超参数（如上所述）。

该函数将执行下列任务（细节将会在后续章节介绍）：

*   配置模型，在这里，鲍鱼年龄预测器使用的是神经网络。
*   定义损失函数，用来计算模型预测的结果是否与真实结果的吻合程度。
*   定义训练模型时所使用的`优化器`，它可以不断的优化模型，从而最小化损失函数的值。

`model_fn` 必须返回 @{tf.estimator.EstimatorSpec} 对象，该对象包含了下面几个值：

*   `mode`（必选）：模型运行在什么场景下，一般来说，你只要返回 `model_fn` 传入的 `mode` 参数的值即可。

*   `predictions`（在`预测`场景下）：一个字典类型的变量，key 值是您输入数据的索引，而 value 则是模型对应给出的预测值，举个例子：
    ```python
    predictions = {"results": tensor_of_predictions}
    ```
	在`预测`场景下，模型返回的预测值实际上是 `EstimatorSpce` 中的 `predict()` 方法返回的，因此您可以自行修改该方法，从而返回自定义的数据格式并消费它。
	

*   `loss`（在`评价`或`训练`场景下），一个损失值的`张量`：该值是模型损失函数的输出值（在后面章节将深入讨论，[为模型定义损失函数](#defining-loss)）。通过遍历所有样本与模型预测结果的差距，我们可以在`训练`场景下输出日志来监控模型训练是否出现异常，而在`评价`场景下则可以作为模型性能好坏的评价指标之一。
    

*   `train_op`（只在`训练`场景下）：执行一次训练的操作。

*   `eval_metric_ops`（可选）：一个字典（名称/值对）类型的返回值，包含了在`评价`场景下需要计算的指标。它的「名称」是您要计算的指标名，而「值」是该计算该指标后的值。@{tf.metrics} 模块为很多常见的指标提供了预定义的函数。下面的 `eval_metric_ops` 操作则使用了 `tf.metrics.accuracy` 来计算`准确率`。
    
    ```python
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels, predictions)
    }
    ```
    
    如果你没有指定 `eval_metric_ops `，那么在模型评价中只有`损失值`会被计算。

### 使用 `tf.feature_column` 和 `tf.layers` 配置神经网络

构建[神经网络](https://en.wikipedia.org/wiki/Artificial_neural_network)需要创建和连接输入层，隐藏层，和输出层。

输入层是一系列的结点（代表模型里的特征），它会接收特征数据，而特征数据 `features` 是作为参数传递到 `model_fn` 函数中的。如果 `features` 包含一个 n 维的`张量`，每一维包含一条特征数据，那么它就可以直接输入到输入层中去。如果 `features` 是一个字典类型的特征列 @{$linear#feature-columns-and-transformations$feature columns}，并且是通过输入函数传入到模型中的，那么你可以使用 @{tf.feature_column.input_layer} 函数将其转换为`张量`输入到输入层中。

```python
input_layer = tf.feature_column.input_layer(
    features=features, feature_columns=[age, height, weight])
```

如上所述，`input_layer()` 接收两个参数：

*   `features`：一个字典类型的变量，其中键是字符串，值是字符串对应的特征数据的`张量`。同时，它也作为 `features` 参数的值传递到 `model_fn` 方法中。
*   `feature_columns`：一个包含模型中所有特征列的列表变量，在上面的例子中，该值为 [`age`, `height`, `weight`]。

一般来说，在神经网络的输入层后面都会连接着一层或者几层的隐藏层，而每一层的输出会经过[激活函数](https://en.wikipedia.org/wiki/Activation_function)（非线性）转换后再输入到下一层。最后的一层隐藏层则会连接到输出层，从而得到模型最终的输出结果。`tf.layers` 模块里提供了 `tf.layers.dense` 函数来构建全连接层。通过控制 `activation` 参数的值，我们可以指定不同的激活函数，其中一些可选的值如下：

*   `tf.nn.relu`：下面的代码中，我们创建了一个全连接的隐藏层，而且上一层的输出会先经过 [ReLU 激活函数](https://en.wikipedia.org/wiki/Rectifier_\(neural_networks\))
    (@{tf.nn.relu})转换后再输入到隐藏层中。

    ```python
    hidden_layer = tf.layers.dense(
        inputs=input_layer, units=10, activation=tf.nn.relu)
    ```    
	
*   `tf.nn.relu6`：下面的代码中，我们创建了一个全连接的隐藏层，而且上一层的输出会先经过 ReLU 6 激活函数
    (@{tf.nn.relu6})转换后再输入到隐藏层中。
    
    ```python
    second_hidden_layer = tf.layers.dense(
        inputs=hidden_layer, units=20, activation=tf.nn.relu6)
    ```    

*   `None`：下面的代码中，我们创建了一个全连接的隐藏层，而且上一层的输出会直接输入到隐藏层中。
    ```python
    output_layer = tf.layers.dense(
        inputs=second_hidden_layer, units=3, activation=None)
    ```
其他的一些激活函数，譬如：

```python
output_layer = tf.layers.dense(inputs=second_hidden_layer,
                               units=10,
                               activation_fn=tf.sigmoid)
```

上述代码创建了一个全连接的`输出层`，它与 `second_hidden_layer` 连接，而且上一层的输出会先经过 sigmoid 激活函数转换后再输入到输出层中。关于 TensorFlow 中预定义的激活函数，请查阅 @{$python/nn#activation_functions$API docs}。

汇总上述的内容，我们可以创建一个全连接的神经网络作为鲍鱼年龄的预测器，并且获得它所对应的预测结果，代码如下：

```python
def model_fn(features, labels, mode, params):
  """评估器所对应的模型函数"""

  # 连接输入层和第一个隐藏层
  # (features["x"]) 将会经过 ReLU 转换
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # 将第二个隐藏层与第一个隐藏层连接在一起，并且激活函数为 ReLU
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # 将输出层与第二个隐藏层连接在一起（无激活函数）
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # 将输出层的值转换为 1 维的张量，然后作为预测值返回
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}
  ...
```

在这里，我们会将鲍鱼的`数据集`通过 `numpy_input_fn` 传递给模型，如上所示，`features` 是一个字典类型的变量 `{"x": data_tensor}`，它会输入到输入层中进行计算。该网络包含了两层的隐藏层，每一层由 10 个结点组成，并且使用了 ReLU 激活函数。最后的输出层输出模型预测结果的时候没有使用激活函数，但使用了 @{tf.reshape} 将模型的预测结果转换成一维数组，并将其保存在变量 `predictions_dict` 中。


## 为模型定义损失函数 {#defining-loss}

`model_fn` 返回的 `EstimatorSpec` 对象是一定要包含`损失值`的：损失值是由损失函数计算所得，它在模型的训练或评价的过程中可以有效的度量预测值与真实值之间的差距。为了方便我们使用，@{tf.losses} 模块封装了一些常用的损失函数：

*   `absolute_difference(labels, predictions)`：使用[绝对差公式](https://en.wikipedia.org/wiki/Deviation_\(statistics\)#Unsigned_or_absolute_deviation)计算损失值（又称 L<sub>1</sub> 损失）

*   `log_loss(labels, predictions)`：使用[对数损失公式](https://en.wikipedia.org/wiki/Loss_functions_for_classification#Logistic_loss)计算损失值（常在逻辑回归中使用）。

*   `mean_squared_error(labels, predictions)`：使用[均方误差公式](https://en.wikipedia.org/wiki/Mean_squared_error)计算损失值（简称 MSE，L<sub>2</sub> 损失）

在 `model_fn` 中，我们使用 `mean_squared_error()`（粗体处）定义了损失函数，如下所示：

<pre class="prettyprint"><code class="lang-python">def model_fn(features, labels, mode, params):
  """评估器的模型函数"""

  # 将输入层和第一个隐藏层连接起来
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # 将第二个隐藏层与第一个隐藏层连接起来
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # 将输出层与第二个隐藏层连接起来
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # 转换输出层值为一维张量，然后作为预测值返回
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}


  <strong># 使用均方误差计算损失值
  loss = tf.losses.mean_squared_error(labels, predictions)</strong>
  ...</code></pre>

如果你想了解 @{tf.losses} 模块中更多的损失函数及其用法，请查看 @{$python/contrib.losses$API guide}

可以将评估的补充指标添加到 `eval_metric_ops` 字典中，下面是添加 `rmse` 指标的一个例子，它会计算根据模型的预测值来计算根均方差。注意`标签`张量被转换成 `float64` 类型，为的是与模型返回的`预测`张量的类型保持一致。

```python
eval_metric_ops = {
    "rmse": tf.metrics.root_mean_squared_error(
        tf.cast(labels, tf.float64), predictions)
}
```

## 为模型定义训练操作

TensorFlow 在用模型拟合训练数据的时候需要使用相关的优化算法，通常的目标是最小化损失值，所以我们需要为训练模型定义一个优化算法，也就是训练操作。一个快捷的创建训练操作的办法就是实例化 `tf.train.Optimizer` 的子类然后调用 `minimize` 方法。

在 `model_fn` 中我们定义了一个梯度下降优化器来最小化损失函数计算出来的值，同时也将学习率传递给 `param` 参数。对于`全局步长`这个参数的设置，我们使用 @{tf.train.get_global_step} 
函数来生成一个合适的整数变量。

```python
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=params["learning_rate"])
train_op = optimizer.minimize(
    loss=loss, global_step=tf.train.get_global_step())
```

关于更多的优化器和其细节，请看 @{$python/train#optimizers$API guide}。

### 完整的模型函数（`model_fn`）

我们已经完成了鲍鱼预测器对应的模型函数的编写，包含了配置神经网络，定制损失函数和训练操作，返回包含 `mode`，`predictions_dict` 和 `train_op` 的 `EstimatorSpec` 对象，代码如下：

```python
def model_fn(features, labels, mode, params):
  """评估器的模型函数"""

  # 将第一个隐藏层连接到输入层
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # 将第二个隐藏层连接到第一个隐藏层，且激活函数为 ReLU
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # 将输出层连接到第二个隐藏（无激活函数）
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # 转换输出层的输出值为一维张量，然后作为预测结果返回
  predictions = tf.reshape(output_layer, [-1])

  # 根据 `ModeKeys.PREDICT` 对应的场景提供评估器
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"ages": predictions})

  # 使用均方误差计算损失值
  loss = tf.losses.mean_squared_error(labels, predictions)

  # 使用均方根误差作为额外的评价指标
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(labels, tf.float64), predictions)
  }

  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params["learning_rate"])
  train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())

  # 为 `ModeKeys.EVAL` 和 `ModeKeys.TRAIN` 场景提供评估器
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)
```

## 运行模型

到此为止，我们已经为鲍鱼年龄预测器定义了`模型函数`，也实例化了一个 `评估器`；剩下的工作就是训练，评价和预测了。

在 `main()` 后面添加如下代码来让神经网络拟合训练数据并评价准确率：

```python
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)

# 训练
nn.train(input_fn=train_input_fn, steps=5000)

# 准确率
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)

ev = nn.evaluate(input_fn=test_input_fn)
print("Loss: %s" % ev["loss"])
print("Root Mean Squared Error: %s" % ev["rmse"])
```

注意：在训练和评价中，训练（`train_input_fn`）和评价（`test_input_fn`）样本都供给了张量 `x` 和 `y`，想了解输入函数相关的细节，请查看 @{$input_fn$Building Input Functions with tf.estimator} 指南。

运行代码后。你应该可以看到下面类似的输出。

```none
...
INFO:tensorflow:loss = 4.86658, step = 4701
INFO:tensorflow:loss = 4.86191, step = 4801
INFO:tensorflow:loss = 4.85788, step = 4901
...
INFO:tensorflow:Saving evaluation summary for 5000 step: loss = 5.581
Loss: 5.581
```

输出的损失值是`模型函数`在 `ABALONE_TEST` 数据集上计算的均方误差值

在 `main()` 后面添加如下代码后，我们可以预测 `ABALONE_PREDICT` 数据集中样本的年龄。

```python
# 输出预测
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": prediction_set.data},
    num_epochs=1,
    shuffle=False)
predictions = nn.predict(input_fn=predict_input_fn)
for i, p in enumerate(predictions):
  print("Prediction %s: %s" % (i + 1, p["ages"]))
```

在这里，`predict()` 函数返回的结果 `predictions` 是可迭代的。然后使用 `for` 循环枚举输出结果。执行这段代码，您应该可以看到如下输出：

```python
...
Prediction 1: 4.92229
Prediction 2: 10.3225
Prediction 3: 7.384
Prediction 4: 10.6264
Prediction 5: 11.0862
Prediction 6: 9.39239
Prediction 7: 11.1289
```

## 附件的资料

恭喜您！您已经成功的使用 tf.estimator 模块创建了一个 `评估器`。更多关于创建 `Estimator`s 的资料，请查看如下的 API 指南：

*   @{$python/contrib.layers$Layers}
*   @{$python/contrib.losses$Losses}
*   @{$python/contrib.layers#optimization$Optimization}
