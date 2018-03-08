# tf.estimator 快速入门

TensorFlow 的高级机器学习 API（tf.estimator）使得配置，训练和评估
不同的机器学习模型变的简单。本篇教程将会使用 tf.estimator 构建一个 
[neural network](https://en.wikipedia.org/wiki/Artificial_neural_network) 分类器
并在 [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) 这个数据集上训练它，
最后基于花萼和花瓣的几何形状来预测花的种类。你将会
用代码完成下面的 5 个步骤：

1.  将包含 Iris 训练和测试数据的 CSV 文件加载到一个 TensorFlow `Dataset` 中
2.  构建一个 @{tf.estimator.DNNClassifier$neural network classifier}
3.  使用训练数据来训练模型
4.  评估模型的准确率
5.  对新的样本进行分类

注意：在开始本篇教程之前先完成 
@{$install$install TensorFlow on your machine}。

## 完成神经网络源代码

这里是神经网络分类器的全部代码：

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"


def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb") as f:
      f.write(raw)

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))

if __name__ == "__main__":
    main()
```

下面的部分会详细解读这些代码。

## 加载 Iris CSV 数据

[Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) 包含
150 行数据，这些数据由 3 组相关的分类 *Iris setosa*，*Iris virginica* 和 *Iris versicolor*，每组 50 个样本
的数据组成。

![Petal geometry compared for three iris species: Iris setosa, Iris virginica, and Iris versicolor](https://www.tensorflow.org/images/iris_three_species.jpg) **从左到右依次为,
[*Iris setosa*](https://commons.wikimedia.org/w/index.php?curid=170298) (by
[Radomil](https://commons.wikimedia.org/wiki/User:Radomil), CC BY-SA 3.0),
[*Iris versicolor*](https://commons.wikimedia.org/w/index.php?curid=248095) (by
[Dlanglois](https://commons.wikimedia.org/wiki/User:Dlanglois), CC BY-SA 3.0),
and [*Iris virginica*](https://www.flickr.com/photos/33397993@N05/3352169862)
(by [Frank Mayfield](https://www.flickr.com/photos/33397993@N05), CC BY-SA
2.0).**


对于每朵花的样本来说，下面的数据每行都包含这些内容：
[sepal](https://en.wikipedia.org/wiki/Sepal) 的长度，宽度，
[petal](https://en.wikipedia.org/wiki/Petal) 的长度，宽度和花的种类。
花的种类使用整型来表示，0 是 *Iris setosa*，1 
是 *Iris versicolor*， 2 是 *Iris virginica*。

Sepal Length | Sepal Width | Petal Length | Petal Width | Species
:----------- | :---------- | :----------- | :---------- | :-------
5.1          | 3.5         | 1.4          | 0.2         | 0
4.9          | 3.0         | 1.4          | 0.2         | 0
4.7          | 3.2         | 1.3          | 0.2         | 0
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
7.0          | 3.2         | 4.7          | 1.4         | 1
6.4          | 3.2         | 4.5          | 1.5         | 1
6.9          | 3.1         | 4.9          | 1.5         | 1
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
6.5          | 3.0         | 5.2          | 2.0         | 2
6.2          | 3.4         | 5.4          | 2.3         | 2
5.9          | 3.0         | 5.1          | 1.8         | 2

本篇教程中 Iris 的数据已经被随机打乱，分到两个独立的 CSV 文件中。

*   120 个样本的训练集
    ([iris_training.csv](http://download.tensorflow.org/data/iris_training.csv))
*   30 个样本的测试集
    ([iris_test.csv](http://download.tensorflow.org/data/iris_test.csv))。

在开始之前，首先导入一些必要的模块，并定义下载地址和
储存数据集的地址。

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
```

然后，如果本地没有训练和测试集数据，
那就下载它们。

```python
if not os.path.exists(IRIS_TRAINING):
  raw = urlopen(IRIS_TRAINING_URL).read()
  with open(IRIS_TRAINING,'wb') as f:
    f.write(raw)

if not os.path.exists(IRIS_TEST):
  raw = urlopen(IRIS_TEST_URL).read()
  with open(IRIS_TEST,'wb') as f:
    f.write(raw)
```

然后使用 `learn.datasets.base` 中的方法 
[`load_csv_with_header()`](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/datasets/base.py) 
将训练和测试集的数据加载到 `Dataset` 中。`load_csv_with_header()` 需要
接收 3 个必传参数：

*   `filename`，声明 CSV 文件的路径
*   `target_dtype`，声明了 dataset 目标值
    的 [`numpy` datatype](http://docs.scipy.org/doc/numpy/user/basics.types.html)。
*   `features_dtype`，声明了 dataset 特征值
    的 [`numpy` datatype](http://docs.scipy.org/doc/numpy/user/basics.types.html)。

在这里，目标值（就是你训练模型预测的值）是花的种类，
用一个整型表示 0&ndash;2，所以正确的 `numpy` datatype 是 `np.int`：

```python
# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)
```

tf.contrib.learn 中的 `Dataset` 是 
[named tuples](https://docs.python.org/2/library/collections.html#collections.namedtuple)。
你可以通过访问 `data` 和 `target` 字段来获取特征数据。
在这里，`training_set.data` 和 `training_set.target` 分别包含
训练集的特征数据和目标值，同时 `test_set.data` 和 `test_set.target` 也
分别包含测试集的特征数据和目标值。

之后，在 
["Fit the DNNClassifier to the Iris Training Data,"](#fit-dnnclassifier) 中，
你将会使用 `training_set.data` 和 `training_set.target` 来
训练你的模型，在 
["Evaluate Model Accuracy,"](#evaluate-accuracy) 中你则会使用 `test_set.data` 和 `test_set.target`。
但是首先你需要先构建你的模型。

## 构建一个深度神经网络分类器

tf.estimator 提供了很多叫作 `Estimator` 的预定义的模型，使用这些模型
你可以『开箱即用』的在你的数据集上面
运行训练和评估的操作。
这里，你将会配置一个深度神经网络分类器模型来训练 Iris 的数据。
使用 tf.estimator 你可以用下面这些代码
来配置你的 @{tf.estimator.DNNClassifier}：

```python
# Specify that all features have real-value data
feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 20, 10],
                                        n_classes=3,
                                        model_dir="/tmp/iris_model")
```

上面的代码首次定义了模型的特征列，这些特征列指定了
数据集中特征的数据类型。所有的特征数据都是连续的，
所以 `tf.feature_column.numeric_column` 是适合构建特征列的方法。
在数据集中有这 4 个特征（sepal width，sepal height，petal width，和 petal height），
所以相应的形状
必须被设置到 `[4]` 中来保存所有的数据。

然后，代码使用下面的参数创建了一个 `DNNClassifier` 模型：

*   `feature_columns=feature_columns`，上面定义的一套特征列。
*   `hidden_units=[10, 20, 10]`，3 个 
    [hidden layers](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw) 
    分别包含 10，20 和 30 个神经元。
*   `n_classes=3`，3 个目标类，代表 3 个 Iris 种类。
*   `model_dir=/tmp/iris_model`，在模型训练过程中 TensorFlow 将会保存的检查点
    和 TensorBoard 摘要的数据的目录。

## 描述训练输入的管道 {#train-input}

`tf.estimator` API 使用的输入函数创建
了生成模型数据的 TensorFlow 操作。
我们可以使用 `tf.estimator.inputs.numpy_input_fn` 来产生输入管道：

```python
# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)
```

## 使用 Iris 训练数据训练 DNNClassifier {#fit-dnnclassifier}

现在你已经配置好了你的 DNN `classifier` 模型，你可以
用 @{tf.estimator.Estimator.train$`train`} 这个方法
和 Iris 训练数据来训练这个模型。传递 `train_input_fn` 作为 `input_fn` 和
步数来进行训练（这里步数为 2000）。

```python
# Train model.
classifier.train(input_fn=train_input_fn, steps=2000)
```

模型的状态保存在 `classifier` 中，这意味着你可以
随意反复的训练。举个例子，
上面的代码和下面的代码其实是一样的：

```python
classifier.train(input_fn=train_input_fn, steps=1000)
classifier.train(input_fn=train_input_fn, steps=1000)
```

然而，如果你想追踪模型训练的过程，那么你需要使用
一个 TensorFlow @{tf.train.SessionRunHook$`SessionRunHook`} 来
进行日志操作。

## 评估模型的准确性 {#evaluate-accuracy}

你现在已经用 Iris 训练数据训练好了你的 `DNNClassifier` 模型。现在，
你可以使用 @{tf.estimator.Estimator.evaluate$`evaluate`} 这个方法
来在 Iris 测试数据上检验模型的准确性。像 `train` 和 `evaluate` 需要
一个创建输入管道的输入方法。`evaluate` 返回
一个评估结果的 `dict`。下面的代码会
给 Iris 测试数据传递&mdash;`test_set.data` 和 `test_set.target`&mdash;来从结果 `evaluate` 并
打印出 `accuracy`。

```python
# Define the test inputs
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(test_set.data)},
    y=np.array(test_set.target),
    num_epochs=1,
    shuffle=False)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
```

注意：在这里 `num_epochs=1` 这个参数对于 `numpy_input_fn` 非常重要。
`test_input_fn` 将会遍历一次数据然后抛出 `OutOfRangeError`。
这个错误表示分类器已经停止了评估，所以它
将会再评估一次输入。

当你把脚本全部跑完，它会打印出类似下面的东西：

```
Test Accuracy: 0.966667
```

你的准确率可能会有一点偏差，但是也应该高于 90%。这对于
一个如此小的数据集来说是可以接受的。

## 对新样本进行分类

使用 estimator 的 `predict()` 方法来对新的样本进行分类。比如你有
下面个这两个新的花朵样本：

Sepal Length | Sepal Width | Petal Length | Petal Width
:----------- | :---------- | :----------- | :----------
6.4          | 3.2         | 4.5          | 1.5
5.8          | 3.1         | 5.0          | 1.7

你可以使用 `predict()` 方法来预测它们的种类。`predict` 返回一个
字典 generator，这个 generator 可以轻易的被转换成 list。下面的代码
找到并打印出预测的种类：

```python
# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5],
     [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": new_samples},
    num_epochs=1,
    shuffle=False)

predictions = list(classifier.predict(input_fn=predict_input_fn))
predicted_classes = [p["classes"] for p in predictions]

print(
    "New Samples, Class Predictions:    {}\n"
    .format(predicted_classes))
```

你的结果应该像下面这样：

```
New Samples, Class Predictions:    [1 2]
```

因此这个模型预测的结果为第一个样本是 *Iris versicolor*，
第二个样本是是 *Iris virginica*。

## 其他资料

*   想了解更多关于使用 tf.estimator 创建线性模型的知识，请查看 
    @{$linear$Large-scale Linear Models with TensorFlow}。

*   想用 tf.estimator APIs 创建你自己的 Estimator，请查看 
    @{$estimators$Creating Estimators in tf.estimator}。

*   想尝试神经网络模型并在浏览器中可视化的，请查看 
    [Deep Playground](http://playground.tensorflow.org/)。

*   关于神经网络的更高更高级的教程请查看 
    @{$deep_cnn$Convolutional Neural Networks} 和 @{$recurrent$Recurrent Neural Networks}。
