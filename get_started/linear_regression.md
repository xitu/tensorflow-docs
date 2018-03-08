# 回归示例

本章提供下面这些简短的例子来说明如何在 Estimators 中实现回归算法：

<table>
  <tr> <th>Example</th> <th>Data Set</th> <th>Demonstrates How To...</th></tr>

  <tr>
    <td><a href="https://www.tensorflow.org/code/tensorflow/examples/get_started/regression/linear_regression.py">linear_regression.py</a></td>
    <td>[imports85](https://archive.ics.uci.edu/ml/datasets/automobile)</td>
    <td>使用 @{tf.estimator.LinearRegressor} Estimator 基于数值数据来训练一个回归模型。</td>
  </tr>

  <tr>
    <td><a href="https://www.tensorflow.org/code/tensorflow/examples/get_started/regression/linear_regression_categorical.py">linear_regression_categorical.py</a></td>
    <td>[imports85](https://archive.ics.uci.edu/ml/datasets/automobile)</td>
    <td>使用 @{tf.estimator.LinearRegressor} Estimator 基于分类数据来训练一个回归模型。</td>
  </tr>

  <tr>
    <td><a href="https://www.tensorflow.org/code/tensorflow/examples/get_started/regression/dnn_regression.py">dnn_regression.py</a></td>
    <td>[imports85](https://archive.ics.uci.edu/ml/datasets/automobile)</td>
    <td>使用 @{tf.estimator.DNNRegressor} Estimator 基于离散数据和深度神经网络来训练一个回归模型。</td>
  </tr>

  <tr>
    <td><a href="https://www.tensorflow.org/code/tensorflow/examples/get_started/regression/custom_regression.py">custom_regression.py</a></td>
    <td>[imports85](https://archive.ics.uci.edu/ml/datasets/automobile)</td>
    <td>使用 @{tf.estimator.Estimator} 来训练一个自定义 dnn 回归模型。</td>
  </tr>

</table>

上面的例子依赖于以下数据集程序：

<table>
  <tr> <th>Utility</th> <th>Description</th></tr>

  <tr>
    <td><a href="../../examples/get_started/regression/imports85.py">imports85.py</a></td>
    <td><tt>imports85</tt> 这个程序提供了一些工具函数，它们可以将数据集加载成其他 TensorFlow 程序 (例如, <tt>linear_regression.py</tt> 和 <tt>dnn_regression.py</tt>) 可以使用的格式。</td>
  </tr>


</table>


<!--
## Linear regression concepts
## 线性回归概念

If you are new to machine learning and want to learn about regression,
watch the following video:
如果你是机器学习的新人并且还想多了解关于回归的知识的话，可以观看下面的视频：

(todo:jbgordon) Video introduction goes here.
-->

<!--
[When MLCC becomes available externally, add links to the relevant MLCC units.]
-->


<a name="running"></a>
## 运行示例

你必须先 @{$install$安装 TensorFlow} 才能运行这些示例。根据你安装 TensorFlow 的方式，你或许还需要激活你的 TensorFlow 的运行环境。然后按照下面的步骤操作：

1. 从 github 上 clone TensorFlow 的仓库。
2. 使用 `cd` 命令切换至下载路径的顶层。
3. 切换到你的 tensorflow 当前版本所在的分支：`git checkout rX.X`
4. `cd tensorflow/examples/get_started/regression`。

你现在可以通过 Python 运行 TensorFlow  `tensorflow/examples/get_started/regression` 目录下的任何示例：

```bsh
python linear_regressor.py
```

在训练期间，三个程序都会输出以下信息：

* 检查点目录的名字，这对于 TensorBoard 来说很重要。
* 每迭代 100 步之后的训练误差，这个数值可以帮助你确定模型是否正在收敛。

举例来说，对于 `linear_regressor.py` 这个程序，下面的就是一些可能的输出：

```bsh
INFO:tensorflow:Saving checkpoints for 1 into /tmp/tmpAObiz9/model.ckpt.
INFO:tensorflow:loss = 161.308, step = 1
INFO:tensorflow:global_step/sec: 1557.24
INFO:tensorflow:loss = 15.7937, step = 101 (0.065 sec)
INFO:tensorflow:global_step/sec: 1529.17
INFO:tensorflow:loss = 12.1988, step = 201 (0.065 sec)
INFO:tensorflow:global_step/sec: 1663.86
...
INFO:tensorflow:loss = 6.99378, step = 901 (0.058 sec)
INFO:tensorflow:Saving checkpoints for 1000 into /tmp/tmpAObiz9/model.ckpt.
INFO:tensorflow:Loss for final step: 5.12413.
```


<a name="basic"></a>
## linear_regressor.py

`linear_regressor.py` 训练了一个可以基于两个数值特征预测汽车价格的模型。

<table>
  <tr>
    <td>Estimator</td>
    <td><tt>LinearRegressor</tt>, which is a pre-made Estimator for linear
        regression.</td>
  </tr>

  <tr>
    <td>Features</td>
    <td>Numerical: <tt>body-style</tt> and <tt>make</tt>.</td>
  </tr>

  <tr>
    <td>Label</td>
    <td>Numerical: <tt>price</tt>
  </tr>

  <tr>
    <td>Algorithm</td>
    <td>Linear regression.</td>
  </tr>
</table>

训练模型之后，这个程序会输出这两个汽车模型的预测价格。


<a name="categorical"></a>
## linear_regression_categorical.py

这个程序向我们说明了标识分类特征的一些方法。同时也展示了如何基于分类和数值特征来训练一个线性模型。

<table>
  <tr>
    <td>Estimator</td>
    <td><tt>LinearRegressor</tt>, which is a pre-made Estimator for linear
        regression. </td>
  </tr>

  <tr>
    <td>Features</td>
    <td>Categorical: <tt>curb-weight</tt> and <tt>highway-mpg</tt>.<br/>
        Numerical: <tt>body-style</tt> and <tt>make</tt>.</td>
  </tr>

  <tr>
    <td>Label</td>
    <td>Numerical: <tt>price</tt>.</td>
  </tr>

  <tr>
    <td>Algorithm</td>
    <td>Linear regression.</td>
  </tr>
</table>


<a name="dnn"></a>
## dnn_regression.py

像 `linear_regression_categorical.py` 一样，`dnn_regression.py` 这个例子也训练了一个基于两个特征预测汽车价格的模型，但是 `dnn_regression.py` 这个例子使用了一个深度神经网络来训练模型。这两个例子都依赖于同样的特征。`dnn_regression.py` 展示了如何在一个深度神经网络中处理分类特征。

<table>
  <tr>
    <td>Estimator</td>
    <td><tt>DNNRegressor</tt>, which is a pre-made Estimator for
        regression that relies on a deep neural network.  The
        `hidden_units` parameter defines the topography of the network.</td>
  </tr>

  <tr>
    <td>Features</td>
    <td>Categorical: <tt>curb-weight</tt> and <tt>highway-mpg</tt>.<br/>
        Numerical: <tt>body-style</tt> and <tt>make</tt>.</td>
  </tr>

  <tr>
    <td>Label</td>
    <td>Numerical: <tt>price</tt>.</td>
  </tr>

  <tr>
    <td>Algorithm</td>
    <td>Regression through a deep neural network.</td>
  </tr>
</table>

在打印出损失值之后，程序会在测试集中输出均方误差。


<a name="dnn"></a>
## custom_regression.py

`custom_regression.py` 这个例子同样也是基于真实价格和分类作为输入来预测汽车价格的模型。不像 `linear_regression_categorical.py` 和 `dnn_regression.py`，这个例子没有使用预估计量，而是使用 @{tf.estimator.Estimator$`Estimator`} 类定义了一个自定义模型。这个自定义模型与 `dnn_regression.py` 定义的模型十分相似。

这个自定义模型是通过构造器的参数 `model_fn` 来定义的。当 `model_fn` 被调用时，`params` 字典会被传入 `model_fn`，这样使得模型的复用性更高。

`model_fn` 将会返回一个 @{tf.estimator.EstimatorSpec$`EstimatorSpec`}，它是一个简单的可以表示 `Estimator` 的结构。
