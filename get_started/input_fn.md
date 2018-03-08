# 使用 `tf.estimator` 构建输入函数

本教程将向你展示如何使用 `tf.estimator` 构建输入函数。你会看到如何通过构造一个 `input_fn` 来对数据进行预处理并输入模型之中。最后你会实现一个 `input_fn` ，将训练、评估以及预测数据输入神经网络的 Regressor 之中，并用于预测平均房价。

## 使用 `input_fn` 自定义输入管道

`input_fn` 用于将特征和目标数据传递给 `Estimator` 的 `train` 、`evaluate` 和 `predict` 方法。用户可以在 `input_fn` 内部来对数据实施特征工程或预处理。下面这个例子取自 @{$estimator$tf.estimator Quickstart tutorial}：

```python
import numpy as np

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float32)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(training_set.data)},
    y=np.array(training_set.target),
    num_epochs=None,
    shuffle=True)

classifier.train(input_fn=train_input_fn, steps=2000)
```

### 解密 `input_fn` 

下面的代码说明了输入函数的基本框架：

```python
def my_input_fn():

    # 在这里对你的数据进行预处理...

    # ...然后返回
    #     1) 一个包含相应的特征数据的特征列到张量的映射
    #     2) 包含标签的张量
    return feature_cols, labels
```

输入函数的本体包含了用于预处理输入数据的特定逻辑，比如清除坏数据或[特征缩放](https://en.wikipedia.org/wiki/Feature_scaling)。

输入函数必须返回一下两个值，其中包含要输入给模型的最终特征和标签数据（如上代码所示）：

<dl>
  <dt><code>feature_cols</code></dt>
  <dd>将特征列名映射到包含对应特征数据的 <code>Tensor</code> (或 <code>SparseTensor</code>) 字典。</dd>
  <dt><code>labels</code></dt>
  <dd>包含标签值的 <code>Tensor</code>，即模型的预测值。</dd>
</dl>

### 将特征数据转为张量

若特征或标签数据存储在 [pandas](http://pandas.pydata.org/) dataframe 或者 [numpy](http://www.numpy.org/) array 中，则可以用以上方式构造 input_fn：

```python
import numpy as np
# numpy input_fn.
my_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_data)},
    y=np.array(y_data),
    ...)
```

```python
import pandas as pd
# pandas input_fn.
my_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=pd.DataFrame({"x": x_data}),
    y=pd.Series(y_data),
    ...)
```

对于[稀疏、类别数据](https://en.wikipedia.org/wiki/Sparse_matrix)（大多数值为 0  的数据）而言，你需要使用一个具有三个参数的 `SparseTensor` 来填充数据：

<dl>
  <dt><code>dense_shape</code></dt>
  <dd>张量的形状，需列出每个维度的元素数量。例如，<code>dense_shape=[3, 6]</code> 表示一个 3x6 的二维张量，<code>dense_shape=[2, 3, 4]</code> 表示一个 2x3x4 的二维张量，而 <code>dense_shape=[9]</code> 则表示具有 9 个元素的一维张量。</dd>
  <dt><code>indices</code></dt>
  <dd>张量中非零元素的索引组成的列表，即索引列表中各元素为稀疏张量非零元素索引值所成的列表。例如，<code>indices=[[1,3], [2,4]]</code> 表示索引为 [1, 3] 和 [2, 4] 的元素具有非零值。
</dd>
  <dt><code>values</code></dt>
  <dd>包含所有索引列表对应元素值的一维张量，张量中的各值对应与索引所对应元素的值。例如<code>indices=[[1,3], [2,4]]</code>与所对应参数<code>values=[18, 3.6]</code>表示元素 [1, 3] 的值为 18，且元素 [2, 4] 的值为 3.6。</dd>
</dl>

下面的代码定义了一个 3x5 的二维  `SparseTensor` 。位于 [0,1] 的元素值为 6，而位于 [2,4] 的元素值为 0.5 （其他所有元素为 0）：

```python
sparse_tensor = tf.SparseTensor(indices=[[0,1], [2,4]],
                                values=[6, 0.5],
                                dense_shape=[3, 5])
```

即对应以下稀疏张量：

```none
[[0, 6, 0, 0, 0]
 [0, 0, 0, 0, 0]
 [0, 0, 0, 0, 0.5]]
```

更多关于 `SparseTensor`的信息，请参考 @{tf.SparseTensor}。

### 将 `input_fn` 传递给模型

为了将数据输入给模型进行训练，你需要将创建的输入函数传递给 `train` 函数的 `input_fn` 参数，例如：

```python
classifier.train(input_fn=my_input_fn, steps=2000)
```

注意，`input_fn` 参数只接受函数对象，即 `input_fn=my_input_fn`，而不是函数调用的返回值 `input_fn=my_input_fn()`。所以如果你尝试将参数传递给你的 `input_fn`，并将其最终传递给 `train` 函数，将导致 `TypeError` 的类型错误，如下所示：

```python
classifier.train(input_fn=my_input_fn(training_set), steps=2000)
```

然而，若你希望参数化输入函数，还有其他方法可以实现。你可以使用一个不带参数的函数对你的 `input_fn` 进行封装，并用它来调用你所需参数的输入函数，例如：

```python
def my_input_fn(data_set):
  ...

def my_input_fn_training_set():
  return my_input_fn(training_set)

classifier.train(input_fn=my_input_fn_training_set, steps=2000)
```

亦或者，你可以只用  Python 的 [`functools.partial`](https://docs.python.org/2/library/functools.html#functools.partial) 函数来构造一个新的函数对象，其所有参数值都是固定的：

```python
classifier.train(
    input_fn=functools.partial(my_input_fn, data_set=training_set),
    steps=2000)
```

还有一种方法则是将你的 `input_fn` 调用包含在 [`lambda`](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) 函数中并将其传递给 `input_fn` 这个参数：

```python
classifier.train(input_fn=lambda: my_input_fn(training_set), steps=2000)
```

设计你自己的输入管道具有巨大的优势，正如上面所展示的那样，你可以在只改变数据设置参数的情况下将相同的 `input_fn` 传递给`evaluate` 和 `predict` 函数，例如：

```python
classifier.evaluate(input_fn=lambda: my_input_fn(test_set), steps=2000)
```

这种方法大大增强的代码的可维护性，因为你不再需要定义多个 `input_fn`（比如：`input_fn_train`、`input_fn_test ` 和 `input_fn_predict`）。

最后，你可以使用 `tf.estimator.inputs` 从 numpy 或 pands 数据集中创建 `input_fn`。它的一个额外的优点是你可以使用更多参数，比如 `num_epochs` 和 `shuffle` 参数可以控制 `input_fn` 每次迭代的数据量：

```python
import pandas as pd

def get_input_fn_from_pandas(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pdDataFrame(...),
      y=pd.Series(...),
      num_epochs=num_epochs,
      shuffle=shuffle)
```

```python
import numpy as np

def get_input_fn_from_numpy(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.numpy_input_fn(
      x={...},
      y=np.array(...),
      num_epochs=num_epochs,
      shuffle=shuffle)
```

### 波士顿房价预测的神经网络模型

本教程接下来的部分将向你介绍如何编写输入函数，并将其用于预处理一部分从 [UCI Housing Data Set](https://archive.ics.uci.edu/ml/datasets/Housing) 中获取的波士顿房屋数据，进而将其传递给神经网络的回归器来预测平均房价。

[波士顿 CSV 数据设置](#setup)了你要用来训练神经网络的数据集，它的相关[特征](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)及描述如下表所示：

| 特征      | 描述                   |
| ------- | -------------------- |
| CRIM    | 犯罪率                  |
| ZN      | 住宅用地超过 25000 平方英里的比例 |
| INDUS   | 城镇非零售商用土地的比例         |
| NOX     | 一氧化氮浓度               |
| RM      | 住宅平均房间数              |
| AGE     | 1940 年之前建成的自用房屋比例    |
| DIS     | 到波士顿五个中心区域的距离        |
| TAX     | 每 10000 美元的全值财产税率    |
| PTRATIO | 城镇师生比例               |

你的的模型将用来预测标签 MEDV，即自住房的平均房价，以千美元为单位。

## 数据设置{#setup}

这里能够下载到相关数据集：[boston_train.csv](http://download.tensorflow.org/data/boston_train.csv)、[boston_test.csv](http://download.tensorflow.org/data/boston_test.csv) 和 [boston_predict.csv](http://download.tensorflow.org/data/boston_predict.csv)。

下面的小节介绍了如何一步步创建输入函数，并将这些数据集提供给神经网络的回归器，训练、模型评估并对房价进行预测。最终的完整代码[在此](https://www.tensorflow.org/code/tensorflow/examples/tutorials/input_fn/boston.py)。

### 导入房屋数据

开始之前，请导入 `pandas` 和 `tensorflow` 并设置日志等级 `INFO` 来获得详细的日志输出：

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
```

`COLUMNS` 定义了数据集每列的名字。而 `FEATURES` 和 `LABEL` 的用于区分特征和标签。接下来将三个 CSV 文件 (@{tf.train}, @{tf.test}, 和 [predict](http://download.tensorflow.org/data/boston_predict.csv)) 读取到 _pandas_
 的 `DataFrame` 中:

```python
COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"

training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                       skiprows=1, names=COLUMNS)
prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
```

### 定义 FeatureColumn 并创建 Regressor

接下来，给输入数据创建一个 `FeatureColumn` 列表，用来指定正式用于训练的特征集。由于住房数据所有特征都包含连续值，因此可以使用 `tf.contrib.layers.real_valued_column()` 来创建 `FeatureColumn`:

```python
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
```

提示：关于特征列的深入讨论，请参考 @{$linear#feature-columns-and-transformations$this introduction}；而有关如何对分类数据定义 `FeatureColumns`，请参考 @{$wide$Linear Model Tutorial}。

现在，为了将神经网络回归模型 `DNNRegressor` 实例化，你需要提供两个实参： `hidden_units` 用来指定每个隐藏层节点数（这里我们用两个包含十个节点的隐藏层）和 `feature_columns` 用来指定你刚刚定义的 `FeatureColumns`：

```python
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[10, 10],
                                      model_dir="/tmp/boston_model")
```

### 构建 `input_fn`

为了将输入数据传递给 `regressor`，需要编写一个工厂方法来接收 _pandas_ `Dataframe` 并返回一个 `input_fn`：

```python
def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y = pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)
```

注意，输入数据会在通过形参 `data_set` 传递给  `input_fn` 的实参，也就是说函数会处理你导入的全部 `DataFrame` 数据：`training_set`、 `test_set` 和 `prediction_set`。

有两个额外的实参需要提供：
* `num_epochs`：控制迭代数据的时期。对于训练而言，若将其设置为 `None`，则 `input_fn` 持续返回数据直到满足完成一次训练所需。为了评估及预测，将其设置为1，`input_fn` 会将数据一次性全部返回，然后引发`OutOfRangeError`。这个异常会告诉 `Estimator` 停止评估或预测。
* `shuffle`：确定是否打乱数据。对于评估和预测，若将其设置为 `False`，则 `input_fn` 会逐个迭代数据；而对于训练而言，应该将其设置为 `True`。

### 训练 Regressor

对于训练 Regressor 来说，需要调用 `train` 来接收传递给 `input_fn` 的 `training_set`：

```python
regressor.train(input_fn=get_input_fn(training_set), steps=5000)
```

然后你就会看到类似于下面的输出，每个输出表明了每一百步的训练损失。

```none
INFO:tensorflow:Step 1: loss = 483.179
INFO:tensorflow:Step 101: loss = 81.2072
INFO:tensorflow:Step 201: loss = 72.4354
...
INFO:tensorflow:Step 1801: loss = 33.4454
INFO:tensorflow:Step 1901: loss = 32.3397
INFO:tensorflow:Step 2001: loss = 32.0053
INFO:tensorflow:Step 4801: loss = 27.2791
INFO:tensorflow:Step 4901: loss = 27.2251
INFO:tensorflow:Saving checkpoints for 5000 into /tmp/boston_model/model.ckpt.
INFO:tensorflow:Loss for final step: 27.1674.
```

### 模型评估

接下来，来看看训练好的模型在测试集上的表现如何。这次则是运行 `evaluate` 函数并将 `test_set` 传递给 `input_fn`：

```python
ev = regressor.evaluate(
    input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
```

然后，通过 `ev` 来获取损失函数的值并将其输出：

```python
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))
```

你就能够看到下面的输出结果了：

```none
INFO:tensorflow:Eval steps [0,1) for training step 5000.
INFO:tensorflow:Saving evaluation summary for 5000 step: loss = 11.9221
Loss: 11.922098
```

### 预测

最后，你可以使用这个模型来对 `prediction_set` 里的房价进行预测，其中包含除去没有标签的样本之外的特征数据：

```python
y = regressor.predict(
    input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
# .predict() returns an iterator of dicts; convert to a list and print
# predictions
predictions = list(p["predictions"] for p in itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))
```

你的结果应包含六项房价预测值，单位为『千美元』，例如：

```none
Predictions: [ 33.30348587  17.04452896  22.56370163  34.74345398  14.55953979
  19.58005714]
```

## 其他资源

本教程只介绍了如何为神经网络的 Regressor 创建一个 `input_fn`。若需详细了解如何将 `input_fn` 应用于其他类型的模型，请查看下面的资源：

*   @{$linear$Large-scale Linear Models with TensorFlow}：这篇文章介绍了 TensorFlow 中的线性模型，包括 TensorFlow 所提供的特征列的高层综述及数据转换技巧。

*  @{$wide$TensorFlow Linear Model Tutorial}：这篇教程介绍了如何创建 `FeatureColumns` 和 `input_fn` 并根据人口普查数据的线性分类模型对收入范围进行预测。

*   @{$wide_and_deep$TensorFlow Wide & Deep Learning Tutorial}：这篇教程介绍了使用 `NNLinearCombinedClassifier` 构建的线性模型与神经网络模型的组合泛型模型的  `FeatureColumn` 和 `input_fn` 创建方法。
