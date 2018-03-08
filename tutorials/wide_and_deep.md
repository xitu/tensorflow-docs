> * 译者：[charsdavy](https://github.com/charsdavy)
> * 校对者：[MRNIU](https://github.com/MRNIU)

# TensorFlow 广度和深度学习的教程

在前文 @{$wide$TensorFlow Liner Model Tutorial} 中，我们使用 [人口收入普查数据集](https://archive.ics.uci.edu/ml/datasets/Census+Income) 训练了一个 logistic 线性回归模型去预测个人年收入超过 5 万美元的概率。TensorFlow 在训练深度神经网络方面效果也很好，那么你可能会考虑该如何取舍它的功能了 -- 可是，为什么不选择两者兼得呢？那么，是否可以将两者的优势结合在一个模型中呢？

在这篇文章中，我们将会介绍如何使用 TF.Learn API 同时训练一个广度线性模型和一个深度前馈神经网络。这种方法结合了记忆和泛化的优势。它在一般的大规模回归和具有稀疏输入特性的分类问题（例如，分类特征存在一个很大的可能值域）上很有效。如果你有兴趣学习更多关于广度和深度学习如何工作的问题，请参考 [研究论文](http://arxiv.org/abs/1606.07792)

![Wide & Deep Spectrum of Models](https://www.tensorflow.org/images/wide_n_deep.svg "Wide & Deep")

现在，我们来看一个简单的例子。

上图展示了广度模型（具有稀疏特征和转换性质的 logistic 回归模型），深度模型（具有一个嵌入层和多个隐藏层的前馈神经网络），广度和深度模型（两者的联合训练）的区别比较。在高层级里，只需要通过以下三个步骤就能使用 TF.Learn API 配置广度，深度或广度和深度模型。

1. 选择广度部分的特征：选择要使用的稀疏基本列和交叉列。

2. 选择深度部分的特征：选择连续列，每个分类列的嵌入维度和隐藏层大小。

3. 将它们一起放入广度和深度模型（`DNNLinearCombinedClassifier`）。

## 安装

如果想要尝试本教程中的代码：

1. 安装 TensorFlow，[请前往此处](http://chars.tech/2017/09/26/tensorflow-pycharm-mac/)。

2. 下载 [教程代码](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)。

3. 安装 pandas 数据分析库。因为本教程中需要使用 pandas 数据。虽然 tf.learn 不要求 pandas，但是它支持 pandas。安装 pandas：

a. 获取 pip：
	
```
# Ubuntu/Linux 64-bit
$ sudo apt-get install python-pip python-dev

# Mac OS X
$ sudo easy_install pip
$ sudo easy_install --upgrade six
```

b. 使用 pip 安装 pandas

```
$ sudo pip install pandas
```

如果你在安装过程中遇到问题，请前往 pandas 网站上的 [说明](http://pandas.pydata.org/pandas-docs/stable/install.html) 。

4. 执行以下命令来训练教程中描述的线性模型：

```
$ python wide_n_deep_tutorial.py --model_type=wide_n_deep
```

请继续阅读，了解次代码如何构建其线性模型。

## 定义基本特征列

首先，定义我们使用的基本分类和连续特征的列。这些列将被作为模型的广度部分和深度部分的构件块。

```python
import tensorflow as tf

gender = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["Female", "Male"])
education = tf.feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    "marital_status", [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ])
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ])
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])

# 展示一个哈希的例子：
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

# 连续基列
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

# 转换
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

## 广度模型：具有交叉特征列的线性模型

广度模型是一个具有稀疏和交叉特征列的线性模型：


```python
base_columns = [
    gender, native_country, education, occupation, workclass, relationship,
    age_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ["education", "occupation"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, "education", "occupation"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ["native_country", "occupation"], hash_bucket_size=1000)
]
```

具有交叉特征列的广度模型可以有效地记忆特征之间的稀疏交互。也就是说，交叉特征列不能概括没有在训练数据中出现的特征组合。让我们采用嵌入方式来添加一个深度模型来修复这个问题。

## 深度模型：嵌入式神经网络

深度模型是一个前馈神经网络，如前图所示。每一个稀疏，高维度分类特征首先都会被转换成一个低维度密集的实值矢量，通常被称为嵌入式矢量。这些低维度密集的嵌入式矢量与连续特征相连，然后在正向传递中馈入神经网络的隐藏层。嵌入值随机初始化，并与其他模型参数一起训练，以最大化减少训练损失。如果你有兴趣了解更多关于嵌入的知识，请在查阅教程 [Vector Representations of Words](https://www.tensorflow.org/versions/r0.9/tutorials/word2vec/index.html) 或在 Wikipedia 上查阅 [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)。

我们将使用 `embedding_column` 配置分类嵌入列，并将它们与连续列连接：

```python
deep_columns = [
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(gender),
    tf.feature_column.indicator_column(relationship),
    # 展示一个嵌入例子
    tf.feature_column.embedding_column(native_country, dimension=8),
    tf.feature_column.embedding_column(occupation, dimension=8),
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
]
```

嵌入的 `dimension` 越高，自由度就越高，模型将不得不学习这些特性的表示。为了简单起见，我们设置所有特征列的维度为 8。从经验上看，关于维度的设定最好是从 \log_{2}(n) 或 k\sqrt[4]{n} 值开始，这里的 n 代表特征列中唯一特征的数量，k 是一个很小的常量（通常小于10）。

通过密集嵌入，深度模型可以更好的概括，并更好对之前没有在训练数据中遇见的特征进行预测。然而，当两个特征列之间的底层交互矩阵是稀疏和高等级时，很难学习特征列的有效低维度表示。在这种情况下，大多数特征对之间的交互应该为零，除了少数几个，但密集的嵌入将导致所有特征对的非零预测，从而可能过度泛化。另一方面，具有交叉特征的线性模型可以用更少的模型参数有效地记住这些“异常规则”。

现在，我们来看看如何联合训练广度和深度模型，让它们优势和劣势互补。

## 将广度和深度模型结合为一体

通过将其最终输出的对数几率作为预测结合起来，然后将预测提供给 logistic 损失函数，将广度模型和深度模型相结合。所有的图形定义和变量分配都已经被处理，所以你只需要创建一个 `DNNLinearCombinedClassifier`：

```python
import tempfile
model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
```

## 训练和评估模型

在训练模型之前，请先阅读人口普查数据集，就像在 [《TensorFlow 线性模型教程》](https://www.tensorflow.org/tutorials/wide) 中所做的一样。 输入数据处理的代码再次为你提供方便：

```python
import pandas as pd
import urllib

# 为数据集定义列名
CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"
]

def maybe_download(train_data, test_data):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        train_file.name)  # pylint: disable=line-too-long
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        test_file.name)  # pylint: disable=line-too-long
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s"% test_file_name)

  return train_file_name, test_file_name

def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # 移除 NaN 元素
  df_data = df_data.dropna(how="any", axis=0)
  labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)
```

阅读数据之后，你可以训练并评估模型：


```python
# 将 num_epochs 设置为 None，以获得无限的数据流
m.train(
    input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
    steps=train_steps)
# 在所有数据被消耗之前，为了运行评估，设置 steps 为 None
results = m.evaluate(
    input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
    steps=None)
print("model directory = %s" % model_dir)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))
```

输出的第一行应该类似 `accuracy: 0.84429705`。我们可以看到使用广度和深度模型将广度线性模型精度约 83.6% 提高到了约 84.4%。如果你想看端对端的工作示例，你可以下载我们的 [示例代码](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)。

请注意，本教程只是一个小型数据基的简单示例，为了让你快速熟悉 API。如果你有大量具有稀疏特征列和大量可能特征值的数据集，广度和深度学习将会更加强大。此外，请随时关注我们的 [研究论文](http://arxiv.org/abs/1606.07792)，以了解更多关于在实际中广度和深度学习在大型机器学习方面如何应用的思考。
