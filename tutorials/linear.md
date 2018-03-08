# TensorFlow 大规模线性模型

tf.estimator 的 API (和其他工具一起）已经为在 TensorFlow 中使用线性模型
提供了一系列丰富的工具。这个文档将是对这些工具的一个综述。
它包括：

   * 什么是线性模型。
   * 为什么要使用线性模型。
   * 在 TensorFlow 中 tf.estimator 是如何使线性模型的构建更简单的。
   * 怎样使用 tf.estimator 融合线性模型和深度学习，更好的发挥两者的优势



你可以通过这个综述知道 tf.estimator 的线性模型工具是否对你有帮助。
而后你可以在 @{$wide$线性模型教程} 中尝试一下。
这个综述的代码用例就来自于那个教程，但是教程会对代码有更详细的说明。

为了更好的理解这个综述，你应该首先对机器学习的基本概念
和 @{$estimator$tf.estimator} 有所了解。

[TOC]

## 什么是线性模型？



*线性模型*使用多个特征的加权和做出预测。
例如，如果你有一个人群的年龄，受教育年限，每周的工作时长的数据，
你可以从这些数据中学习到每个特征的权重值，使得它们的加权和可以预测出一个人的薪水。
你同样可以用线性模型来做分类。

一些线性模型把这个加权和转换成为一种更简便的形式。
例如，逻辑回归将加权和导入一个逻辑函数中，获得一个在 0 和 1 之间的输出。
但是对于每个输入的特征依然只有一个权重值。


## 为什么要使用线性模型？

在当前研究已经显示出更复杂的多层神经网络的巨大威力的情况下，
我们为什么还要使用如此简单的线性模型呢？

线性模型：


   * 相对于深度神经网络，线性模型的训练速度更快。
   * 在非常巨大的特征集上依然有效。
   * 训练算法更简单，不需要经常调试学习速率
   * 比神经网络更容易理解和调试。
   你可以查看分配给每一个特征的权重值来搞清楚什么会对预测产生最大的影响。
   * 是学习机器学习的一个绝佳的起始点。
   * 在工业中广泛使用。


## tf.estimator 是如何帮助你构建线性模型的？


在 TensorFlow 中你可以不借助于任何特殊的 API 来从头创建一个线性模型。
但是 tf.estimator 提供了一些工具使构建有效的大规模线性模型更容易。

### 特征列和转换


设计一个线性模型的大部分工作集中在把原始数据转换成合适的输入特征。
TensorFlow 使用 `特征列` 的抽象方式使这些转换成为可能。

一个 `特征列` 表示你的数据中的一个单一特征。
一个 `特征列` 可能表示一个数量，如高度；
也可能代表一种分类，如眼睛的颜色，
其取值来自于一个离散集合，比如 {'蓝', '棕', '绿'}。



不管是连续性特征（如身高）还是类别性特征（如眼睛颜色），
数据中的一个单一值在输入模型之前都可能会被转换成一个数值序列。
抽象的 `特征列` 使你能像操作单个语义单元一样对特征进行操作。
你可以指定进行哪种转换，选择要加入的特征而不用担心模型输入张量的特定索引。


### 稀疏列


线性模型中的分类特征通常会被转换到一个稀疏向量中，
向量中的每个可能值都有相应的 id 或索引。
例如，如果只有三种可能的眼睛颜色，你可以使用一个长度为 3 的向量
来表示：[1, 0, 0] 表示 '棕'，[0, 1, 0] 表示 '蓝'，[0, 0, 1] 表示 '绿'。
这些向量之所以叫 '稀疏' 是因为当可能的
取值非常大的时候（例如所有的英文单词），向量就会非常长而且会有很多 0.


虽然，你不一定为了处理分类特征数据列而使用 tf.estimator 线性模型，
但线性模型的优势之一就是它们处理大型稀疏向量的能力。
稀疏特征是 tf.estimator 线性特征模型工具的一个主要使用场景。

##### 编码稀疏列


 `特征列` 会自动处理分类值到向量的转换过程，代码如下：

```python
eye_color = tf.feature_column.categorical_column_with_vocabulary_list(
    "eye_color", vocabulary_list=["blue", "brown", "green"])
```

这里的 `eye_color` 就是你的源数据中某一列的名字。


你还可以为你不知道所有可能取值的分类特征生成 `特征列`。
这种情况下，你应该使用 `categorical_column_with_hash_bucket()`，
这个方法会使用哈希函数为特征值建立索引。

```python
education = tf.feature_column.categorical_column_with_hash_bucket(
    "education", hash_bucket_size=1000)
```

##### 交叉特征



由于线性模型会为不同的特征分配单独的权重，
线性模型无法学习特定的特征组合的相对重要性。
如果你有 "最喜欢的运动" 和 "家乡城市" 这两个特征，然后尝试预测是否一个人喜欢穿红色，
你的线性模型是没有办法学到来自圣路易斯的棒球迷特别喜欢穿红色的。



你可以通过创建一个表示 "最喜欢的运动-家乡城市" 的新特征来绕开这个限制。
对于给定的一个人这个特征的值正好是那两个源特征的值的连接：例如，"棒球-圣路易斯"。
这种结合特征被叫做 *"交叉特征"*。

`crossed_column()` 方法使设置交叉特征很容易：

```python
sport_x_city = tf.feature_column.crossed_column(
    ["sport", "city"], hash_bucket_size=int(1e4))
```

#### 连续性特征列

你能像下面这样指定一个连续性特征：

```python
age = tf.feature_column.numeric_column("age")
```


用单个实数表示的连续性特征通常可以直接输入模型中，
TensorFlow 同样为这种特征列提供了很有用的转换方式。

##### 离散化

*离散化* 能把连续性特征列转换成分类性特征列。
这种转换使你能在交叉特征中使用连续性特征列，
在特定区间比较重要的情况下，这种转换也很有用。


离散化把可能值的区间划分成一个个子区间，这些子区间称为 bucket 。

```python
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```


一个值落入子区间，该子区间就成了这个值的分类标签

#### 输入函数


`特征列` 为模型提供了一种输入数据规格，指明如何表示和转换数据。
但是它们本身不提供数据。你需要通过一个输入函数提供数据。



这个输入函数必须返还一个张量字典。其中的每一个键对应某个 `特征列` 的名字，
键所对应的值是一个张量，包含所有数据实例在该特征下的值。
想要更多地了解输入函数请看 @{$input_fn$ 使用 tf.estimator 构建输入函数}，
一个输入函数的实现例子参见：
[线性模型教程代码](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)

输入函数在调用 `train()` 和 `evaluate()` 初始化训练和测试时被传进去，
这将在下一部分说明。

### 线性估算器



Tensorflow 估算器类为回归和分类模型提供一套统一的训练和评估框架。
它们会处理好训练和评估循环过程中的细节，让用户专注于模型的输入和结构。


你可以使用 `tf.estimator.LinearClassifier` `tf.estimator.LinearRegressor`
来创建分别用于分类和回归的估算器。


对于所有的 tensorflow 估算器，运行一个估算器只需要：


   1. 实例化估算器。对于上述两个线性估算器类，你要为构造器传入一个`特征列`列表。
   2. 调用估算器的 `train()` 方法训练它。
   3. 调用估算器的 `evaluate()` 方法查看训练的效果。

例如：

```python
e = tf.estimator.LinearClassifier(
    feature_columns=[
        native_country, education, occupation, workclass, marital_status,
        race, age_buckets, education_x_occupation,
        age_buckets_x_race_x_occupation],
    model_dir=YOUR_MODEL_DIRECTORY)
e.train(input_fn=input_fn_train, steps=200)
# 对训练进行评估（通过测试数据）
results = e.evaluate(input_fn=input_fn_test)

# 打印评估的统计数据
for key in sorted(results):
    print("%s: %s" % (key, results[key]))
```

### 宽深学习


tf.estimator API 还提供了一个估算器类能让你同时训练一个线性模型和
一个深度神经网络。这个新颖的方法结合了线性模型对关键特征的记忆和神经网络的泛化能力。
可以使用 `tf.estimator.DNNLinearCombinedClassifier`
创建这种"宽深"模型：

```python
e = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=YOUR_MODEL_DIR,
    linear_feature_columns=wide_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
```
更多信息，参见 @{$wide_and_deep$宽深学习教程}.
