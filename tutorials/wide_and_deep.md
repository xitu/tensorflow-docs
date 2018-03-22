# TensorFlow Wide & Deep Learning 教程

在之前的 @{$wide$TensorFlow Linear Model Tutorial} 中, 我们使用 [Census Income 数据集](https://archive.ics.uci.edu/ml/datasets/Census+Income)训练了一个逻辑回归模型来预测个人年收入超过5万美元的概率。
TensorFlow 也非常适合训练深度神经网络，您可能会考虑选择哪一个 -- 呃，为什么不是两个？是否有可能在一个模型中结合两者的优势？

在本教程中，我们将介绍如何使用 tf.estimator API 来联合训练广度线性模型和深度前馈神经网络。这种方法结合了记忆和泛化的优势。 它对于具有稀疏输入特征的一般大规模回归和分类问题（例如，具有大量可能特征值的类别型特征）很有用。如果您有兴趣详细了解 Wide＆Deep Learning 的工作原理，请参阅我们的[研究论文](https://arxiv.org/abs/1606.07792)。

![Wide & Deep Spectrum of Models](https://www.tensorflow.org/images/wide_n_deep.svg "Wide & Deep")

上图展示了一个广度模型（具有稀疏特征和变换的逻辑回归），深度模型（具有嵌入层和多个隐藏层的前馈神经网络）和 Wide＆Deep 模型（两者的联合训练））。在较高的层面上，只需 3 个步骤即可使用 tf.estimator API 配置广度，深度或 Wide&Deep 模型：

1.  为广度部分选择特征：选择你想用的基本的稀疏特征列和交叉列。
1.  为深度部分选择特征：选择连续列，每个类别列的嵌入维度以及隐藏层大小。
1.  将他们放进一个 Wide&Deep 模型（`DNNLinearCombinedClassifier`）。

就是这样！我们来看一个简单的例子。

## 快速构建

可以通过如下步骤尝试本教程的代码：

1. @{$install$Install TensorFlow} 如果还没安装。

2. 下载[教程代码](https://github.com/tensorflow/models/tree/master/official/wide_deep/).

3. 执行我们提供的数据下载程序：

        $ python data_download.py

4. 使用如下命令执行本教程的代码，训练一个教程中描述的 Wide&Deep 模型：

        $ python wide_deep.py

继续阅读可以了解此代码是如何构建其模型的。


## 定义基本特征列

首先，我们定义将使用的基础类别型和连续型特征列。这些基础列将成为模型的广度部分和深度部分使用的构建块。

```python
import tensorflow as tf

# 连续列
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

# 举个哈希的例子
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)

# 转换。
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

## 广度模型：具有交叉特征列的线性模型

广度模型是一个线性模型，具有一系列稀疏和交叉的特征列：

```python
base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]
```

更多细节可以参看 @{$wide$TensorFlow Linear Model Tutorial}。

具有交叉特征列的广度模型可以有效记住特征之间的稀疏交互。话虽如此，交叉特征列的一个限制是它们不能推广到没有出现在训练数据中的特征组合。 让我们添加一个嵌入深层模型来解决这个问题。

## 深层模型：带嵌入的神经网络

如上图所示，深层模型是一个前馈神经网络。首先将每个稀疏高维类别型特征转换成低维且稠密的实值向量，通常称为嵌入向量。这些低维稠密嵌入向量与连续特征串联，然后被送到前馈轮中的神经网络的隐藏层。嵌入值被随机初始化，并与所有其他模型参数一起训练，以最大限度地减少训练误差。 如果您有兴趣了解更多关于嵌入的知识，请查看 TensorFlow 教程 @{$word2vec$Vector Representations of Words} 或维基百科上的[词嵌入](https://en.wikipedia.org/wiki/Word_embedding)

另一种表达馈入神经网络的类别列的方法是通过 one-hot 或 multi-hot 表示。这通常适用于只有少数可能值的类别列。 作为 one-hot 表示的例子，对于 `relationship` 列，`"Husband"` 可以表示为 [1, 0, 0, 0, 0, 0]，并且可以将 `"Not-in-family"` 表示为[0, 1, 0, 0, 0, 0]等。这是一个固定的表示，而嵌入更加灵活并在训练时计算。

我们将使用 `embedding_column` 为类别列配置嵌入，并将它们与连续列连接起来。我们也使用 `indicator_column` 来创建一些类别列的 multi-hot 表示。

```python
deep_columns = [
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(marital_status),
    tf.feature_column.indicator_column(relationship),
    # To show an example of embedding
    tf.feature_column.embedding_column(occupation, dimension=8),
]
```

嵌入的“维数”越高，模型将必须学习的特征表示的自由度越高。为了简单起见，我们在此处为所有特征列设置维数为 8。 从经验上来说，关于维数的更明智的决定是以 \\(\log_2(n)\\) 或 \\(k\sqrt[4]n\\) 的顺序开始，其中 \\(n\\) 是特征列中唯一特征的数量，\\(k\\) 是一个小常量（通常小于 10）。

通过稠密嵌入，深度模型可以更好地泛化，并对之前在训练数据中看不到的特征对进行预测。然而，当两个特征列之间的基本交互矩阵是稀疏且高阶的时，很难学习特征列的有效低维表示。在这种情况下，大多数特征对之间的相互作用应该为零，除了少数特征对之间的相互作用之外，稠密嵌入将导致所有特征对的预测为非零，因此可能会过度泛化。另一方面，具有交叉特征的线性模型可以用较少的模型参数有效地记住这些“例外规则”。

现在，让我们看看如何共同训练广度和深度模型，并让它们互相补充优点和缺点。

## 将广度和深度模型组合成一个模型

广度模型和深度模型通过将它们的最终输出的对数似然的和作为预测，然后将预测结果提供给对数损失函数。 所有的计算图定义和变量分配已经在你的框架底层处理过了，所以你只需要创建一个 `DNNLinearCombinedClassifier` ：

```python
model = tf.estimator.DNNLinearCombinedClassifier(
    model_dir='/tmp/census_model',
    linear_feature_columns=base_columns + crossed_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])
```

## 训练和评估模型

在训练模型之前，让我们先读入 Census 数据集，就像我们在 @{$wide$TensorFlow Linear Model tutorial} 中所做的一样。参见 [`wide_deep.py`](https://github.com/tensorflow/models/tree/master/official/wide_deep/wide_deep.py) 中的`data_download.py`以及`input_fn`。

读入数据之后，你可以训练和评估模型：

```python
# 每 `FLAGS.epochs_per_eval` 轮训练、评估一次模型。
for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
  model.train(input_fn=lambda: input_fn(
      FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

  results = model.evaluate(input_fn=lambda: input_fn(
      FLAGS.test_data, 1, False, FLAGS.batch_size))

  # 显示评估度量
  print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
  print('-' * 30)

  for key in sorted(results):
    print('%s: %s' % (key, results[key]))
```

最终的输出精度应该在 85.5% 左右。 如果您希望看到一个可用的端到端示例，则可以下载我们的[样例代码](https://github.com/tensorflow/models/tree/master/official/wide_deep/wide_deep.py).

请注意，本教程只是一个小数据集的简单示例，可帮助你熟悉 API。如果你在具有大量可能特征值的稀疏特征列的大型数据集上进行试验，Wide&Deep Learning 功能将更加强大。再次说明，请随时查看我们的[研究论文](https://arxiv.org/abs/1606.07792)，了解如何将 Wide＆Deep Learning 应用于实际的大型机器学习问题。
