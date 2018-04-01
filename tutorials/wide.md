# TensorFlow 线性模型

在本教程中，我们将会使用 TensorFlow 中的 tf.estimator API 来解决一个二分类问题：给定人口普查数据，有关一个人的如年龄，教育程度，婚姻状况和职业等信息（特征），我们将尝试预测该人每年是否收入超过 5 万美元（目标标签）。我们将训练一个**逻辑回归**模型，给定一个个体的信息，我们的模型会输出一个0到1之间的数，这个数可以解释成此个体年收入超过 5 万美元的概率。

## 快速构建

可以通过如下步骤尝试本教程的代码:

1. 如果还没安装，请 @{$install$Install TensorFlow}。

2. 下载[教程代码](https://github.com/tensorflow/models/tree/master/official/wide_deep/)。

3. 执行我们提供的数据下载程序：

        $ python data_download.py

4. 使用如下命令执行本教程的代码，训练一个教程中描述的线性模型：

        $ python wide_deep.py --model_type=wide

继续阅读可以了解此代码是如何构建其线性模型的。

## 读取 Census 数据

我们将要使用的数据集是 [Census Income 数据集](https://archive.ics.uci.edu/ml/datasets/Census+Income)。我们提供了 [data_download.py](https://github.com/tensorflow/models/tree/master/official/wide_deep/data_download.py) 来下载数据并进行一些清理。

由于该任务是一个二元分类问题，我们将构造一个名为 “label” 的标签列，如果收入超过 5 万美元，那么它的值为 1，否则为 0。有关参考，请参阅 [wide_deep.py](https://github.com/tensorflow/models/tree/master/official/wide_deep/wide_deep.py)。

接下来，让我们看看数据结构，看看我们可以使用哪些列来预测目标标签。这些列可以分为两类：类别列和连续列：

- 如果某列的值只能是有限集合中的某个类别，则称该列为**类别列**。例如，一个人的关系状况（妻子，丈夫，未婚等）或教育水平（高中，大学等）都是类别列。
- 如果某个列的值可以是连续范围内的任何数值，则称该列为**连续列**。例如，一个人的资本收入（例如 $14,084）是一个连续列。

以下是 Census Income 数据集中可用列的列表：

| 列名 | 类型        | 描述                                                  |
| ----------- | ----------- | ------------------------------------------------------------ |
| age         | 连续型  | 此人的年龄。                                  |
| workclass   | 类别型 | 其雇主类型 (政府、军队、 私企 等)。 |
| fnlwgt         | 连续型  | 普查者认为本样本所能代表的人数（样本权重）。最终权重不会被用到。   |
| education      | 类别型 | 此人最高学历。   |
| education_num  | 连续型  | 最高学历数值表示。 |
| marital_status | 类别型 | 婚姻状态。 |
| occupation     | 类别型 | 职业。 |
| relationship   | 类别型 | 家庭关系：Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried。         |
| race           | 类别型 | 种族：White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black。  |
| gender         | 类别型 | 性别：Female, Male。                     |
| capital_gain   | 连续型  | 登记资本增值。         |
| capital_loss   | 连续型  | 登记资本亏损。         |
| hours_per_week | 连续型  | 每周工作时长。  |
| native_country | 类别型 | 原国籍。  |
| income_bracket | 类别型 | 年收入分类：">5万" 或 "<=5万"，即此人年收入是否高于 5 万。       |

## 将数据转化为张量

在构建 tf.estimator 模型时，输入数据通过 Input Builder 函数指定。 此构建函数在稍后传递给 tf.estimator.Estimator 方法（如 `train` 和 `evaluate`）之前不会被调用。这个函数的目的是构造输入数据，它以 @{tf.Tensor}s 或 @{tf.SparseTensor}s 的形式表示。更详细地说，输入构建函数将返回以下配对：

1. `features`：一个由特征列名到 `Tensors` 或 `SparseTensors` 的映射。
2. `labels`：一个包含标签列的 `Tensor`。

`features` 的键将用于构建下一节中提到的列。由于我们想在调用 `train` 和 `evaluate` 方法时使用不同的数据，所以我们定义一个方法，它根据给定的数据返回一个输入函数。 请注意，返回的输入函数将在构建 TensorFlow 计算图时调用，而不是在运行计算图时调用。它返回的是输入数据作为 `Tensor`（或 `SparseTensor`）的表示，即 TensorFlow 计算的基本单位。

训练集和测试集中的每个连续列都将被转换成 `Tensor`，这通常是表示稠密数据的良好格式。对于类别型数据，我们必须将数据表示为 `SparseTensor`。这种数据格式适合表示稀疏数据。我们的 `input_fn` 使用 `tf.data` API，可以很容易地将转换我们的数据集：

```python
def input_fn(data_file, num_epochs, shuffle, batch_size):
  """为 Estimator 生成输入函数"""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')
    return features, tf.equal(labels, '>50K')

  # 使用 Dataset API 从输入文件中提取行
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # 我们在 shuffle 之后调用 repeat，而不是在之前调用，以防止不同的 epoch 混到一起。
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels
```

## 模型的选择和特征工程

选择和制定正确的特征列是学习一个有效模型的关键。一个 **feature column** 可以是原始 Dataframe 中的原始列之一（我们称之为**基本特征列**），也可以是基于在一个或多个基本列上定义的某些变换创建的任何新列（让我们 称他们**派生特征列**）。基本上，“feature column” 是任何可用于预测目标标签的原始或派生变量的抽象概念。

### 基本的类别型特征列

要为类别型特征定义特征列，我们可以使用 tf.feature_column API 创建一个 `CategoricalColumn`。如果你知道一列中所有可能的特征值的集合，并且只有少数几个，你可以使用 `categorical_column_with_vocabulary_list`。列表中的每个键都将被分配一个从 0 开始的自动增长的 ID。例如，对于 `relationship` 列，通过以下操作，我们可以为特征字符串 “Husband” 分配给一个整数 0 作为 ID ，为 “Not-in-family” 设置 1 作为 ID 等等：

```python
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship', [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried',
        'Other-relative'])
```

但是若我们事先并不知道所有可能值的集合呢？也不成问题，我们可以使用 `categorical_column_with_hash_bucket` 来代替：

```python
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)
```

会发生的是，当我们在训练中遇到它们时，特征列 `occupation` 中的每个可能值都将被散列为一个整数 ID。参看下面的示例插图：

ID  | Feature
--- | -------------
... |
9   | `"Machine-op-inspct"`
... |
103 | `"Farming-fishing"`
... |
375 | `"Protective-serv"`
... |

无论我们选择哪种方式定义 `SparseColumn`，通过查找固定的映射或散列，每个特征字符串都将被映射为一个整数 ID。请注意，哈希碰撞是可能的，但可能不会显着影响模型质量。在底层实现上，`LinearModel` 类负责管理映射并创建 `tf.Variable` 来存储每个特征 ID 的模型参数（也称为模型权重）。模型参数将通过后面将要讨论的模型训练过程来学习。

我们会做类似的技巧来定义其他的类别型特征：

```python
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

# 举个哈希的例子:
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)
```

### 基本的连续型特征列

同样，我们可以为我们想要在模型中使用的每个连续特征列定义一个 `NumericColumn`：

```python
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
```

### 通过分桶使连续特征类别化

有时连续特征和标签之间的关系不是线性的。 假设这么一个例子，一个人的收入可能会在职业生涯的早期阶段随着年龄的增长而增长，然后增长可能会放缓，最终退休后的收入会减少。 在这种情况下，使用原始的 `age` 作为实值特征列可能不是一个好选择，因为模型只能学习三种情况之一：

1.  随着年龄增长，收入总是以某种速度增加（正相关），
2.  随着年龄增长，收入总是以某种速度减少（负相关），或者
3.  无论年龄多少，收入都保持不变（不相关）

如果我们想分别学习收入和各个年龄组之间的细微关联，我们可以利用 **bucketization**。 分桶是将连续特征的整个范围划分为一组连续的桶，然后根据该值落入哪个桶将原始数值特征转换为桶 ID（作为类别型特征）。所以，我们可以在 `age` 上定义 `bucketized_column` 为：

```python
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

这里的 `boundaries` 是一个桶边界的列表。在这个例子中，有 10 个边界，导致有 11 个桶（从 17 岁以下，18-24 岁，25-29 岁......到 65 岁以上）。

### 用 CrossedColumn 交叉多列

单独使用每个基本特征列可能不足以解释数据。例如，不同职业的人的教育程度与标签（收入 > 50,000 美元）之间的关联可能不同。 因此，如果我们只为`教育=“学士”`和`教育=“硕士”`学习一个单一的模型权重，我们将无法捕捉每一个教育-职业组合（例如区分`教育=“学士”和职业=“执行管理”`和`教育=“学士”和职业=“工艺修理”`）。要了解不同特征组合之间的差异，我们可以将**交叉特征列**添加到模型中。

```python
education_x_occupation = tf.feature_column.crossed_column(
    ['education', 'occupation'], hash_bucket_size=1000)
```

我们也可以在两列以上创建一个 `CrossedColumn`。 每个组成列可以是类别型的基本特征列（`SparseColumn`），实值特征列（`BucketizedColumn`），甚至是另一个 `CrossColumn`。下面是一个例子：

```python
age_buckets_x_education_x_occupation = tf.feature_column.crossed_column(
    [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)
```

## 定义逻辑回归模型

处理完输入数据并定义所有特征列后，我们现在准备将它们放在一起并构建逻辑回归模型。在上一节中，我们已经看到了几种基本的和派生的特征列，其中包括：

- `CategoricalColumn`
- `NumericColumn`
- `BucketizedColumn`
- `CrossedColumn`

他们都是抽象类 `FeatureColumn` 的子类，都可以添加到模型的 `feature_columns` 字段中：

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

model_dir = tempfile.mkdtemp()
model = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns + crossed_columns)
```

该模型还会自动学习一个偏差项，它可以控制在不观察任何特征的情况下进行的预测（有关更多解释，请参见“逻辑回归的工作原理”一节）。学习的模型文件将被存储在 `model_dir` 中。

## 训练和评估我们的模型

将所有特征添加到模型后，现在让我们看看如何实际训练模型。使用 tf.estimator API 训练模型仅需用一个命令：

```python
model.train(input_fn=lambda: input_fn(train_data, num_epochs, True, batch_size))
```

模型训练完后，我们就可以评估到底我们的模型在预测留出数据的标签上表现有多好了。

```python
results = model.evaluate(input_fn=lambda: input_fn(
    test_data, 1, False, batch_size))
for key in sorted(results):
  print('%s: %s' % (key, results[key]))
```

最终输出的第一行应该类似于 `accuracy: 0.83557522`，这意味着准确率为 83.6%。自由地尝试更多的特征和转换，你能做得更好！

在堆模型进行评估之后，我们可以使用该模型对个人年收入是否超过五万美元进行预测。

```python
pred_iter = model.predict(input_fn=lambda: input_fn(FLAGS.test_data, 1, False, 1))
for pred in pred_iter:
  print(pred['classes'])
```

模型的预测结果类似于 `b['1']` 或者 `b['0']`，分别表示个人年收入超过五万美元或没有超过五万美元。

如果您希望看到一个可用的端到端示例，则可以下载我们的[样例代码](https://github.com/tensorflow/models/tree/master/official/wide_deep/wide_deep.py)并设置 `model_type` 为 `wide`。

## 加入正则化以防止过拟合

正则化是一种用来避免**过拟合**的技术。当模型在训练数据上表现良好，但在模型以前从未见过的测试数据（如实时交通）上却更糟糕时，过拟合就发生了。过拟合通常发生在模型过于复杂时，例如相比于观察到的训练数据的数量太多的参数。正则化允许你控制模型的复杂性，并使模型在未见数据上具有更强的泛化能力。

在线性模型库中，你可以将L1和L2正则化添加到模型中，如下所示：

```python
model = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=1.0,
        l2_regularization_strength=1.0))
```

L1 和 L2 正则化之间的一个重要区别是，L1 正则化倾向于使模型权重保持为零，从而创建更稀疏的模型，而 L2 正则化也尝试使模型权重接近于零，但不一定为零。因此，如果增加 L1 正则化的强度，你将获得一个更小的模型，因为模型的许多权重将为零。如果特征空间非常大但稀疏，并且存在资源限制，从而无法训练和上线太大的模型时，这通常是可取的。

在实践中，你应该尝试 L1，L2 正则化强度的各种组合，找到最佳参数以最优地控制过拟合并给你理想的模型大小。

## 逻辑回归是如何工作的？

最后，让我们花点时间谈一谈 Logistic 回归模型实际上是什么样子，以防你不熟悉它。我们将标签表示为 \\(Y\\)，并将观察到的特征集合表示为特征向量 
\\(\mathbf{x}=[x_1, x_2, ..., x_d]\\)。我们定义 \\(Y=1\\) 表示一个个体年薪超过 50,000 美元，否则 \\(Y=0\\)。在逻辑回归中，给定特征 \\(\mathbf{x}\\) 的标签为正的概率 (\\(Y=1\\)) 给出如下：

$$ P(Y=1|\mathbf{x}) = \frac{1}{1+\exp(-(\mathbf{w}^T\mathbf{x}+b))}$$

这里 \\(\mathbf{w}=[w_1, w_2, ..., w_d]\\) 是特征 \\(\mathbf{x}=[x_1, x_2, ..., x_d]\\) 的模型权重。\\(b\\) 是常数，通常称为模型的“偏差”。
模型分为两部分 -- 一个线性模型和一个 Logistic 函数：

*   **线性模型**：首先我们可以看到 \\(\mathbf{w}^T\mathbf{x}+b = b + w_1x_1 + ... +w_dx_d\\) 是一个线性模型，它的输出是一个输入特征 \\(\mathbf{x}\\) 的线性函数。偏差 \\(b\\) 是在未观察任何特征做出来的预测。模型权重 \\(w_i\\) 反映了特征 \\(x_i\\) 和正类标签具有怎样的相关关系。如果 \\(x_i\\) 与正类标签是正相关的，权重 \\(w_i\\) 增加，并且概率 \\(P(Y=1|\mathbf{x})\\) 会接近于 1。另一方面，如果 \\(x_i\\) 与正类标签是负相关的，权重 \\(w_i\\) 会降低，概率 \\(P(Y=1|\mathbf{x})\\) 会接近于 0。

*   **Logistic函数**：第二，我们可以看到这有一个 Logistic 函数（也叫作 Sigmoid 函数）\\(S(t) = 1/(1+\exp(-t))\\)应用到了线性模型上。Logistic 函数是用来将线性模型的输出 \\(\mathbf{w}^T\mathbf{x}+b\\) 从任意实数转换到 \\([0, 1]\\) 这个范围的，转化的结果也被视作概率。

模型训练是一个优化问题：目标是找到一组模型权重（即模型参数），以最小化定义在训练数据上的**损失函数**，例如逻辑回归模型的逻辑损失。损失函数衡量真实标签与模型预测之间的差异。如果预测与真实标签非常接近，损失值将会很低；如果预测离标签很远，那么损失值会很高。

## 深入学习

如果您有兴趣了解更多，请查看我们的 @{$wide_and_deep$Wide & Deep Learning Tutorial}，我们将在这里向您展示如何通过通过使用 tf.estimator API 联合训练以结合线性模型和深度神经网络的优势。
