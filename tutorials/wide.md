# TensorFlow 线性模型教程

在这个教程中,我们将使用 Tensorflow 中的 tf.estimator API 来解决二分类问题：
给定含有年龄、性别、教育程度和职业（的这些特征）的人口普查数据，
我们将用其预测每个人是否达到年收入 50,000 美元（目标标签）。
我们会训练一个 **Logistic 回归**模型，对于每个人的信息，输出 0 或 1，表明这个人是否能达到年收入 50,000 美元。

## 起步

为了尝试本教程中的代码：

1.  @{$install$Install TensorFlow} 如果你还没有安装。

2.  下载 [教程代码](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)。

3.  安装 pandas 库（用作数据分析的库）。tf.estimator 不依赖于 pandas，但是支持它，在本教程中也使用了 pandas。安装 pandas：

    a. 安装 `pip`：

        # Ubuntu/Linux 64 位
        $ sudo apt-get install python-pip python-dev

        # macOS
        $ sudo easy_install pip
        $ sudo easy_install --upgrade six

    b. 使用 `pip` 安装 pandas：

        $ pip install -U pandas

    如果你在安装 pandas 中遇到了麻烦，请参阅 pandas 网站中的 [说明](https://pandas.pydata.org/pandas-docs/stable/install.html)。

4. 使用以下命令执行教程代码，这将训练本教程中描述的线性模型：

        $ python wide_n_deep_tutorial.py --model_type=wide

以下内容说明了此代码是如何建立线性模型的。


## 读取人口普查数据

我们将使用的数据集是 [人口普查收入数据集](https://archive.ics.uci.edu/ml/datasets/Census+Income)。 您可以自行下载 [训练数据](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data) 和 [测试数据](https：//archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test) 或使用如下代码：

```python
import tempfile
import urllib
train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)
```

在 CSV 文件下载完成后，将它们读到 pandas 的 dataframes 里。

```python
import pandas as pd
CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"]
df_train = pd.read_csv(train_file.name, names=CSV_COLUMNS, skipinitialspace=True)
df_test = pd.read_csv(test_file.name, names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)
```

由于该任务是一个二分类问题，我们将创建一个名为“标签”的标签列，如果收入超过 50K 美元，那么它的值为 1，否则为 0。

```python
train_labels = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
test_labels = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
```

接下来，我们来看 dataframe，看看我们可以使用哪些列来预测目标标签。这些列可以分为两类：分类和连续的列：

*   如果某列的数据只可能是有限集合中的某个值，那么这一列就称之为**分类**。比如，一个人的出生国家（美国，印度，日本，等等）或者教育程度（高中，大学，等等）就是分类的列。
*   如果某列的数据是连续范围内的任何一个数值，那么这一列就称之为**连续**。比如，一个人的资本收入（比如 14,084 美元）就是连续的列。

以下是人口普查收入数据集中可用的列的列表：

| 列名            | 类型 | 描述                                   | {.sortable}
| -------------- | --- | -------------------------------------- |
| age            | 连续 | 市民的年龄。                             |
| workclass      | 分类 | 市民职位的类型(政府, 军队, 私人, 等等)。    |
| fnlwgt         | 连续 | 人口普查员认为该市民所代表的人数(样本权重)。  :
:                :             : 最终的权重不会被使用。             :
| education      | 分类 | 市民的最高学历。                          |
| education_num  | 连续 | 市民最高学历的数字形式。                   |
| marital_status | 分类 | 市民的婚姻状况。                          |
| occupation     | 分类 | 市民的职位。                             |
| relationship   | 分类 | Wife, Own-child, Husband,              |
:                :     : Not-in-family, Other-relative,         :
:                :     : Unmarried. 妻子, 育儿, 丈夫, 未在家庭,     :
:                :     : 其他亲属, 未婚。这些值之一。                :
| race           | 分类 | White, Asian-Pac-Islander,             |
:                :     : Amer-Indian-Eskimo, Other, Black.      :
:                :     : 白种人, 亚太岛民, 美洲-印度-爱斯基摩人,     :
:                :     : 其他, 黑种人。这些值之一。 :
| gender         | 分类 | Female, Male.女性，男性。这些值之一。     |
| capital_gain   | 连续 | 收益资本。                              |
| capital_loss   | 连续 | 损失资本。                              |
| hours_per_week | 连续 | 每周工作时间。                           |
| native_country | 分类 | 市民的出生国家。                         |
| income         | 分类 | ">50K" 或 "<=50K",表明该人的年收入       |
:                :     : 是否超过 50,000 美元                    :


## 将数据转换成张量(Tensor)

当建立了一个 tf.estimator 模型，输入数据是通过输入构建函数来指定的。
这个构建函数在传递给 tf.estimator.Estimator 方法（如 `train` 和 `evaluate`）之前不会被调用。
这个函数是用来构建输入数据，它以 @{tf.Tensor}s 或者 @{tf.SparseTensor} 的形式表示。具体来说，
输入构建函数返回以下一对数据：

1. `feature_cols`：从特征列中被命名为 `Tensors` 或 `SparseTensors` 的字典。
2. `label`：包含标签列的 `Tensor`。

`feature_cols` 的键值将会在下一节中用到。 因为我们想分别使用不同的数据调用 `train` 和 `evaluate` 方法，
所以我们定义一个方法，根据给定的数据返回一个输入函数。
请注意，返回的输入函数将在 TensorFlow 图被构造时进行调用，而不是在其运行时被调用。
它返回的 `Tensor`（或`SparseTensor`）是作为 TensorFlow 计算的基本单位的输入数据。

我们使用 `tf.estimator.inputs.pandas_input_fn` 方法从 pandas 的 dataframe 中创建一个输入函数。
训练或测试数据的 dataframe 中的每个连续列都将被转换成一个 `Tensor`，一般来说这是表示稠密数据的一种很好的格式。
对于分类数据，我们必须将数据表示为 `SparseTensor`。这种数据格式适合用来表示稀疏数据。
表示输入数据的另一种更高级的方法是构建代表文件或其他数据源的 @{$python/io_ops#inputs-and-readers$Inputs And Readers}，并在 TensorFlow 运行图时遍历文件。

```python
def input_fn(data_file, num_epochs, shuffle):
  """输入构建函数"""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # 去掉 NaN 数据
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
## 为模型选择和构造特征

选择和构造正确的特征是学习有效模型的关键。
**特征列**可以是原始数据框中的原始列之一（我们称之为**基本特征列**），
也可以是基于在一个或多个基本列上定义的进行某些转换而创建的任何新列（我们称之为**派生特征列**）。
基本上来说，“特征列"是用于预测目标标签的任何原始或派生变量的抽象概念。

### 基本分类特征列

要为分类特征定义特征列，我们可以使用 tf.feature_column API 创建一个 `CategoricalColumn`。
如果您知道该列的所有可能的特征值的集合，并且只有很少可能的特征值，您可以使用 `categorical_column_with_vocabulary_list`。
例如，对于 `gender` 列，我们可以通过执行以下操作将特征字符串 `Female` 分配给整数 0，`Male` 分配给整数 1：

```python
gender = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["Female", "Male"])
```

如果我们不知道所有可能的特征值的集合怎么办？没关系。我们可以使用 `categorical_column_with_hash_bucket` 来代替。

```python
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
```

以上代码会把 `occupation` 的每个可能的特征值散列成一个整数 ID （就像训练时那么散列）。看以下这个例子：

ID  | Feature
--- | -------------
... |
9   | `"Machine-op-inspct"`
... |
103 | `"Farming-fishing"`
... |
375 | `"Protective-serv"`
... |

不论我们如何选择定义一个 `SparseColumn` 的方式，都会通过查找一个固定的映射或散列，将每个特征字符串映射到一个整数 ID。
请注意，散列冲突是可能的，但可能不会显著影响模型质量。
在这种情况下，LinearModel 类负责管理映射并创建 `tf.Variable` 来存储每个特征 ID 的模型参数（也称为模型权重）。
模型参数将会通过我们之后会用到的模型训练来学习。

我们会使用类似的技巧来定义其他的分类特征：

```python
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
native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)
```

### 基本连续特征列

同样地，我们可以为每一个我们想在模型中使用的连续特征列定义一个 `NumericColumn`：

```python
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")
```

###　通过桶化连续性特征的分类

有时连续特征和标签之间的关系不是线性的。
假设一下，一个人的收入在职业生涯初期可能会随着年龄的增长而增长，然后增长会有所放缓，退休后收入会减少。
在这种情况下，使用原始的 `age` 作为实值特征列可能不是一个好的选择，因为模型只能学习到三种情况之一：

1. 收入总是随着年龄的增加而增加（正相关）。
2. 收入总是随着年龄的增加而减少（负相关）。
3. 收入总是和年龄无关。（不相关）。

如果我们想要学习收入和每个年龄组之间的细粒度的相关性，我们需要利用**桶化**。
桶化是将连续特征的整个范围划分为一组连续的 箱/桶 的过程，然后根据值落入哪个桶中，
就将其原始数值特征转换为桶 ID（作为一个分类特征）。
所以，我们可以定义 `bucketized_column` 为 `age`：

```python
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```
其中 `boundaries` 是桶边界的列表。在上述情况下，有 10 个边界值，产生了 11 个年龄组桶（从 0-17，18-24，25-29...到65及其以上）。

### 使用 CrossedColumn 相交多列

分别使用每个基本特征列可能并不足以解释数据。
例如，不同职业的受教育水平与目标标签（赚取大于 50,000 美元）的关系可能不同。
因此，如果我们只学习 `education="Bachelors"` 和 `education="Masters"` 的单个模型权重，
就无法获取到每一个教育--职业的组合（例如 `education="Bachelors" AND occupation="Exec-managerial"` 和 `education="Bachelors" AND occupation="Craft-repair"` 就是不同的）。
要了解不同特征组合之间的差异，我们可以引入**交叉特征列**到这个模型中。

```python
education_x_occupation = tf.feature_column.crossed_column(
    ["education", "occupation"], hash_bucket_size=1000)
```

我们也可以在两个以上的列上创建一个 `CrossedColumn`。
每一个构成的列可以是分类基本特征列（`SparseColumn`），一个经过桶化的实值特征列（`BucketizedColumn`），或甚至是其他的 `CrossedColumn`。
举个例子：

```python
age_buckets_x_education_x_occupation = tf.feature_column.crossed_column(
    [age_buckets, "education", "occupation"], hash_bucket_size=1000)
```

## 定义 Logistic 回归模型


在处理输入数据并定义所有特征列后，我们现在准备将它们放在一起并构建一个 Logistic 回归模型。
在上一节中，我们已经看到了几种类型的基本和派生特征列，包括：

* `CategoricalColumn`
* `NumericColumn`
* `BucketizedColumn`
* `CrossedColumn`

所有的这些列都是抽象类 `FeatureColumn` 的子类，并且都可以添加到模型的 `feature_columns` 字段中：

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

model_dir = tempfile.mkdtemp()
m = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns + crossed_columns)
```

该模型也会自动学习一个偏置项，偏置项控制着不预测任何特征的预测（更多解释请参见“Logistic 回归是如何工作的”一节）。
经过学习的模型文件将被存储在 `model_dir` 中。

## 训练和评估我们的模型

将所有的特征添加到模型之后，是时候让我们看看该如何训练这个模型了。
使用 TF.Learn API 训练一个模型只需一行代码：

```python
# 将 num_epochs 设为 None 表示需要获得无限的数据流。
m.train(
    input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
    steps=train_steps)
```

在模型训练完成后，我们可以评估我们的模型在对保留数据预测目标标签时的好坏：

```python
results = m.evaluate(
    input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
    steps=None)
print("model directory = %s" % model_dir)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))
```

输出的第一行应该类似于 `accuracy: 0.83557522`，表示准确率达到了 83.6%。
你可以尝试更多的特征和转换，看看你能不能做得更好！

如果你想要看完整的代码，你可以下载我们的 [样例代码](https://www.tensorflow.org/code/tensorflow/examples/learn/wide_n_deep_tutorial.py)，
然后将 `model_type` 设为 `wide`。


## 添加正则化以防止过拟合

正则化是一种用来避免**过拟合**的方法。
模型在训练数据上表现良好，但模型在以前从未见过的测试数据（如实时流量）上很糟糕时，就是过拟合。
过拟合通常发生在模型过于复杂的情况下，例如相对于观测的训练数据的数量使用了太多的参数。
正则化允许你控制你的模型的复杂性，并且使模型对于未知数据具有良好的鲁棒性。

在这个线性模型库中，你可以为模型添加 L1 或 L2 正则化：

```
m = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=1.0,
      l2_regularization_strength=1.0))
```
L1 和 L2 正则化之间的一个重要区别是，L1正则化倾向于使模型权重保持为零，从而创建更稀疏的模型，
而 L2 正则化也尝试使模型权重接近于零，但不一定是零。
因此，如果您增加 L1 正则化的强度，您会有一个较小的模型尺寸，因为许多模型权重都为零。
通常在特征空间非常大有很稀疏的时候，以及在你的资源限制让你无法保存一个过大的模型时，L1 正则化是可取的。


在实践中，你应该尝试 L1，L2 正则化强度的各种组合，找到控制过拟合的最佳参数，你会得到一个理想大小的模型。

## Logistic 回归是如何工作的


最后，让我们花一点时间谈一谈 Logistic 回归模型实际上是什么样子的，以免你还不熟悉它。
我们定义标签为 \\(Y\\) ，然后设置可观测的特征为一个特征向量，即 \\(\mathbf{x}=[x_1, x_2, ..., x_d]\\)。
我们再定义 \\(Y=1\\) 表明一个人年收入大于或等于 50,000 美元，反之为 \\(Y=0\\)。
在 Logistic 回归中，对给定特征 \\(\mathbf{x}\\) 预测标签为正 (\\(Y=1\\)) 的概率是：

$$ P(Y=1|\mathbf{x}) = \frac{1}{1+\exp(-(\mathbf{w}^T\mathbf{x}+b))}$$

其中 \\(\mathbf{w}=[w_1, w_2, ..., w_d]\\) 是特征 \\(\mathbf{x}=[x_1, x_2, ..., x_d]\\) 的模型权重。
\\(b\\) 是一个被称为模型**偏置项**的常量。该方程由两部分组成——一个线性模型和一个 Logistic 函数：

*   **线性模型**：首先，我们可以看到 \\(\mathbf{w}^T\mathbf{x}+b = b +
    w_1x_1 + ... +w_dx_d\\) 是一个线性模型，它的输出是输入特征 \\(\mathbf{x}\\)。
    偏置项 \\(b\\) 是没有观测到任何特征的预测情况。模型权重 \\(w_i\\) 表明了特征 \\(x_i\\) 是否和正标签
    相关。如果 \\(x_i\\) 和正标签是正相关的，那么当权重 \\(w_i\\) 增大时，\\(P(Y=1|\mathbf{x})\\) 概率会
    更接近于 1。反而言之，如果 \\(x_i\\) 和正标签是负相关的，那么当权重 \\(w_i\\) 减小时，
    \\(P(Y=1|\mathbf{x})\\) 概率会更接近于 0。

*   **Logistic 函数**：其次，我们可以看这个 logistic 函数（也被称之为 sigmoid 函数）\\(S(t) = 1/(1+\exp(-t))\\) 
    也被应用到这个线性模型中。这个 logistic 函数是用于将线性函模型的输出 \\(\mathbf{w}^T\mathbf{x}+b\\) 从任意实数
    转变成 \\([0, 1]\\) 的范围，我们可以将其称之为概率。

模型训练是一个优化问题：目标是找到一组模型权重（即模型参数），在训练数据上最小化**损失函数**，
例如 Logistic 回归模型的 logistic 损失。
损失函数量化真实标签和模型预测结果之间的差异。
如果预测结果与真实标签非常接近，损失值会很低；如果预测结果与真实标签很远，那么损失值就会很高。

## 深入学习

如果你很想深入学习，可以看看我们的 @{$wide_and_deep$Wide & Deep Learning Tutorial}，
在其中我们会向你展示如何使用 tf.estimator API 结合线性模型与深度神经网络优势去训练模型。
