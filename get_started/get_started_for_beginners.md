# 机器学习新手入门

该文档阐述了如何使用机器学习，对鸢尾花的种属（Iris flowers Dataset）进行分类，深入 TensorFlow 源码，阐述机器学习基本原理。

如果你符合下列三个条件，就继续看下去吧：

*   或多或少听说过机器学习
*   想学习编写 TensorFlow 程序
*   会使用 Python 编程

如果你已经熟悉基础的机器学习概念，只是 TensorFlow 新手，建议移步 @{$premade_estimators$Getting Started with TensorFlow: for ML Experts}。

## 鸢尾花分类问题

假设你是一个植物学家，想将鸢尾花自动分类。机器学习提供多种分类算法。比如，优秀的分类算法通过图像识别对花进行分类。而我们不想止步于此，我们想要在仅知道花瓣、花萼的长度以及宽度的情况下对花进行分类。

鸢尾花专家能识别出 300 多个花种，不过我们的程序目前在以下三种中进行分类：

*   setosa 类
*   virginica 类
*   versicolor 类

<div style="margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%"
  alt="三种鸢尾花呈现出的不同花瓣花萼外形"
  src="../images/iris_three_species.jpg">
</div>

**从左至右，[*Iris setosa*](https://commons.wikimedia.org/w/index.php?curid=170298) (by [Radomil](https://commons.wikimedia.org/wiki/User:Radomil), CC BY-SA 3.0)，[*Iris versicolor*](https://commons.wikimedia.org/w/index.php?curid=248095) (by [Dlanglois](https://commons.wikimedia.org/wiki/User:Dlanglois), CC BY-SA 3.0) 和 [*Iris virginica*](https://www.flickr.com/photos/33397993@N05/3352169862) (by [Frank Mayfield](https://www.flickr.com/photos/33397993@N05)，CC BY-SA 2.0)。**
<p>&nbsp;</p>

我们找来 [Iris 数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set)，包含 120 条带有花萼、花瓣测量的数据。该数据集非常典型，是机器学习分类问题很好的入门材料。([MNIST 数据集](https://en.wikipedia.org/wiki/MNIST_database)，包含大量手写数字，也是分类问题的典型常用数据)。

Iris 数据集的前 5 行如下：

| 花萼长度 | 花萼宽度 | 花瓣长度 | 花瓣宽度 | 种属
| ---          | ---         | ---          | ---         | ---
|6.4           | 2.8         | 5.6          | 2.2         | 2
|5.0           | 2.3         | 3.3          | 1.0         | 1
|4.9           | 2.5         | 4.5          | 1.7         | 2
|4.9           | 3.1         | 1.5          | 0.1         | 0
|5.7           | 3.8         | 1.7          | 0.3         | 0

我们首先介绍一些术语：

*   最后一列 (种属) 被称为 [**标记**](https://developers.google.com/machine-learning/glossary/#label)（label）；前四列被称为 [**特征**](https://developers.google.com/machine-learning/glossary/#feature)（feature）。**特征**用来形容样本数据，**标记**用于之后的结果预测。

*   一个 [**样本**](https://developers.google.com/machine-learning/glossary/#example)（example）包含所有特征的集合和样本的标记。上表中，5 条样本数据来自于一个数据量为 120 条数据的数据集。

每个标记都是一个字符串（例如，“setosa”），但由于机器学习通常使用数字，因而我们将每个字符串与数字相对应，对应范式如下：

* 0 对应 setosa
* 1 对应 versicolor
* 2 对应 virginica

## 模型训练

**模型**（model）可以看作是特征与标记之间的关系。在鸢尾花问题中，模型定义了花萼花瓣测量数据与花种属之间的关系。有时短短几行代数符号就可以描述一个简单的模型；而有些复杂的模型包含大量的数学符号与复杂的变量关系，很难数字化表达。

现在问题来了：四个特征，一个花种属标记，你能在不使用机器学习的情况下，定义它们之间的关系么？换句话问，你能使用传统的程序语言（比如大量诸如 if/else 的条件语句）来创建模型么？有这个可能。如果你有大把的时间研究数据集，最终也许会找到花萼花瓣与花种属之间的关系。然而，一个好的机器学习算法能够为你预测模型。只要你有足够数量的，足够有代表性的数据，套用适当的模型，最终程序会帮你完美定义花种属与花萼花瓣的关系。

**训练** （training）是监督式机器学习的一个阶段，是模型逐渐优化（自我学习）的过程。
鸢尾花问题是 [**监督式学习**](https://developers.google.com/machine-learning/glossary/#supervised_machine_learning) 的一个典型，这类模型通过标记的样本数据训练得出。
还有一类机器学习：[**无监督式学习**](https://developers.google.com/machine-learning/glossary/#unsupervised_machine_learning)。这类样本模型是未标记的，模型只通过特征寻找规律。）

## 运行示例程序前的准备工作

在运行示例程序前，先安装 TensorFlow：

1.  @{$install$Install TensorFlow}
2.  如果你是使用 virtualenv 或 Anaconda 安装 TensorFlow 的，初始化 TensorFlow 环境。
3.  安装/升级 pandas :

     `pip install pandas`

按照以下步骤，找到示例程序：

1. 将 TensorFlow 模型 远程仓库从 github 克隆到本地，命令如下：

       `git clone https://github.com/tensorflow/models`

2. 在该分支下，cd 到包含本文示例代码的目录下：

       `cd models/samples/core/get_started/`

在 `get_started` 文件目录下，找到名为 `premade_estimator.py`的 python 文件。

## 运行示例程序

像运行 Python 程序一样运行 TensorFlow 程序。在命令行敲如下命令运行 `premade_estimators.py`：

``` bash
python premade_estimator.py
```

运行程序后会输出一大堆信息，结尾 3 行是预测结果，如下：

```None
...
Prediction is "Setosa" (99.6%), expected "Setosa"

Prediction is "Versicolor" (99.8%), expected "Versicolor"

Prediction is "Virginica" (97.9%), expected "Virginica"

```
如果程序报错，没有生成预测结果。查看以下问题：

* 是否成功安装 TensorFlow ？
* 是否使用了正确版本的 TensorFlow ？程序`premade_estimators.py`需要版本号至少为 TensorFlow v1.4。
* 如果你通过 virtualenv 或 Anaconda 安装的 TensorFlow，是否初始化环境？

## TensorFlow 技术栈

如下图所示，TensorFlow 技术栈提供了多层 API

<div style="margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/tensorflow_programming_environment.png">
</div>

**TensorFlow 编程环境**
<p>&nbsp;</p>

在开始写 TensorFlow 程序时，我们强烈建议您使用下列两类高层 API：

*   Estimators
*   Datasets

尽管我们偶尔需要使用到其它底层 API ，这篇文档将主要介绍这两类 API。

## 程序代码

有耐心看到这里的读者，来，我们继续深挖代码。和大部分 TensorFlow 程序相似，如下是`premade_estimator.py`程序的例行步骤：

*   引入数据集并解析
*   创建特征列描述数据
*   选择模型
*   训练模型
*   评估模型
*   使用训练后的模型进行预测。

下面各小节展开解释。

### 引入数据集并解析

鸢尾花问题需要引入下列两个 csv 文件的数据：

*   训练数据集` http://download.tensorflow.org/data/iris_training.csv`
*   测试数据集`  http://download.tensorflow.org/data/iris_test.csv`

**训练数据集** 包含用来训练模型的样本；**测试数据集** 包含用来评估模型的样本。

训练数据集和测试数据集在最开始是在同一个数据集中，后来该样本数据集被处理：其中的大部分作为训练数据、剩余部分作为测试数据。增加训练集样本数量通常能构造出更好的模型，而增加测试集样本的数量能够更好的评估模型。

`premade_estimators.py` 程序通过 `load_data` 函数读取相邻路径的 [`iris_data.py`](https://github.com/tensorflow/models/blob/master/samples/core/get_started/iris_data.py) 文件并解析为训练集和测试集。

代码如下（包含详细注释）

```python
# 定义数据 csv 文件地址
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

...

def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # 新建路径本地训练集文件
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    # 训练集路径为: ~/.keras/datasets/iris_training.csv

    # 解析本地 CSV 文件
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # 列
                        header=0  # 忽略 CSV 文件首行
                       )
    # 定义 train 变量为 DataFrame（pandas 库中类似表的数据结构）。

    # 1. 定义变量 train_label 为样本标记，DataFrame 的最右行，
    # 2. 从 DataFrame 中删除最右行，
    # 3. 定义 DataFrame 中的剩余行为 train_features 样本特征。
    train_features, train_label = train, train.pop(label_name)

    # 对测试数据集执行上述操作
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # 返回解析好的 DataFrame
    return (train_features, train_label), (test_features, test_label)
```

Keras 是一个开源机器学习库；`tf.keras` 是 TensorFlow 对 Keras 的实现。`premade_estimator.py` 程序只是 `tf.keras` 的一个函数入口，即： `tf.keras.utils.get_file` 方法，使拷贝远程 CSV 文件到本地系统更便捷。

调用 `load_data` 函数返回值为两组 `(feature,label)` 对，两组数据相对应训练集和测试集：

```python
    # 调用 load_data() 解析 CSV 文件
    (train_feature, train_label), (test_feature, test_label) = load_data()
```

Pandas 是一个开源的 Python 库，被用于 TensorFlow 函数中。Pandas 的[**DataFrame**](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) 是类似表的数据结构，每一列有列头，每一行有行标。下例为 `test_feature` DataFrame。

```none
    SepalLength  SepalWidth  PetalLength  PetalWidth
0           5.9         3.0          4.2         1.5
1           6.9         3.1          5.4         2.1
2           5.1         3.3          1.7         0.5
...
27          6.7         3.1          4.7         1.5
28          6.7         3.3          5.7         2.5
29          6.4         2.9          4.3         1.3
```

### 描述数据

**特征列** 可以看作是一个数据结构，为你的模型解释每一个特征的数据。在鸢尾花问题中，我们想让模型将每一特征按照字面浮点值解释。就是说，我们希望模型将 5.4 这样的输入值直接解析为，呃，5.4。而在某些机器学习问题中，我们喜欢将数据解析地不那么直接。特征列数据解释是一个很深的话题，我们在另一篇文档 @{$feature_columns$document} 中整篇描述。

从代码中来看，通过调用 @{tf.feature_column} 模块函数创建了一个 `feature_column` 对象列表。每个对象描述了模型的一个输入。我们想要模型以浮点数值解释数据，可以调用 @{tf.feature_column.numeric_column) 函数。在 `premade_estimator.py`中，四列特征被直接解释为字面浮点数值，程序创建了特征列如下：

```python
# 为所有特征创建特征列
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

下面代码不那么优雅，但更清楚地编码了上述过程，

```python
my_feature_columns = [
    tf.feature_column.numeric_column(key='SepalLength'),
    tf.feature_column.numeric_column(key='SepalWidth'),
    tf.feature_column.numeric_column(key='PetalLength'),
    tf.feature_column.numeric_column(key='PetalWidth')
]
```

### 选择模型类型

接下来我们需要选择要训练的模型类型。模型有很多，但找到最理想的模型需要一定经验。我们选择神经网络解决鸢尾花问题。
通过 [**神经网络**](https://developers.google.com/machine-learning/glossary/#neural_network) 可以找到特征和标记间的复杂关系。神经网络是一个高度结构化的图，组成了一个或多个 [**隐藏层**](https://developers.google.com/machine-learning/glossary/#hidden_layer)。每个隐藏层包含一个或多个 [**神经元**](https://developers.google.com/machine-learning/glossary/#neuron)。神经网络有不同的类别。这里我们使用 [**全连接神经网络**](https://developers.google.com/machine-learning/glossary/#fully_connected_layer)，就是说：每一层中神经元的输入，来自于上一层的 **所有** 神经元。举个例子，下图阐述了全连接神经网络，它包含 3 个隐藏层：

*   第一层有 4 个神经元，
*   第二次有 3 个神经元，
*   第三层有 2 个神经元。

<div style="margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/simple_dnn.svg">
</div>

**包含 3 个隐藏层的神经网络**
<p>&nbsp;</p>

我们通过实例化一个 [**Estimator**](https://developers.google.com/machine-learning/glossary/#Estimators) 类来指定模型类型。TensorFlow 提供两类 Estimator：

*   [**预定义 Estimator**](https://developers.google.com/machine-learning/glossary/#pre-made_Estimator)，代码已经由他人写好。
*   [**定制 Estimator**](https://developers.google.com/machine-learning/glossary/#custom_estimator)，你需要或多或少自己写代码。

为了实现这个神经网络，`premade_estimators.py`程序使用预定义 Estimator @{tf.estimator.DNNClassifier}，构建神经网络将样本分类。接下来调用一个实例化的`DNNClassifier`。

```python
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3)
```

使用 `hidden_units` 参数定义每一隐藏层中神经元的数量。赋值该参数一个列表。如下：

```python
        hidden_units=[10, 10],
```

`hidden_units`列表的长度即隐藏层数（此处为 2 层）。列表中的每一个数值代表着该层神经元的个数（此处第一层有 10 个神经元，第二层有 10 个神经元）。只需简单地改变`hidden_units`的列表参数，就可以调试隐藏层数或神经元的个数。

理想的层数/神经元数量是由数据集或问题本身决定的。正如同机器学习领域的其它方方面面，选择好神经网络的形状，需要大量实验和多方面的知识储备。根据经验法则，增加隐藏层数量/神经元数量*往往*能构造更强大的模型，这需要更多数据的有效训练。

参数规定了神经网络预测可能值的数量。由于该问题中对 3 中鸢尾花进行分类，我们设置`n_classes`为 3。

`tf.Estimator.DNNClassifier` 的构造函数有一个可选参数 `optimizer` 优化器，在这里我们的程序没有声明。[**优化器**](https://developers.google.com/machine-learning/glossary/#optimizer) 控制着模型怎样训练。当你在机器学习领域深入，优化器和[**学习率 **](https://developers.google.com/machine-learning/glossary/#learning_rate) （learning rate）将会变的很重要。

### 训练模型

实例化 `tf.Estimator.DNNClassifier` 搭建了一个学习模型的框架。抽象来说，我们织好了一张网络，但还没有载入数据。

现在通过调用 estimator 对象的 `train` 方法训练神经网络。如下：

```python
    classifier.train(
        input_fn=lambda:train_input_fn(train_feature, train_label, args.batch_size),
        steps=args.train_steps)
```

`steps`参数值指：通过多少次迭代后停止模型训练。`steps` 参数越大，意味着训练模型的时间越长。但训练模型时间越长，并不意味着模型更好。`args.train_steps` 的缺省值为 1000，训练的步骤数是一个可以调优的[**超参数**](https://developers.google.com/machine-learning/glossary/#hyperparameter)。选择恰当的步骤数往往需要大量经验实践的积累。

`input_fn` 参数赋值为获得训练数据的函数，train 方法的调用通过 `train_input_fn` 函数获得训练数据。下面是该函数签名：

```python
def train_input_fn(features, labels, batch_size):
```

给 `train_input_fn` 传入下列参数值：

* `train_feature` 是一个 Python 的字典，该字典中：
    * key 为样本特征名，
    * value 为一个包含训练集所有样本值的数组
* `train_label` 为一个包含训练集所有样本标记的数组
* `args.batch_size` 数据类型为整型，定义了[**批量大小**](https://developers.google.com/machine-learning/glossary/#batch_size)。

`train_input_fn` 函数依赖于 **Dataset API**。这是一个高层 TensorFlow API，用于读取数据并转化成 `train` 方法所需的格式。
下面的函数调用将输入的特征和标记转化为一个 `tf.data.Dataset` 对象，Dataset API的基类:

```python
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
```

`tf.dataset` 类给训练提供了许多有用的预备样本。比如下面 3 个函数:

```python
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
```

随机的训练样本会使训练效果更好。通过函数 `tf.data.Dataset.shuffle` 将样本随机化，设置 `buffer_size` 值大于样本数量（120）以确保数据洗牌效果。

训练过程中，`train` 方法通常要多次处理样本。不带参数调用 `tf.data.Dataset.repeat` 使 `train` 方法有无穷的（通过不断随机化过程模拟）训练样本集。

`train` 方法每次[**批量**](https://developers.google.com/machine-learning/glossary/#batch)处理样本，都通过`tf.data.Dataset.batch`方法串联多个样本创建一个批处理。我们程序中设置默认 [**批量大小**](https://developers.google.com/machine-learning/glossary/#batch_size) 为 100，意味着 `batch` 方法串联几组数量为 100 的样本。理想的批量大小取决于问题本身，根据经验法则，小批量往往可以使 `train` 方法更快地训练模型，但有时候要付出准确率下降的代价。

 `return` 返回一批样本给调用方法（`train` 方法）。

```python
   return dataset.make_one_shot_iterator().get_next()
```

### 评估模型

**评估** 用来判断模型预测结果的有效性。为了评价鸢尾花分类模型的有效性，我们向模型传入一些花瓣花萼的测量值，让其预测传入数据的花种属，然后对比模型的预测结果与实际标记。举例说明，模型若能够预测正确一半的样本数据，则[准确率](https://developers.google.com/machine-learning/glossary/#accuracy)为 0.5。下面例子展示了一个更有效的模型：

<table>
  <tr>
    <th style="background-color:darkblue" colspan="5">
       测试集</th>
  </tr>
  <tr>
    <th colspan="4">特征</th>
    <th colspan="1">标记</th>
    <th colspan="1">预测</th>
  </tr>
  <tr> <td>5.9</td> <td>3.0</td> <td>4.3</td> <td>1.5</td> <td>1</td>
          <td style="background-color:green">1</td></tr>
  <tr> <td>6.9</td> <td>3.1</td> <td>5.4</td> <td>2.1</td> <td>2</td>
          <td style="background-color:green">2</td></tr>
  <tr> <td>5.1</td> <td>3.3</td> <td>1.7</td> <td>0.5</td> <td>0</td>
          <td style="background-color:green">0</td></tr>
  <tr> <td>6.0</td> <td>3.4</td> <td>4.5</td> <td>1.6</td> <td>1</td>
          <td style="background-color:red">2</td></tr>
  <tr> <td>5.5</td> <td>2.5</td> <td>4.0</td> <td>1.3</td> <td>1</td>
          <td style="background-color:green">1</td></tr>
</table>

**该模型有 80% 正确率**
<p>&nbsp;</p>

为了评估模型的有效性，每个 estimator 都提供了 `evaluate` 方法。`premade_estimator.py` 程序中调用 `evaluate` 如下：

```python
# 评估模型
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```

调用 `classifier.evaluate` 和 `classifier.train` 类似。最大的区别在于`classifier.evaluate` 需要从测试数据集获取数据，而非训练数据集。换句话说，为了公平地评估模型的有效性，用来*评估*模型的样本和用于*训练*的样本必需不同。我们通过调用 `eval_input_fn` 函数处理了测试集的一批样本。如下：

```python
def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is None:
        # 无标记，仅使用特征
        inputs = features
    else:
        inputs = (features, labels)

    # 转换输入为 tf.dataset 对象
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # 批量处理样本
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # 返回流程的读结尾
    return dataset.make_one_shot_iterator().get_next()
```

简单来说，`eval_input_fn` 在调用 `classifier.evaluate` 函数时做了以下步骤：

1.  处理测试集数据，将特征和标记转化为 `tf.dataset` 对象。
2.  创建一批测试集样本（测试集样本不需要洗牌或重复随机化）。
3.  返回测试集样本给 `classifier.evaluate`。

执行代码得出类似下面的输出：

```none
Test set accuracy: 0.967
```

准确率 0.967 意味着：我们训练出的模型能将测试集里 30 个鸢尾花样本中的 29 个正确分类。

### 预测

现在我们训练好模型，而且“证明”了在鸢尾花分类问题中它还不错，虽然并不完美。现在我们用训练的模型在[**无标记样本**](https://developers.google.com/machine-learning/glossary/#unlabeled_example)（没有标记仅有特征的样本）上做预测；

在实际生活中，无标记的样本来自不同来源：应用中，CSV 文件，数据流等。不过现在我们简单起见，人造下面几个无标记样本：

```python
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
```

每个 estimator 提供一个提供一个 `predict` 方法，`premade_estimator.py` 这样调用：

```python
predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x, batch_size=args.batch_size))
```

同 `evaluate` 方法一样，`predict` 方法通过 `eval_input_fn` 收集样本。

预测时，我们**不**传标记给 `eval_input_fn`，而是做如下步骤：

1.  将我们刚刚人造的 3-元素 数据集特征转换。
2.  从刚才的数据集中创建批量的 3 个样本。
3.  返回批量的样本给 `classifier.predict`。

`predict` 方法返回了一个 python iterable 对象，以字典结构输出每个样本的预测结果。该字典包含多个键值对。`probabilities` 的值是一个包含 3 个浮点值的列表，每个浮点值代表输入样本是该鸢尾花种属的可能性。例如，下面这个 `probabilities` 列表：

```none
'probabilities': array([  1.19127117e-08,   3.97069454e-02,   9.60292995e-01])
```

该列表表明：

*   该鸢尾花样本是 Setosa 的概率忽略不计。
*   有 3.97% 概率为 Versicolor 类。
*   有 96.0% 概率为 Virginica 类。

`class_ids` 的值为仅有一个元素的数组，表明该样本最有可能是哪个种类：

```none
'class_ids': array([2])
```

第 `2` 类对应 Virginica 类鸢尾花。下面代码迭代整个 `predictions` 并针对每个 `predictions` 生成报告：

``` python
for pred_dict, expec in zip(predictions, expected):
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print(template.format(SPECIES[class_id], 100 * probability, expec))
```

程序输出如下：

``` None
...
Prediction is "Setosa" (99.6%), expected "Setosa"

Prediction is "Versicolor" (99.8%), expected "Versicolor"

Prediction is "Virginica" (97.9%), expected "Virginica"
```

## 小结

<!--TODO(barryr): When MLCC is released, add pointers to relevant sections.-->
此文档提供一个机器学习的简短介绍。

由于 `premade_estimators.py` 依赖于高层 API，机器学习中大部分的复杂数学被隐藏。如果你想要深入学习机器学习，我们推荐学习[**梯度下降**](https://developers.google.com/machine-learning/glossary/#gradient_descent)，批量，还有神经网络。

推荐阅读 @{$feature_columns$Feature Columns} 文档，了解机器学习中的不同类别数据表述。
