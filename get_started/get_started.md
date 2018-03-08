# TensorFlow 入门指南

本指南可以让你在 TensorFlow 中进行编程。在开始本指南之前，请先[安装 TensorFlow](https://www.tensorflow.org/install/index)。为了能够更好的利用本指南，你应当了解如下知识：

*   如何使用 Python 编程
*   至少了解数组知识
*   理想情况下了解一些机器学习知识，当然，即使你不了解机器学习，本指南也会是你应该阅读的第一个指南。

TensorFlow 提供了多种 API。其中最底层的 API —— TensorFlow Core 能够为你提供完整的编程控制功能。我们向机器学习研究人员和其他需要对模型进行精准控制人员推荐使用 TensorFlow Core。更高级的 API 是构建在 TensorFlow Core 之上。这些更高级的 API 通常比 TensorFlow Core 更容易学习和使用。此外，这些更高级的 API 使得不同用户之间的重复性任务更加简单且更具有一致性。像 `tf.estimator` 这样高级的 API 能够帮你管理数据集、评估器、训练以及推理。

本指南将从 TensorFlow Core 开始介绍。然后我们会演示如何在 `tf.estimator` 中实现相同的模型。了解 TensorFlow Core 的原理，能够提供一个更好的经验模型，以便于当你使用更为简洁的高级 API 时了解其内部工作情况。

# 张量

TensorFlow 的核心数据单位就是**张量**。张量可看做是由原始值组成的任意维度数组。张量的**阶**就是它的维度。下面是一些关于张量的例子：

```python
3 # 0 阶张量，shape 为 [] 的标量
[1., 2., 3.] # 1 阶张量，shape 为 [3] 的向量
[[1., 2., 3.], [4., 5., 6.]] # 2 阶张量，shape 为 [2, 3] 的矩阵
[[[1., 2., 3.]], [[7., 8., 9.]]] # 3 阶张量，shape 为 [2, 1, 3] 的数据立体
```

## TensorFlow Core 教程

### 导入 TensorFlow

导入 TensorFlow 程序模块的语法格式如下：

```python
import tensorflow as tf
```
这使 Python 可以使用 TensorFlow 所有的类、方法和符号。绝大部分文档都会假定你已经完成了导入的部分。

### 计算图

你可以认为 TensorFlow Core 程序由以下两个独立部分组成：

1.  创建计算图
2.  运行计算图

**计算图**是一系列 TensorFlow 运算作为节点组成的节点图。我们来构建一个简单的计算图。每个节点采用零个或者多个张量作为输入，并产生一个张量作为输出。常量就是一种节点类型，就像所有的 TensorFlow 常量一样，它不需要任何输入，然后它会输出一个内部存储的值。我们可以创建两个浮点型张量节点 `node1` 和 `node2`，如下所示：

```python
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # 如果未设置 dtype，dtype 的默认值为 tf.float32
print(node1, node2)
```

最终打印结果是：

```
Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
```

请注意，打印出的节点结果并没有像你预期那样输出值是 `3.0` 和 `4.0`。取而代之的是，打印出的是两个节点，这两个节点被计算后会分别产生 3.0 和 4.0。要实际的计算节点，我们必须使用 **会话（session）** 来运行计算图。会话封装了 TensorFlow 运行时的控制和状态。

下面的代码创建了一个 `Session` 对象，然后调用其 `run` 方法运行足够的计算图来计算 `node1` 和 `node2`。使用一个会话来运行计算图的代码如下：

```python
sess = tf.Session()
print(sess.run([node1, node2]))
```

我们看到了期望的输出值 3.0 和 4.0：

```
[3.0, 4.0]
```

我们可以通过将 `张量（Tensor）` 节点
和操作（操作也是节点）相结合来构建更为复杂的计算。举个例子，我们可以将两个常量节点相加然后生成一个新的图，如下：

```python
from __future__ import print_function
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))
```

最后两个输出语句将生成：

```
node3: Tensor("Add:0", shape=(), dtype=float32)
sess.run(node3): 7.0
```

TensorFlow 提供了一个叫做 TensorBoard 的实用程序，通过它可以显示计算图的图片。下面的截图显示了 TensorBoard 如何可视化计算图：

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_add.png)

事实上，这样的图并不是很有趣，因为它总是产生一个常量结果。一个图可以被参量化成接受外部输入，称之为**占位符**。一个**占位符**随后会被输入中的一个值所替代。

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # 加号 `+` 提供了一个进行 tf.add(a, b) 的简单方法
```

前面三行有点像一个函数或者 lambda 表达式，在其中我们定义了两个输入参量（a 和 b），然后对它们进行操作。我们可以通过使用 [run 方法](https://www.tensorflow.org/api_docs/python/tf/Session#run)的参数 feed_dict 来对计算图进行多值输入，为这些占位符提供具体的值来进行计算图的求值。

```python
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))
```
输出结果：

```
7.5
[ 3.  7.]
```

在 TensorBoard 中，图形如下：

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_adder.png)

我们可以通过添加另外的操作来让计算图更加复杂。例如：

```python
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))
```
输出结果：
```
22.5
```

前面的计算图在 TensorBoard 会像如下所示：

![TensorBoard screenshot](https://www.tensorflow.org/images/getting_started_triple.png)

在机器学习里，我们通常会想要一个可以接受任意输入的模型，比如上面那一个。为了让模型可训练，我们需要对计算图进行修改，以便于达到相同输入下有新的输出结果。**变量**允许我们向计算图中加入可训练的参数。它们可以通过给定初始值和类型进行构造，方法如下: ：


```python
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
```

常量使用 `tf.constant` 进行初始化，其值永远不会改变。相比之下，使用 `tf.Variable` 并不会初始化变量。要在 TensorFlow 程序中初始化所有变量，你必须显式的调用下面的特定操作：

```python
init = tf.global_variables_initializer()
sess.run(init)
```
重要的是要认识到 `init` 是一个 TensorFlow 子图的一个句柄，它可以初始化所有全局变量。注意，我们调用 `sess.run` 后，变量才会被初始化。


既然 `x` 是占位符，我们可以利用线性模型（`linear_model`）同时对 `x` 的多个值进行计算，如下：

```python
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
```
产生输出
```
[ 0.          0.30000001  0.60000002  0.90000004]
```

我们已经创建了一个模型，但是我们不知道它的性能。为了在训练数据集上评估模型，我们需要一个 `y` 占位符来提供所需的值，然后我们需要编写一个损耗函数。

损耗函数用于衡量当前模型和提供的数据集之间的偏离程度。我们将使用线性回归的标准损耗模型，它会将当前模型和提供的数据集之间的误差平方求和。`linear_model - y` 创建一个向量，其中每个元素都是对应实例的误差值。我们调用 `tf.square` 来平方这些误差，然后我们将所有平方后的误差进行求和，以此来创建一个单量，并使用 `tf.reduce_sum` 来将所有实例的误差提取出来：

```python
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
```
产生损耗值：
```
23.66
```

我们可以通过手动将 `W` 和 `b` 的值重新赋值为最优的 -1 和 1 来改善损耗函数的结果。变量可以初始化为由 `tf.Variable` 所提供的值，但也可以使用例如 `tf.assign` 操作来进行改变。例如，`W=-1`和 `b=1` 是我们模型的最优参数。我们可以相应的改变 `W` 和 `b`: 

```python
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
```
最终的输出显示现在的损耗为零:
```
0.0
```

我们猜测了“最优”的 `W` 和 `b` 值，但是整个机器学习的重点是自动寻找正确的模型参数。我们将会在下一节中展示如何完成这项工作。

## tf.train API

对于机器学习完整的讨论已经超出了本教程的范围。然而，TensorFlow 提供了**优化器**来缓慢地更改每个变量，从而最大程度的降低损耗函数。最简单的优化器就是**梯度下降**。它根据相对于变量的损耗函数导数的大小来修改每个变量。通常来说，手动计算函数导数是很乏味且易出错的。因此，TensorFlow 可以只根据模型描述用 `tf.gradients` 自动的生成导数。为了方便起见，优化器通常会帮助用户做这样的操作。例如，

```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
```

```python
sess.run(init) # 将数值重置为不正确的默认值
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
```
最终模型参数的结果：
```
[array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]
```

现在我们完成了真正的机器学习！尽管这只是简单的线性回归模型，它并不需要很多 TensorFlow 的核心代码，但是将数据输入到更复杂的模型和方法就会需要更多的代码。因此，TensorFlow 为常见模型、结构和函数提供了更高阶的抽象例程。我们将会在下一节学习如何使用这些抽象例程。

### 完整程序代码

训练线性回归模型的完整代码如下：

```python
import tensorflow as tf

# 模型参数
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# 模型输入和输出
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# 损耗
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# 优化器
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# 训练集数据
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# 训练循环
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# 计算训练的准确度
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```
运行之后，产生结果：
```
W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```

注意，损耗函数的值非常小（接近于零）。如果你运行这段代码，你的损耗可能不会和上述损耗结果一模一样，这是因为模型是由随机值来初始化的。

这段更复杂的代码依然可以在 TensorBoard 中可视化：：
![TensorBoard final model visualization](https://www.tensorflow.org/images/getting_started_final.png)

## `tf.estimator`

`tf.estimator` 作为 TensorFlow 高级库，简化了机器学习的机制，其中包括：

*   运行训练循环
*   运行评估循环
*   管理数据集

tf.estimator 定义了许多常见的模型。

### 基础用法

注意看 `tf.estimator` 使得线性回归程序变得更简单：

```python
# NumPy 经常被用来载入、操纵和预处理数据集
import numpy as np
import tensorflow as tf

# 声明特征列表。这里我们只有一组纯数字特征
# tf.feature_column 中还有许多其他类型的更复杂更有用的 column
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# 估算器是调用训练（拟合）和评估（推理）方法的前端
# 这里有许多预定义的函数类型，比如线性回归、线性分类
# 并且也包含许多神经网络分类器和回归器
# 下面的代码提供了一个线性回归的优化器
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)

# TensorFlow 提供了许多 `helper` 方法来读取和设置数据集
# 这里我们有两组数据集：一组是训练集，另一组是测试集用于评估
# 我们需要告诉函数我们想要设置多少组数据（num_epochs），并且每组数据的大小应该是多少
# 
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# 我们可以通过调用方法和传递训练数据集来进行 1000 次训练步骤
# 
estimator.train(input_fn=input_fn, steps=1000)

# 在这里我们评估我们的模型的性能
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```
运行之后，产生如下结果：
```
train metrics: {'average_loss': 1.4833182e-08, 'global_step': 1000, 'loss': 5.9332727e-08}
eval metrics: {'average_loss': 0.0025353201, 'global_step': 1000, 'loss': 0.01014128}
```
可以注意到，我们在评估数据集上有更高的损耗值，但是它仍然接近于零。这就说明我们的学习比较适合。

### 自定义模型

`tf.estimator` 并不会把你限制在它预定义的模型中。假设我们要创建一个未内建于 TensorFlow 的自定义模型。我们仍然能够使用 `tf.estimator` 中关于数据集、数据输入、训练等的高层抽象。为了演示这一切，我们将会展示如何利用低阶 TensorFlow API 的知识来实现一个我们自己的 `LinearRegressor` 等价模型。

定义一个和 `tf.estimator` 协同工作的自定义模型，我们需要使用 `tf.estimator.Estimator`。`tf.estimator.LinearRegressor` 实际上是 `tf.estimator.Estimator` 的子类。相比于实现一个 `Estimator` 的子类，我们只简单地通过给 `Estimator` 提供一个 `model_fn` 函数来告诉  `tf.estimator` 如何进行预测、训练和损耗函数的计算。代码如下：

```python
import numpy as np
import tensorflow as tf

# 声明特征列表，这里我们只有实值特征
def model_fn(features, labels, mode):
  # 创建线性模型和预测值
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # 损耗的子图
  loss = tf.reduce_sum(tf.square(y - labels))
  # 训练的子图
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # EstimatorSpec 连接我们所创建的子图到相关的函数
  # 
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.estimator.Estimator(model_fn=model_fn)
# 定义我们的数据集
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7., 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# 训练
estimator.train(input_fn=input_fn, steps=1000)
# 这里对我们模型的性能进行评估
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```
运行之后，结果如下：
```
train metrics: {'loss': 1.227995e-11, 'global_step': 1000}
eval metrics: {'loss': 0.01010036, 'global_step': 1000}
```

可以注意到，这个自定义 `model_fn()` 函数的代码和我们利用低阶 API 进行手动训练的模型（model）代码是多么地相似。

## 下一阶段

现在你已经了解了 TensorFlow 运行的基础知识。我们还有更多的教程，你可以查看并学习更多内容。如果你是机器学习的初学者，请阅读 [MNIST 机器学习入门](https://www.tensorflow.org/get_started/mnist/beginners)，否则请阅读[深入学习 MNIST](https://www.tensorflow.org/get_started/mnist/pros)。
