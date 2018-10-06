# Eager Execution

TensorFlow 的 eager execution 是一种可以立即评估操作，无需构建图的命令式编程环境；操作会返回具体值，而不是一个供之后运行的计算图。这降低了 TensorFlow 入门以及调试模型入门的门槛，同时也减少了文档范例。为了遵循本指南，请在 `python` 的交互式解释器中运行下述示例代码。

Eager execution 为实验和研究提供了一个灵活的机器学习平台：

* **直观的界面** — 为你合理地构建代码并使用 Python 数据结构。在小型模型和小型数据中可快速迭代。
* **更简单的调试** — 直接调用 ops 来检测运行模型已经测试更改。使用标准化 Python 调试工具进行即时错误报告。
* **合理的控制流** — 使用 Python 控制流来取代图控制流，简化了动态模型的规范。

Eager execution 支持大多数 TensorFlow 操作和 GPU 加速功能。如果需要 eager execution 运行的示例集合，请参阅： [tensorflow/contrib/eager/python/examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples)。

注意：一些模型在开启 eager execution 后，可能会增加开销。虽然已经在进行性能优化，但如果你发现了问题，请向我们提交[错误文件报告](https://github.com/tensorflow/tensorflow/issues)并分享你的基准测试。

## 安装和基本用法

升级到最新版本的 TensorFlow：

```
$ pip install --upgrade tensorflow
```

启用 eager execution 时，请在程序或者控制台会话的开头添加 `tf.enable_eager_execution()`。不要将此操作添加到程序将要调用的其他模块中。

```py
from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()
```

现在你可以运行 TensorFlow 操作了，结果会被立即返回： 

```py
tf.executing_eagerly()        # => True

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))  # => "hello, [[4.]]"
```

启用 eager execution 会改变 TensorFlow 操作的行为——现在它们会立即计算值并将结果返回给 Python。`tf.Tensor` 对象会为节点引用计算图中的实际值而不是符号句柄。由于在 session 中没有要构建和运行的计算图，因此使用 `print()` 或调试器来检查结果会很容易。计算、打印以及检查张量值不会中断计算梯度的流程。

Eager execution 可以和 [NumPy](http://www.numpy.org/) 完美结合。NumPy 运算可以接受来自 `tf.Tensor` 的参数。TensorFlow [数学操作](https://www.tensorflow.org/api_guides/python/math_ops)将 Python 对象和 NumPy 数组转换成 `tf.Tensor` 对象。`tf.Tensor.numpy` 方法将对象值作为 NumPy `ndarray` 返回。

```py
a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
# => tf.Tensor([[1 2]
#               [3 4]], shape=(2, 2), dtype=int32)

# Broadcasting support
b = tf.add(a, 1)
print(b)
# => tf.Tensor([[2 3]
#               [4 5]], shape=(2, 2), dtype=int32)

# Operator overloading is supported
print(a * b)
# => tf.Tensor([[ 2  6]
#               [12 20]], shape=(2, 2), dtype=int32)

# Use NumPy values
import numpy as np

c = np.multiply(a, b)
print(c)
# => [[ 2  6]
#     [12 20]]

# Obtain numpy value from a tensor:
print(a.numpy())
# => [[1 2]
#     [3 4]]
```

`tf.contrib.eager` 模块拥有在 eager 和 graph 执行环境中都可以使用的符号，并且对于[使用图](#work_with_graphs)的代码编写非常有用：

```py
tfe = tf.contrib.eager
```

## 动态控制流

eager execution 的一个主要优势是，在运行模型时，主机语言的所有功能都是可用的。因此，例如，编写 [fizzbuzz](https://en.wikipedia.org/wiki/Fizz_buzz) 是轻而易举的事情：

```py
def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(max_num.numpy()):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num)
    counter += 1
  return counter
```

这是有条件的，即依赖于张量值并且可以在运行时打印这些值。

## 构建模型

许多机器学习模型都是由组合网络层构成。在使用具有 eager execution 的 TensorFlow 时，你可以编写自己的网络层或者使用 `tf.keras.layers` 包提供的网络层。

尽管你可以使用任意的 Python 对象来表示网络层，但 TensorFlow 仍然有 `tf.keras.layers.Layer` 来作为便捷的基类。你可以通过继承它来实现自己的网络层：

```py
class MySimpleLayer(tf.keras.layers.Layer):
  def __init__(self, output_units):
    super(MySimpleLayer, self).__init__()
    self.output_units = output_units

  def build(self, input_shape):
    # 第一次使用 layer 时，会调用 `build` 方法。
    # 在 build() 上创建变量使其形状依赖于输入形状，从而消除了用户指定完整形状的需要，如果你已经知道变量的全部形状，则可以在 _init_() 期间创建变量。
    self.kernel = self.add_variable(
      "kernel", [input_shape[-1], self.output_units])

  def call(self, input):
    # 重载 call() 而不是 __call__ ，这样我们就可以执行一些 bookkeeping 操作。
    return tf.matmul(input, self.kernel)
```

使用 `tf.keras.layers.Dense` 层来替换上述的 `MySimpleLayer`，因为它有其功能的超集（它还可以添加偏差）。

将层合并为模型时，你可以使用 `tf.keras.Sequential` 来表示模型，这些模型是层的线性堆栈。这对于基础模型来说，很容易使用：

```py
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, input_shape=(784,)),  # must declare input shape
  tf.keras.layers.Dense(10)
])
```

它是一个网络层的容器，但它自己也是一个网络层，`tf.keras.Model` 对象可以包含其他的 `tf.keras.Model` 对象。

```py
class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=10)
    self.dense2 = tf.keras.layers.Dense(units=10)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result

model = MNISTModel()
```

无需为 `tf.keras.Model` 类设置 input shape，因为参数是 input 第一次传递给层时设置的。

`tf.keras.layers` 类创建并包含与其对象层生命周期相关联的模型变量。为了共享变量层，请共享它们的对象。


## Eager 训练

### 计算梯度

[自动微分法](https://en.wikipedia.org/wiki/Automatic_differentiation)在训练神经网络方面中，对于实现像[向后传播](https://en.wikipedia.org/wiki/Backpropagation)这样的机器学习算法是非常有用的。在 eager execution 中，使用 `tf.GradientTape` 来跟踪之后的梯度计算操作。

在不进行跟踪时，`tf.GradientTape` 是一个提供最佳性能的可选特性。因为每次调用都会发生不同的操作，因此所有向前传播操作都会被记录在一个 "tape" 中。为了计算梯度，需要向后播放 tape，然后丢弃。一个特定的 `tf.GradientTape` 只能计算一个梯度，之后的调用会导致运行时错误。

```py
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)
```

这是 `tf.GradientTape` 在训练简单模型时记录向前传播操作的一个示例:

```py
# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

def prediction(input, weight, bias):
  return input * weight + bias

# A loss function using mean-squared error
def loss(weights, biases):
  error = prediction(training_inputs, weights, biases) - training_outputs
  return tf.reduce_mean(tf.square(error))

# Return the derivative of loss with respect to weight and bias
def grad(weights, biases):
  with tf.GradientTape() as tape:
    loss_value = loss(weights, biases)
  return tape.gradient(loss_value, [weights, biases])

train_steps = 200
learning_rate = 0.01
# Start with arbitrary values for W and B on the same batch of data
W = tf.Variable(5.)
B = tf.Variable(10.)

print("Initial loss: {:.3f}".format(loss(W, B)))

for i in range(train_steps):
  dW, dB = grad(W, B)
  W.assign_sub(dW * learning_rate)
  B.assign_sub(dB * learning_rate)
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

print("Final loss: {:.3f}".format(loss(W, B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))
```

Output (exact numbers may vary):

```
Initial loss: 71.204
Loss at step 000: 68.333
Loss at step 020: 30.222
Loss at step 040: 13.691
Loss at step 060: 6.508
Loss at step 080: 3.382
Loss at step 100: 2.018
Loss at step 120: 1.422
Loss at step 140: 1.161
Loss at step 160: 1.046
Loss at step 180: 0.996
Final loss: 0.974
W = 3.01582956314, B = 2.1191945076
```

重放 `tf.GradientTape` 来计算梯度并应用在循环训练中。这在 [mnist_eager.py](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_eager.py) 中有演示示例：

```py
dataset = tf.data.Dataset.from_tensor_slices((data.train.images,
                                              data.train.labels))
...
for (batch, (images, labels)) in enumerate(dataset):
  ...
  with tf.GradientTape() as tape:
    logits = model(images, training=True)
    loss_value = loss(logits, labels)
  ...
  grads = tape.gradient(loss_value, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
```

下述示例创建了一个对标准 MNIST 手写体数字进行了分类的多层模型。它演示了在 eager execution 环境中如何利用优化器和网络层 API 来构建可训练图。

### 训练模型

即使没有进行训练，也可以调用模型，并在 eager execution 中检查输出：

```py
# Create a tensor representing a blank image
batch = tf.zeros([1, 1, 784])
print(batch.shape)  # => (1, 1, 784)

result = model(batch)
# => tf.Tensor([[[ 0.  0., ..., 0.]]], shape=(1, 1, 10), dtype=float32)
```

本示例使用 [TensorFlow MNIST example](https://github.com/tensorflow/models/tree/master/official/mnist) 中的 [dataset.py module](https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py)；将本文件下载到你的本地目录。运行以下内容将 MNIST 数据文件下载到你的工作目录，并为训练准备一个 `tf.data.Dataset`：

```py
import dataset  # download dataset.py file
dataset_train = dataset.train('./datasets').shuffle(60000).repeat(4).batch(32)
```

为了训练模型，为优化定义一个损失函数并计算梯度。使用优化器进行更新变量：

```py
def loss(model, x, y):
  prediction = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=prediction)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

x, y = iter(dataset_train).next()
print("Initial loss: {:.3f}".format(loss(model, x, y)))

# 循环训练
for (i, (x, y)) in enumerate(dataset_train):
  # 根据参数计算输入函数的导数。
  grads = grad(model, x, y)
  # 对模型应用梯度
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
  if i % 200 == 0:
    print("Loss at step {:04d}: {:.3f}".format(i, loss(model, x, y)))

print("Final loss: {:.3f}".format(loss(model, x, y)))
```

Output（确切的数字可能会发生偏差）：

```
Initial loss: 2.674
Loss at step 0000: 2.593
Loss at step 0200: 2.143
Loss at step 0400: 2.009
Loss at step 0600: 2.103
Loss at step 0800: 1.621
Loss at step 1000: 1.695
...
Loss at step 6600: 0.602
Loss at step 6800: 0.557
Loss at step 7000: 0.499
Loss at step 7200: 0.744
Loss at step 7400: 0.681
Final loss: 0.670
```

为了进行更快速的训练，将计算转移到 GPU 中：

```py
with tf.device("/gpu:0"):
  for (i, (x, y)) in enumerate(dataset_train):
    # minimize() is equivalent to the grad() and apply_gradients() calls.
    optimizer.minimize(lambda: loss(model, x, y),
                       global_step=tf.train.get_or_create_global_step())
```

### 变量和优化器

`tf.Variable` 对象存储在训练时可以访问的可变 `tf.Tensor` 值来让自动微分更加简单。模型参数可以作为变量封装在类中。 

使用结合 `tf.GradientTape` 的 `tf.Variable` 可以更好的封装模型参数。例如，可以重写上述的自动微分示例：

```py
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tf.Variable(5., name='weight')
    self.B = tf.Variable(10., name='bias')
  def call(self, inputs):
    return inputs * self.W + self.B

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# The loss function to be optimized
def loss(model, inputs, targets):
  error = model(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, [model.W, model.B])

# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# Training loop
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, [model.W, model.B]),
                            global_step=tf.train.get_or_create_global_step())
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))
```

Output（确切的数字可能会有所不同）：

```
Initial loss: 69.066
Loss at step 000: 66.368
Loss at step 020: 30.107
Loss at step 040: 13.959
Loss at step 060: 6.769
Loss at step 080: 3.567
Loss at step 100: 2.141
Loss at step 120: 1.506
Loss at step 140: 1.223
Loss at step 160: 1.097
Loss at step 180: 1.041
Loss at step 200: 1.016
Loss at step 220: 1.005
Loss at step 240: 1.000
Loss at step 260: 0.998
Loss at step 280: 0.997
Final loss: 0.996
W = 2.99431324005, B = 2.02129220963
```

## 在使用 eager execution 环境时将对象应用于状态中

在 graph execution 中，程序状态（例如变量）被存储在全局集合中，它们的生命周期由 `tf.Session` 对象管理。与之相反的是，在 eager execution 中，状态对象的生命周期则是由与它们对应的 Python 对象的生命周期决定的。

### 变量即对象

在 eager execution 环境中，变量会持续存在，直到对象的最后一个引用被删除，然后变量才会被删除。

```py
with tf.device("gpu:0"):
  v = tf.Variable(tf.random_normal([1000, 1000]))
  v = None  # v no longer takes up GPU memory
```

### 基于对象的保存方式

`tf.Checkpoint` 可以对检查点进行保存并恢复 `tf.Variable`。

```py
x = tf.Variable(10.)

checkpoint = tf.Checkpoint(x=x)  # save as "x"

x.assign(2.)   # Assign a new value to the variables and save.
save_path = checkpoint.save('./ckpt/')

x.assign(11.)  # Change the variable after saving.

# Restore values from the checkpoint
checkpoint.restore(save_path)

print(x)  # => 2.0
```

在保存并加载模型时，无需请求隐藏变量，`tf.Checkpoint` 便可以存储对象的内部状态。想要记录 `model`、`optimizer` 的状态和全局步骤，只需将它们传递给 `tf.Checkpoint`：

```py
model = MyModel()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
checkpoint_dir = ‘/path/to/model_dir’
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.train.get_or_create_global_step())

root.save(file_prefix=checkpoint_prefix)
# or
root.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

### 面向对象度量

`tfe.metrics` 作为对象被存储。通过将新数据传递给可调用对象来更新度量标准，并使用 `tfe.metrics.result` 方法来检索结果，例如：

```py
m = tfe.metrics.Mean("loss")
m(0)
m(5)
m.result()  # => 2.5
m([8, 9])
m.result()  # => 5.5
```

#### Summary 以及 TensorBoard

[TensorBoard](../guide/summaries_and_tensorboard.md) 是一个用于帮助理解、调试和优化模型训练过程的可视化工具。它使用在执行程序时生成的 summary event 进行可视化。

`tf.contrib.summary` 同时兼容 eager 和 graph 执行环境。Summary 操作，例如 `tf.contrib.summary.scalar`，是在模型构建时插入的。例如，每执行 100 个全局的 step，就记录一次 summary：

```py
global_step = tf.train.get_or_create_global_step()
writer = tf.contrib.summary.create_file_writer(logdir)
writer.set_as_default()

for _ in range(iterations):
  global_step.assign_add(1)
  # Must include a record_summaries method
  with tf.contrib.summary.record_summaries_every_n_global_steps(100):
    # your model code goes here
    tf.contrib.summary.scalar('loss', loss)
     ...
```

## 高级自动微分主题

### 动态模型

`tf.GradientTape` 也可以用于动态模型。这个用于[回溯行搜索](https://wikipedia.org/wiki/Backtracking_line_search)算法的示例看起来像普通的 NumPy 代码，尽管有着复杂的控制流，但它确实存在梯度并且是可微分的。

```py
def line_search_step(fn, init_x, rate=1.0):
  with tf.GradientTape() as tape:
    # Variables are automatically recorded, but manually watch a tensor
    tape.watch(init_x)
    value = fn(init_x)
  grad = tape.gradient(value, init_x)
  grad_norm = tf.reduce_sum(grad * grad)
  init_value = value
  while value > init_value - rate * grad_norm:
    x = init_x - rate * grad
    value = fn(x)
    rate /= 2.0
  return x, value
```

### 计算梯度的附加函数

`tf.GradientTape` 是一个用于计算梯度的功能强大的接口，但其实在自动微分方面，还有另一个 [Autograd](https://github.com/HIPS/autograd)-风格的 API。在只使用张量和梯度函数编写数学代码，而且不使用 `tf.Variables` 的时候，这些函数是有用的：

* `tfe.gradients_function` — 返回一个计算其输入函数参数导数的函数。输入函数参数必须返回标量值。返回函数被调用时，它会返回一个 `tf.Tensor` 对象列表：输入函数的每个参数都有一个元素。因为任何感兴趣的东西都必须作为函数参数传递，如果依赖于许多可训练的参数，这就变得很困难。
* `tfe.value_and_gradients_function` — 类似于 `tfe.gradients_function`，但是当调用返回函数时，除了输入函数的导数列表和参数之外，它还会从输入函数返回值。

在下述示例中，`tfe.gradients_function` 将 `square` 函数作为参数，并返回一个用于计算其输入 `square` 的偏导数的函数。为了计算 `3` 处 `square` 的导数，`grad(3.0)` 返回了 `6`。

```py
def square(x):
  return tf.multiply(x, x)

grad = tfe.gradients_function(square)

square(3.)  # => 9.0
grad(3.)    # => [6.0]

# The second-order derivative of square:
gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
gradgrad(3.)  # => [2.0]

# The third-order derivative is None:
gradgradgrad = tfe.gradients_function(lambda x: gradgrad(x)[0])
gradgradgrad(3.)  # => [None]


# 通过流进行控制：
def abs(x):
  return x if x > 0. else -x

grad = tfe.gradients_function(abs)

grad(3.)   # => [1.0]
grad(-3.)  # => [-1.0]
```

### 自定义梯度

在 eager 和 graph 执行环境中，自定义梯度是重载梯度的一种简单方法。在向前函数中，定义相对于输入、输出或中间结果的梯度。例如，下面是在向后传参时用于裁剪梯度的标准：

```py
@tf.custom_gradient
def clip_gradient_by_norm(x, norm):
  y = tf.identity(x)
  def grad_fn(dresult):
    return [tf.clip_by_norm(dresult, norm), None]
  return y, grad_fn
```

自定义梯度通常用于为一系列操作提供数值稳定的梯度：

```py
def log1pexp(x):
  return tf.log(1 + tf.exp(x))
grad_log1pexp = tfe.gradients_function(log1pexp)

# The gradient computation works fine at x = 0.
grad_log1pexp(0.)  # => [0.5]

# However, x = 100 fails because of numerical instability.
grad_log1pexp(100.)  # => [nan]
```

这里的 `log1pexp` 函数可以用自定义函数解析简化。以下实现重用了 `tf.exp(x)` 在正向传递过程中计算出的值——通过消除冗计算来提高效率：

```py
@tf.custom_gradient
def log1pexp(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * (1 - 1 / (1 + e))
  return tf.log(1 + e), grad

grad_log1pexp = tfe.gradients_function(log1pexp)

# As before, the gradient computation works fine at x = 0.
grad_log1pexp(0.)  # => [0.5]

# And the gradient computation also works at x = 100.
grad_log1pexp(100.)  # => [1.0]
```

## 性能

在 eager execution 期间，计算会自动加载到 GPU。如果你希望控制计算运行的位置，可以将其封装在 `tf.device('/gpu:0')` 块（或与 CPU 等效的块中）：

```py
import time

def measure(x, steps):
  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
  # tf.matmul 可以在完成矩阵乘法运算之前返回（例如，可以在对 CUDA 流进行操作之后返回）。
  # 下面的 x.numpy() 调用将确保所有排队的操作都已完成（并且还将会结果复制到主机内存中，所以我们要包括的不只是 matmul 操作时间）。
  _ = x.numpy()
  end = time.time()
  return end - start

shape = (1000, 1000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: {} secs".format(measure(tf.random_normal(shape), steps)))

# Run on GPU, if available:
if tfe.num_gpus() > 0:
  with tf.device("/gpu:0"):
    print("GPU: {} secs".format(measure(tf.random_normal(shape), steps)))
else:
  print("GPU: not found")
```

Output (exact numbers depend on hardware):

```
Time to multiply a (1000, 1000) matrix by itself 200 times:
CPU: 1.46628093719 secs
GPU: 0.0593810081482 secs
```

A `tf.Tensor` object can be copied to a different device to execute its operations:

```py
x = tf.random_normal([10, 10])

x_gpu0 = x.gpu()
x_cpu = x.cpu()

_ = tf.matmul(x_cpu, x_cpu)    # Runs on CPU
_ = tf.matmul(x_gpu0, x_gpu0)  # Runs on GPU:0

if tfe.num_gpus() > 1:
  x_gpu1 = x.gpu(1)
  _ = tf.matmul(x_gpu1, x_gpu1)  # Runs on GPU:1
```

### 基准

对于计算量很大的模型，例如 [ResNet50](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/resnet50) 在 GPU 上的训练，eager execution 性能可以与 graph execution 像媲美。但是对于计算量较小的模型来说，这种差距会越来越大，而且对于具有大量小计算的模型来说，优化热代码路径仍然有许多工作需要完成。


## 使用图

尽管 eager execution 使开发和调试更具交互性，但 TensorFlow 的 graph execution 在分布式训练、性能优化以及产品部署上仍旧具有优势。然而，编写 graph 代码不同于编写常规 Python 代码，而且更难进行调试。

为了构建和训练图形构造模型,Python 程序首先构建一个用于表示计算的图，然后调用 `Session.run` 来发送此图，以便在基于 C++ 运行时执行。这提供了：

* 使用静态自动微分法来自动微分。
* 简单部署到独立于平台的服务器。
* 基于图形的优化（常见的子表达式消除，常量折叠等）。
* 编译和内核融合。
* 自动分发和复制（在分布式系统上放置节点）。

为 eager execution 编写部署代码更加困难：要么从模型生成图形，要么在服务器上直接运行 Python 运行时代码。

### 编写兼容性代码

为 eager execution 编写的相同代码也会在 graph execution 期间构建图形。只需在未启用 eager execution 的新 Python session 中运行相同的代码即可。

大多数 TensorFlow 操作都在 eager execution 期间都是可以运行的，但有些事需要记住：

* 使用 `tf.data` 而不是队列，来进行输入处理。这会更快，更简单。
* 使用面向对象的层 API——例如 `tf.keras.layers` 和 `tf.keras.Model`——因为它们对变量进行显示存储。
* 大多数模型在 eager execution 和 graph execution 中的表现是一样的，但也有特列。（例如，动态模型使用 Python 控制流来改变基于输入的计算。）
* 一旦通过 `tf.enable_eager_execution` 启用 eager execution，它就不会被关闭。启动一个新的 Python session 来返回到 graph execution。

最好是同时为 eager execution **和** graph execution 编写代码。这将为你提供 eager 的交互式体验和可调式性，以及 graph execution 的分布式性能优势。

在 eager execution 中编写，调试和迭代，然后为生产部署导入模型图。使用 `tf.train.Checkpoint` 来保存和存储模型变量，这允许在 eager 和 graph execution 环境之间移动。请参阅以下示例：[tensorflow/contrib/eager/python/examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples)。

### 在图环境中使用 eager execution 执行

使用 `tfe.py_func` 选择性地启用 TensorFlow 图形化环境中的 eager execution 执行。当 `tf.enable_eager_execution()` **尚未**被调用时使用。

```py
def my_py_func(x):
  x = tf.matmul(x, x)  # 你可以使用 tf ops
  print(x)  # 但这是 eager！
  return x

with tf.Session() as sess:
  x = tf.placeholder(dtype=tf.float32)
  # 在图形中调用 eager 函数
  pf = tfe.py_func(my_py_func, [x], tf.float32)
  sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]
```
