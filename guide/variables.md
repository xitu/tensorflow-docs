# 变量

Tensorflow **变量**是用来表述程序控制的共享、持续状态的最好的方式。

变量是通过 `tf.Variable` 类来控制的。`tf.Variable` 描述了一个张量，其值可以通过运行操作来改变。不像 `tf.Tensor` 对象，`tf.Variable` 存在于 `session.run` 调用的上下文之外。

从内部来说，`tf.Varable` 保存了一个常态的张量。你可以通过特定的操作读取和修改该张量的值。这种修改对于多个 `tf.Session` 都是可见的，因此多个工作线程都能获取 `tf.Variable` 相同的值。

## 创建变量

最好的创建变量的方式就是调用 `tf.get_variable` 方法。这个方法要求你指定变量名，变量名将被用于其他副本获取相同变量，同时也用于备份和导出模型时给变量值命名。`tf.get_variable` 也允许你重用一个以前创建的同名的变量，重用层的方式使得定义模型变得很简单。

用 `tf.get_variable` 创建一个变量，你只需提供变量名和形状

``` python
my_variable = tf.get_variable("my_variable", [1, 2, 3])
```

这将会创建一个名为 "my_variable" 的三维变量，形状是 `[1, 2, 3]`。默认情况下，这个变量的 `dtype` 为 `tf.float32` 并且会被 `tf.glorot_uniform_initializer` 赋上随机的初始值。

你也能可选地给 `tf.get_variable` 指定 `dtype` 和初始值。比如：

``` python
my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.int32,
  initializer=tf.zeros_initializer)
```

TensorFlow 提供了很多便捷的初始化方式。方案之一，你可以指定 `tf.Variable` 的初始值为 `tf.Tensor`。比如：

``` python
other_variable = tf.get_variable("other_variable", dtype=tf.int32,
  initializer=tf.constant([23, 42]))
```

要注意的是当初始值是 `tf.Tensor` 你就不应当指定变量的形状了，因为变量会使用初始化张量的形状。

<a name="collections"></a>
### 变量集合

因为 TensorFlow 程序中不相连的部分可能会需要创建变量，所以有时候很有必要通过单一的方式来获取所有的变量。因此，TensorFlow 提供了**集合**，被称为张量或其他对象的列表，比如 `tf.Variable` 实例的列表。

默认情况下，每个 `tf.Variable` 都会出现在以下两个集合中：

 * `tf.GraphKeys.GLOBAL_VARIABLES` --- 可以在多个设备间共享的变量

 * `tf.GraphKeys.TRAINABLE_VARIABLES` --- TensorFlow 梯度计算的变量

如果你不想让变量在训练中出现，你可以将它加入到 `tf.GraphKeys.LOCAL_VARIABLES` 集合中。比如，下面这个例子示范了如何将一个名为 `my_local` 的变量加入到这个集合中：

``` python
my_local = tf.get_variable("my_local", shape=(),
collections=[tf.GraphKeys.LOCAL_VARIABLES])
```

或者，你可以指定 `trainable=False` 作为 `tf.get_variable` 的一个参数：

``` python
my_non_trainable = tf.get_variable("my_non_trainable",
                                   shape=(),
                                   trainable=False)
```

你也可以自定义集合。任何的字符串都是有效的集合名称，并且无需显式地创建一个集合。如果要将创建的变量（或者其他任何对象）添加到集合，调用 `tf.add_to_collection` 方法即可。比如，以下的代码就酱一个已存在的名为 `my_local` 的变量添加到一个名为 `my_collection_name` 的集合：

``` python
tf.add_to_collection("my_collection_name", my_local)
```

如果要获取你存放在集合中的所有变量（或者其他对象）的列表，你可以这样：

``` python
tf.get_collection("my_collection_name")
```

### 设备部署

正如其他的 TensorFlow 操作一样，你可以将变量部署在特定的设备上。比如，以下的代码段就创建了一个名为 `v` 的变量并将其部署在了第二个 GPU 设备上：

``` python
with tf.device("/device:GPU:1"):
  v = tf.get_variable("v", [1])
```

在分布式设置中，将变量部署在正确的设备上尤为重要。比如，如果不小心将变量部署在了工作线程而不是参数服务器上会严重减慢训练效率，最糟糕的情况下，会使得每个工作线程都在不经意间伪造了各自独立的变量的副本。因此，我们提供了 `tf.train.replica_device_setter` 来自动将变量部署在参数服务器上。比如：

``` python
cluster_spec = {
    "ps": ["ps0:2222", "ps1:2222"],
    "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
  v = tf.get_variable("v", shape=[20, 20])  # 这个变量被 replica_device_setter 部署在参数服务器上
```

## 初始化变量

变量在使用之前必须初始化。如果你在用低级的 TensorFlow API 编程（也就是说，你在显式地创建你自己的计算图和会话），你必须显式地初始化这些变量。大部分高级框架，比如 `tf.contrib.slim`, `tf.estimator.Estimator` 和 `Keras` 都会在训练模型之前自动帮你初始化变量。

另一方面，显式的初始化方式也很实用，因为这能让你从备份中重载模型时，不必重新运行可能会很耗性能的初始化函数，同时也在随机初始化分布式设置中的共享变量时允许因果关联（原文为 determinism，译者注）。

在训练开始之前，可以调用 `tf.global_variables_initializer()` 来一步到位初始化所有可训练的变量。这个函数会返回一个操作，该操作可以初始化所有 `tf.GraphKeys.GLOBAL_VARIABLES` 集合中的变量。运行这个操作可以初始化所有的变量。比如：

``` python
session.run(tf.global_variables_initializer())
# 所有的变量都会被初始化
```

如果你需要自行初始化变量，你可以手动运行变量初始化的操作。比如：

``` python
session.run(my_variable.initializer)
```

你也可以查看哪些变量还没有被初始化。比如，下面这段代码打印出了所有还没被初始化的变量名称：

``` python
print(session.run(tf.report_uninitialized_variables()))
```

需要注意的是，默认情况下 `tf.global_variables_initializer` 并不会指定变量被初始化的顺序。因此，如果一个变量的初始化的值依赖于另一个变量的值，很可能你就会得到一个错误。任何在变量没有完全被初始化的上下文中使用变量值的时候（也就是说，你在初始化一个变量时使用了另一个变量的值），你最好使用 `variable.initialized_value()` 而不要用 `variable`。

``` python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = tf.get_variable("w", initializer=v.initialized_value() + 1)
```

## 使用变量

如果要在 TensorFlow 计算图中使用 `tf.Variable` 的值，你只需把它当作一个普通的 `tf.Tensor` 来使用：

``` python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
w = v + 1  # w 是基于 v 的值来计算的 tf.Tensor
           # 用于表达式中的变量，在任何时候都会自动地转换成 tf.Tensor。
```

如果要给变量指派值，使用 `tf.Variable` 类中的 `assign`, `assign_add` 或者其他相近的方法。比如，你可以这样调用这些方法：

``` python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
tf.global_variables_initializer().run()
sess.run(assignment)  # or assignment.op.run(), or assignment.eval()
```

绝大部分的 TensorFlow 优化器都有特定的操作来依据某些梯度下降算法高效地更新变量的值。我们用 `tf.train.Optimizer` 来解释如何使用优化器。

因为变量时可变的，因此有些情况下及时知道正在使用的变量的值是哪个版本很有必要。在某些操作后强制重新读取变量值可以通过 `tf.Variable.read_value` 实现。比如：

``` python
v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(1)
with tf.control_dependencies([assignment]):
  w = v.read_value()  # w 保证了在 assign_add 操作之后还能反应 v 的值
```

## 共享变量

TensorFlow 支持两种共享变量的方式：

 * 显式地传递 `tf.Variable` 对象。
 * 隐式地将 `tf.Variable` 对象包含在 `tf.variable_scope` 对象中。

尽管显式地传递变量的代码已经很清晰了，有时在 TensorFlow 函数的实现中隐式地使用变量也很方便。`tf.layers` 中的大多数函数层都是用了这种方式，`tf.metrics` 也是，还有一些其他的类库工具也是如此。

变量作用域允许你控制在调用隐式地创建和使用变量的函数时控制变量的重用规则，也允许你用一种有层次的、易懂的方式来命名变量。

比如，我们写了一个函数来创建一个卷积／relu 层：

```python
def conv_relu(input, kernel_shape, bias_shape):
    # 创建一个名为 "weights" 的变量
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # 创建一个名为 "biases" 的变量
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)
```

这个函数使用了 `weights` 和 `biases` 这样的短名字，这样很容易懂。然而在真实的模型中，我们会需要许多这样错综复杂的层，并且在重复调用这些函数的时候将会出乱子：

``` python
input1 = tf.random_normal([1,10,10,32])
input2 = tf.random_normal([1,20,20,32])
x = conv_relu(input1, kernel_shape=[5, 5, 32, 32], bias_shape=[32])
x = conv_relu(x, kernel_shape=[5, 5, 32, 32], bias_shape = [32])  # This fails.
```

因为目标结果并不明确（创建新的变量还是重用已存在的变量？）TensorFlow 将会无法执行。然而在不同作用域中调用 `conv_relu` 会表明我们想要创建新的变量：

```python
def my_image_filter(input_images):
    with tf.variable_scope("conv1"):
        # 此处创建的变量会被命名为 "conv1/weights", "conv1/biases"
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        # 此处创建的变量会被命名为 "conv2/weights", "conv2/biases"
        return conv_relu(relu1, [5, 5, 32, 32], [32])
```

如果你想变量被共享，你有两个选择。第一，你可以用 `reuse=True` 创建一个相同名字的作用域：

``` python
with tf.variable_scope("model"):
  output1 = my_image_filter(input1)
with tf.variable_scope("model", reuse=True):
  output2 = my_image_filter(input2)
```

你也可以调用 `scope.reuse_variables()` 来触发重用：

``` python
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
  scope.reuse_variables()
  output2 = my_image_filter(input2)
```

因此仅仅依赖于不同名称的作用域是很危险的。有时候，我们还会基于其他作用域来初始化变量作用域：

``` python
with tf.variable_scope("model") as scope:
  output1 = my_image_filter(input1)
with tf.variable_scope(scope, reuse=True):
  output2 = my_image_filter(input2)
```
