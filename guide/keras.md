# Keras

Keras 是一个用于构建和训练深度学习模型的高级程序接口。它被应用于快速原型设计，高级研究和生产环境，有着三大核心优势：

- <b>用户友好</b><br>
  Keras 为常见用例优化了一套简单一致的接口。它给用户错误提供清晰且可操作的反馈。
- <b>模块化和组件化</b><br>
  Keras 模型将可配置的模块连接在一起，几乎没有任何限制。
- <b>易扩展</b><br> 编写自定义模块用于研究新想法。创建新的网络层，代价函数和研发先进的模型。

## 导入 tf.keras

`tf.keras` 是 [Keras 接口规范](https://keras.io){:.external} 的 TensorFlow 实现。这是一个用于构建和训练模型的高级接口，其中包括对 TensorFlow 特定功能的一流支持，比如 [Eager execution](#eager_execution)，
`tf.data` 管道和 [估计器](./estimators.md)。
`tf.keras` 使得 TensorFlow 在不牺牲灵活性和性能的基础上更加易于使用。

首先，在你的 TensorFlow 程序开始导入 `tf.keras`：

```python
import tensorflow as tf
from tensorflow import keras
```

`tf.keras` 能够运行任何 Keras 兼容代码，但是要记住：
* 在 TensorFlow 最新发布的版本中，`tf.keras` 的版本可能和 PyPI 中 `keras` 最新的版本不一样。请检查 `tf.keras.__version__`。
* 当[保存模型权重](#weights_only)，`tf.keras` 默认[检查点格式](./checkpoints.md)。将 `save_format='h5'` 作为参数传入，以使用 HDF5 文件格式。

## 构建简单模型

### 序列模型

在 Keras 中, 你可以拼接<b>网络层</b>来构建<b>模型</b>。模型（通常）是包含多个网络层的图。最常见的模型就是由多个网络层堆叠而成的：`tf.keras.Sequential` 模型。

构建简单的全连接网络（比如多层感知器）：

```python
model = keras.Sequential()
# 模型中添加包含 64 个节点的全连接层：
model.add(keras.layers.Dense(64, activation='relu'))
# 添加另外一个：
model.add(keras.layers.Dense(64, activation='relu'))
# 添加包含 10 个输出单元的 softmax 层：
model.add(keras.layers.Dense(10, activation='softmax'))
```

### 配置网络层

许多 `tf.keras.layers` 具有相同的构造参数：

* `activation`：设置网络层的激活函数。此参数由内置函数或可调用对象指定。默认情况下，不应用任何激活函数。
* `kernel_initializer` 与 `bias_initializer`：初始化网络层的权重（核和偏差）。该参数是一个名字或者可调用对象。默认是 `"Glorot uniform"` 初始值。
* `kernel_regularizer` 与 `bias_regularizer`：将正则化方案应用于网络层的权重（核和偏差），比如 L1 和 L2 正则化。默认不使用任何正则化。
下面使用构造函数参数实例化 `tf.keras.layers.Dense`：

```python
# 创建 sigmoid 网络层：
layers.Dense(64, activation='sigmoid')
# 或者：
layers.Dense(64, activation=tf.sigmoid)

# 将 L1 正则化因子为 0.01 的线性层应用于核矩阵：
layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
# 将 L1 正则化因子为 0.01 的线性层应用于偏差向量：
layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))

# 线性层核矩阵初始化为随机正交矩阵：
layers.Dense(64, kernel_initializer='orthogonal')
# 线性层偏置向量初始化为常数值 2.0：
layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))
```

## 训练和评估

### 开始训练

构建模型后，通过调用 `compile` 方法配置其学习过程：

```python
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

`tf.keras.Model.compile` 包含三个重要参数：

* `optimizer`：这个参数指定训练过程。从 `tf.train` 模块传递优化器实例，比如 [`Adam 优化器`](/api_docs/python/tf/train/AdamOptimizer)，[`RMSProp 优化器`](/api_docs/python/tf/train/RMSPropOptimizer) 或者 [`梯度下降优化器`](/api_docs/python/tf/train/GradientDescentOptimizer)。
* `loss`：优化期间的目标最小化的函数。常见的有均方误差（`mse`），`categorical_crossentropy` 和 `binary_crossentropy`。损失函数由名称或通过从 `tf.keras.losses` 模块传递可调用对象来指定。
* `metrics`：用于监督训练。可由名称或通过从 `tf.keras.metrics` 模块传递可调用对象。

以下提供了配置训练模型的几个示例：

```python
# 配置以均方误差作为损失函数的回归模型。
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# 配置类别分类模型。
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])
```

### 输入 NumPy 数据

对于小型数据集，使用内存型数组 [NumPy](https://www.numpy.org/){:.external} 训练和评估模型。模型使用 `fit` 方法 “拟合” 训练数据：

```python
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)
```

`tf.keras.Model.fit` 包含三个重要参数：

* `epochs`：训练过程被划分到 *epochs*。一个 epoch 是对整个输入数据的一次迭代（这是以较小的批次完成的）。
* `batch_size`：当传递 NumPy 数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。此整数指定每个批次的大小。请注意，如果样本总数不能被批次大小整除，则最后一批可能会更小。
* `validation_data`：在对模型进行原型设计时，您希望轻松监控其在某些验证数据上的性能。传递包含输入数据和标签的元组，让模型在每个 epoch 的结束后计算并打印损失和度量值。

这里有个例子使用了 `validation_data`：

```python
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))
```

### 输入 tf.data datasets

使用[数据集接口](./datasets.md)扩展到大型数据集或者多设备训练。将 `tf.data.Dataset` 实例传递给 `fit` 方法：

```python
# 实例化玩具数据集实例。
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# 在数据集上调用 `fit` 方法不要忘记指定 `steps_per_epoch` 参数值。
model.fit(dataset, epochs=10, steps_per_epoch=30)
```

这里，`fit` 方法使用 `steps_per_epoch` 参数，该参数是模型在移动到下一个 epoch 之前训练步数。由 `Dataset` 产生批量数据，因此该代码片段不需要 `batch_size`。

数据集同样可以被用来验证：

```python
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)
```

### 评价和预测

`tf.keras.Model.evaluate` 和 `tf.keras.Model.predict` 方法可以传递 NumPy 数据和 `tf.data.Dataset` 对象。

<b>评估</b>在提供的数据集上的代价损失和指标：

```python
model.evaluate(x, y, batch_size=32)

model.evaluate(dataset, steps=30)
```

并且<b>预测</b>提供的 NumPy 数据在最后一层输出结果：

```
model.predict(x, batch_size=32)

model.predict(dataset, steps=30)
```


## 构建高级模型

### 函数式接口

`tf.keras.Sequential` 模型是一个简单的网络层拼接，不能代表任意模型。使用[Keras 函数式接口](https://keras.io/getting-started/functional-api-guide/){:.external}构建复杂的模型拓扑，比如：

* 多输入模型，
* 多输出模型，
* 包含共享层模型（相同层被调用多次），
* 不包含序列化数据流的模型 (比如残余神经网络模型）。

使用函数式接口构建模型的工作方式如下：

1. 网络层实例可调用并返回张量。
2. 输入张量和输出张量用于定义 `tf.keras.Model` 实例。
3. 模型像 `Sequential` 序列模型一样训练。

以下示例使用函数式接口构建一个简单，完全连接的网络：

```python
inputs = keras.Input(shape=(32,))  # 返回占位张量

# 网络层可调用张量并且返回张量。
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

# 给定输入和输出实例化模型。
model = keras.Model(inputs=inputs, outputs=predictions)

# 编译步骤指定训练的配置。
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练 5 个 epoch
model.fit(data, labels, batch_size=32, epochs=5)
```

### 模型子类化

通过继承 `tf.keras.Model` 并定义自己的前向传播来构建一个完全自定义的模型。在 `__init__` 方法中创建图层并将它们设置为类实例的属性。在 `call` 方法中定义前向传播。

当 [Eager execution](./eager.md) 被启用时，对模型子类化特别有用，因为可以强制写入前向传播。

关键点：工作中使用正确的接口。虽然模型子类化提供了灵活性，但其代价是更高的复杂性和更多的用户错误机会。如果可能，请选择函数式接口。

下面的示例使用自定义前向传播展示了一个子类化 `tf.keras.Model`：


```python
class MyModel(keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # Define your layers here.
    self.dense_1 = keras.layers.Dense(32, activation='relu')
    self.dense_2 = keras.layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.dense_1(inputs)
    return self.dense_2(x)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)


# 实例化子类化模型。
model = MyModel(num_classes=10)

# 编译步骤指定训练配置。
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练 5 epochs。
model.fit(data, labels, batch_size=32, epochs=5)
```


### 自定义网络层

通过继承 `tf.keras.layers.Layer` 创建自定义网络层并且实现以下方法：

* `build`：创建网络层的权重。使用 `add_weight` 方法添加权重。
* `call`：定义前向传播。
* `compute_output_shape`：指定在给定输入大小的情况下如何计算网络层的输出大小。
* 另外，可以通过实现 `get_config` 和 `from_config` 类方法实现序列化。

这是一个自定义网络层的示例，它实现了带有内核矩阵输入的 `matmul`：

```python
class MyLayer(keras.layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    shape = tf.TensorShape((input_shape[1], self.output_dim))
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
                                  shape=shape,
                                  initializer='uniform',
                                  trainable=True)
    # Be sure to call this at the end
    super(MyLayer, self).build(input_shape)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.output_dim
    return tf.TensorShape(shape)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


# 使用自定义网络层构建模型
model = keras.Sequential([MyLayer(10),
                          keras.layers.Activation('softmax')])

# 编译步骤指定训练模型配置
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练 5 个 epoch。
model.fit(data, targets, batch_size=32, epochs=5)
```


## 回调

回调是传递给模型的对象，用于在训练期间自定义和扩展其行为。您可以编写自己的自定义回调，或使用内置的 `tf.keras.callbacks`，其中包括：

* `tf.keras.callbacks.ModelCheckpoint`：定期保存模型的检查点。
* `tf.keras.callbacks.LearningRateScheduler`：动态调整学习率。
* `tf.keras.callbacks.EarlyStopping`：当验证性能停止增长时，中断训练。
* `tf.keras.callbacks.TensorBoard`：使用 [TensorBoard](./summaries_and_tensorboard.md) 监控模型行为。

要使用 `tf.keras.callbacks.Callback`，将其传递给模型的 `fit` 方法：

```python
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_targets))
```


## 保存与恢复

### 仅权重

使用 `tf.keras.Model.save_weights` 保存和加载模型的权重：

```python
# 保存权重到 TensorFlow 检查点文件
model.save_weights('./my_model')

# 恢复模型的状态，
# 这需要具有相同架构的模型。
model.load_weights('my_model')
```

默认情况下，这会将模型的权重保存在 [TensorFlow 检查点](./checkpoints.md) 文件格式中。权重也可以保存为 Keras HDF5 格式（Keras 的多后端实现的默认值）：

```python
# 保存权重到 HDF5 文件
model.save_weights('my_model.h5', save_format='h5')

# 恢复模型状态
model.load_weights('my_model.h5')
```


### 仅配置

可以保存模型的配置 — 这可以在没有任何权重的情况下序列化模型体系结构。即使没有定义原始模型的代码，保存的配置也可以重新创建和初始化相同的模型。Keras 支持 JSON 和 YAML 序列化格式。

```python
# 使用 JSON 格式序列化模型
json_string = model.to_json()

# 重新创建模型（首次初始化）
fresh_model = keras.models.model_from_json(json_string)

# 使用 YAML 格式序列化模型
yaml_string = model.to_yaml()

# 重建模型
fresh_model = keras.models.model_from_yaml(yaml_string)
```

注意：子类化模型不可序列化，因为它们的体系结构是由 `call` 方法中的 Python 代码定义的。


### 全模型

整个模型都可以保存到文件，包含权重值，模型配置甚至优化器配置。这允许您检查模型并稍后从完全相同的状态恢复训练 — 无需访问原始代码。

```python
# 创建一个简单的模型
model = keras.Sequential([
  keras.layers.Dense(10, activation='softmax', input_shape=(32,)),
  keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, targets, batch_size=32, epochs=5)


# 保存整个模型到 HDF5 文件中
model.save('my_model.h5')

# 重建完全一样的模型，包含权重和优化器。
model = keras.models.load_model('my_model.h5')
```


## Eager execution

[Eager execution](./eager.md) 是一个命令式编程环境，可立即评估操作。这不是 Keras 所必需的，但是 `tf.keras` 支持，对于检查程序和调试很有用。

所有 `tf.keras` 模型构建接口都与 eager execution 兼容。虽然可以使用 `Sequential` 和函数式接口，但是 eager execution 对<b>子类化模型</b>和构建<b>自定义层</b>有特别友好 — 仅需要您编写前向传播的接口代码（而不是使用现有的创建模型的接口）。

有关使用自定义训练和 `tf.GradientTape` 的 Keras 模型示例，请参阅 [eager execution 指引](./eager.md#build_a_model)。


## 分布式

### 估计器

[估计器](./estimators.md)接口可以用于分布式环境的训练模型。应用对象主要是工业界，例如可以导出模型进行生产的大型数据集的分布式训练。

通过 `tf.keras.estimator.model_to_estimator` 将 `tf.keras.Model` 转化为 `tf.estimator.Estimator` 对象，使用 `tf.estimator` 接口训练。详见[从 Keras 模型中创建估计器](./estimators.md#creating_estimators_from_keras_models)。

```python
model = keras.Sequential([layers.Dense(10,activation='softmax'),
                          layers.Dense(10,activation='softmax')])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

estimator = keras.estimator.model_to_estimator(model)
```

注意：开启 [eager execution](./eager.md) 可以调试[估计器输入函数](./premade_estimators.md#create_input_functions)和观察数据。

### GPU 集群

可以使用 `tf.contrib.distribute.DistributionStrategy` 在多个 GPU 上运行。 此接口在多个 GPU 上提供分布式训练，几乎不需要对现有代码进行任何更改。

目前，`tf.contrib.distribute.MirroredStrategy` 是唯一受支持的分发策略。`MirroredStrategy` 使用 all-reduce 在一台机器上进行图模型内部的复制与同步。要使用 Keras 的 `DistributionStrategy`，将 `tf.keras.Model` 转换为`tf.estimator.Estimator` 与 `tf.keras.estimator.model_to_estimator`，然后训练估算器。

以下示例在单个计算机上的多个 GPU 之间分发 `tf.keras.Model`。

首先，定义一个简单的模型：

```python
model = keras.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

optimizer = tf.train.GradientDescentOptimizer(0.2)

model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()
```

定义<b>输入管道</b>。`input_fn` 返回一个 `tf.data.Dataset` 对象，用于在多个设备之间分配数据 — 每个设备处理一个批处理输入分片。

```python
def input_fn():
  x = np.random.random((1024, 10))
  y = np.random.randint(2, size=(1024, 1))
  x = tf.cast(x, tf.float32)
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.repeat(10)
  dataset = dataset.batch(32)
  return dataset
```

接下来，创建一个 `tf.estimator.RunConfig` 并将 `train_distribute` 参数设置为 `tf.contrib.distribute.MirroredStrategy` 实例。创建 `MirroredStrategy` 时，可以指定设备列表或设置 `num_gpus` 参数。默认使用所有可用的 GPU，如下所示：

```python
strategy = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)
```

Keras 模型转化为 `tf.estimator.Estimator` 实例：

```python
keras_estimator = keras.estimator.model_to_estimator(
  keras_model=model,
  config=config,
  model_dir='/tmp/model_dir')
```

最后，提供 `input_fn` 和 `steps` 参数训练 `估计器` 实例：

```python
keras_estimator.train(input_fn=input_fn, steps=10)
```
