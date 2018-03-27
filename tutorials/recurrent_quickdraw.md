# 对涂鸦进行分类的循环神经网络

[Quick, Draw!]: http://quickdraw.withgoogle.com

[Quick, Draw!] 是一个让玩家画出物体并让计算机识别所画物体是什么的游戏。

[Quick, Draw!] 的识别工作由一个分类器执行，并将用户的输入试做一个由绘制点组成的序列，然后用于对象识别。

在本教程中，我们将展示如何基于 RNN 为此问题构建识别器。该模型使用卷积层、LSTM 层以及一个 softmax 输出层来对涂鸦进行分类：

<center> ![RNN 模型架构](../images/quickdraw_model.png) </center>

上图展示了此教程中构建的模型架构。输入为一个图形，编码为由 x, y 和 n 构成的坐标点序列，其中 n 表示该点是否是新笔划中的第一个点。

然后，应用一系列一维卷积，然后使用 LSTM 层将所有的输出传递给 softmax 层，进而进行涂鸦分类。

This tutorial uses the data from actual [Quick, Draw!] games [that is publicly available](https://quickdraw.withgoogle.com/data). This dataset contains of 50M drawings in 345 categories.
此教程使用的数据来自实际的 [Quick, Draw!] 游戏[公开数据集](https://quickdraw.withgoogle.com/data)。数据包含了 50M 个涂鸦以及 345 个分类。

## 运行教程代码

运行这个教程的代码：

1.  如果你还没有安装 TensorFlow，那么请@{$install$Install TensorFlow}。
2.  下载[教程代码](https://github.com/tensorflow/models/tree/master/tutorials/rnn/quickdraw/train_model.py).
3.  [下载](http://download.tensorflow.org/data/quickdraw_tutorial_dataset_v1.tar.gz) `TFRecord` 格式的[数据](#下载数据) 并解压。更多细节请参考[可选：下载整个 Quick Draw 数据](#可选：下载整个 Quick Draw 数据) 以及[如何转换原始 Quick, Draw! 数据](可选：数据转换)。

4.  使用一下命令可以执行教程代码，并训练教程中描述的 RNN 模型。请自行填写在第三步中放置的解压好的数据路径。

```shell
  python train_model.py \
    --training_data=rnn_tutorial_data/training.tfrecord-?????-of-????? \
    --eval_data=rnn_tutorial_data/eval.tfrecord-?????-of-????? \
    --classes_file=rnn_tutorial_data/training.tfrecord.classes
```

## 教程细节

### 下载数据

我们将本教程使用的数据作为包含 `TFExamples` 的 `TFRecord` 文件提供下载：

http://download.tensorflow.org/data/quickdraw_tutorial_dataset_v1.tar.gz

或者，你也可以从 Google 云上下载 `ndjson` 格式的原始数据，并将其转换为包含 `TFExamples` 的 `TFRecord` 文件，具体方法参考下一节。

### 可选：下载整个 Quick Draw 数据

完整的 [Quick, Draw!](https://quickdraw.withgoogle.com) [数据集](https://quickdraw.withgoogle.com/data) 可以在 Google Cloud Storage 上下载按类别分类的  [ndjson](http://ndjson.org/) 格式原始文件。你可以在[Cloud Console](https://console.cloud.google.com/storage/quickdraw_dataset) 上直接浏览文件列表。

要下载我们推荐的数据，请使用 [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install) 下载整个数据集。注意，原始 `.ndjson` 格式的文件大约有 22GB。

然后使用下面的命令检查你的 gsutil 安装是否正常，并且可以访问到数据存储区：

```shell
gsutil ls -r "gs://quickdraw_dataset/full/simplified/*"
```

这个命令会输出一长串文件列表：

```shell
gs://quickdraw_dataset/full/simplified/The Eiffel Tower.ndjson
gs://quickdraw_dataset/full/simplified/The Great Wall of China.ndjson
gs://quickdraw_dataset/full/simplified/The Mona Lisa.ndjson
gs://quickdraw_dataset/full/simplified/aircraft carrier.ndjson
...
```

然后，创建一个文件夹并从云端下载数据集：

```shell
mkdir rnn_tutorial_data
cd rnn_tutorial_data
gsutil -m cp "gs://quickdraw_dataset/full/simplified/*" .
```

这个将下载大约 23GB 数据并持续很长一段时间。

### 可选：数据转换

将 `ndjson` 文件转换到 @{$python/python_io#tfrecords_format_details$TFRecord} 格式，可以运行下面的命令：

```shell
   python create_dataset.py --ndjson_path rnn_tutorial_data \
      --output_path rnn_tutorial_data
```

这个命令会将数据作为十个分片存储在 @{$python/python_io#tfrecords_format_details$TFRecord} 文件，数据的每个类别包含 10000 个训练样本，以及 1000 个测试样本

下面描述更为详细的转换过程。

原始的 QuickDraw 数据格式为 `.ndjson`，每行包含一个 JSON 对象，例如：

```json
{"word":"cat",
 "countrycode":"VE",
 "timestamp":"2017-03-02 23:25:10.07453 UTC",
 "recognized":true,
 "key_id":"5201136883597312",
 "drawing":[
   [
     [130,113,99,109,76,64,55,48,48,51,59,86,133,154,170,203,214,217,215,208,186,176,162,157,132],
     [72,40,27,79,82,88,100,120,134,152,165,184,189,186,179,152,131,114,100,89,76,0,31,65,70]
   ],[
     [76,28,7],
     [136,128,128]
   ],[
     [76,23,0],
     [160,164,175]
   ],[
     [87,52,37],
     [175,191,204]
   ],[
     [174,220,246,251],
     [134,132,136,139]
   ],[
     [175,255],
     [147,168]
   ],[
     [171,208,215],
     [164,198,210]
   ],[
     [130,110,108,111,130,139,139,119],
     [129,134,137,144,148,144,136,130]
   ],[
     [107,106],
     [96,113]
   ]
 ]
}
```

根据我们需要创建分类器的目的， 我们只关心字段 `word` 和 `drawing`。因此，在解析 `.ndjson` 文件时，我们逐行将 JSON 通过一个函数将 `drawing` 字段处理成 `[number_of_points, 3]` 大小的张量，并同时返回类别的名称。

```python
def parse_line(ndjson_line):
  """解析一个 ndjson 行，并返回涂鸦数据和类别名称."""
  sample = json.loads(ndjson_line)
  class_name = sample["word"]
  inkarray = sample["drawing"]
  stroke_lengths = [len(stroke[0]) for stroke in inkarray]
  total_points = sum(stroke_lengths)
  np_ink = np.zeros((total_points, 3), dtype=np.float32)
  current_t = 0
  for stroke in inkarray:
    for i in [0, 1]:
      np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
    current_t += len(stroke[0])
    np_ink[current_t - 1, 2] = 1  # stroke_end
  # 预处理.
  # 1. 大小归一
  lower = np.min(np_ink[:, 0:2], axis=0)
  upper = np.max(np_ink[:, 0:2], axis=0)
  scale = upper - lower
  scale[scale == 0] = 1
  np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
  # 2. 计算变化
  np_ink = np_ink[1:, 0:2] - np_ink[0:-1, 0:2]
  return np_ink, class_name
```

由于我们希望将读取的数据随机打乱，因此我们按随机顺序将每个类别文件读取并写入到随机分片中。

对于训练数据而言，我们每个类别读取 10000 个样本，而对于测试样本而言，我们对每个类别读取 1000 个样本。

然后将这些数据重新格式化为形状为 `[num_training_samples, max_length, 3]` 的张量。然后再根据坐标值确定原始图形的边框，并将尺寸标准化，使图形具有单位长度的大小。

<center> ![大小归一](../images/quickdraw_sizenormalization.png) </center>

最后，我们计算连续点之间的差值并将其作为 `VarLenFeature` 存储在 [tensorflow.Example](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)  下，命名为 `ink` 字段。此外，我们还将 `class_index` 作为单个 `FixedLengthFeature` 条目进行存储，`ink` 的 `shape` 作为长度为 2 的 `FixedLengthFeature` 进行存储。

### 定义模型

为定义模型，我们首先需要创建一个新的 `Estimator`。如果你想了解更多关于估计器的内容，推荐阅读 @{$get_started/custom_estimators$this tutorial}

构建模型分为以下几步：

1.  将输入数据的形状进行变换，即把数据的小批量填充为其内容的最大程度。除了涂鸦轨迹数据外，我们还需要每个样本的类别以及长度。这需要函数 [`_get_input_tensors`](#-get-input-tensors) 进行处理。

2.  将输入传递给 [`_add_conv_layers`](#-add-conv-layers) 中一系列卷积层。

3.  将卷积层输出的结果传递给 [`_add_rnn_layers`](#-add-rnn-layers) 中一系列的双向 LSTM 层。最后将每个时间点得到的输出求和得到一个固定长度的输入数据的向量。

4.  使用 [`_add_fc_layers`](#-add-fc-layers) 中的 softmax 图层对此向量进行分类。

代码如下：

```python
inks, lengths, targets = _get_input_tensors(features, targets)
convolved = _add_conv_layers(inks)
final_state = _add_rnn_layers(convolved, lengths)
logits =_add_fc_layers(final_state)
```

### _get_input_tensors

为了获得输入特征，我们首先需要从特征字典中获取并创建一个包含输入序列长度的一维张量，大小为 `[batch_size]`。用户所绘内容作为稀疏张量存储在字典中，我们再将其转换为一个稠密张量，大小为 `[batch_size, ?, 3]`。最后，如果目标通过了，我们就确保其存储在一个尺寸为 `[batch_size]` 的一维张量中。

代码如下：

```python
shapes = features["shape"]
lengths = tf.squeeze(
    tf.slice(shapes, begin=[0, 0], size=[params["batch_size"], 1]))
inks = tf.reshape(
    tf.sparse_tensor_to_dense(features["ink"]),
    [params["batch_size"], -1, 3])
if targets is not None:
  targets = tf.squeeze(targets)
```

### _add_conv_layers

通过参数 `num_conv` 和 `params` 中的 `conv_len` 来配置卷积层的数量以滤波器的长度。

输入为一个序列，其中每个点具有三个维度。我们使用一维卷积来将三个输入作为不同通道。换句话说，输入是一个 `[batch_size, length, 3]` 大小的张量，输出则是 `[batch_size, length, number_of_filters]` 大小的张量。

```python
convolved = inks
for i in range(len(params.num_conv)):
  convolved_input = convolved
  if params.batch_norm:
    convolved_input = tf.layers.batch_normalization(
        convolved_input,
        training=(mode == tf.estimator.ModeKeys.TRAIN))
  # 如果启用且不是第一个卷积层，则加入 dropout 层
  if i > 0 and params.dropout:
    convolved_input = tf.layers.dropout(
        convolved_input,
        rate=params.dropout,
        training=(mode == tf.estimator.ModeKeys.TRAIN))
  convolved = tf.layers.conv1d(
      convolved_input,
      filters=params.num_conv[i],
      kernel_size=params.conv_len[i],
      activation=None,
      strides=1,
      padding="same",
      name="conv1d_%d" % i)
return convolved, lengths
```

### _add_rnn_layers

我们将卷积输出传递给双向 LSTM 层，为此我们使用了 `contrib` 的辅助函数。

```python
outputs, _, _ = contrib_rnn.stack_bidirectional_dynamic_rnn(
    cells_fw=[cell(params.num_nodes) for _ in range(params.num_layers)],
    cells_bw=[cell(params.num_nodes) for _ in range(params.num_layers)],
    inputs=convolved,
    sequence_length=lengths,
    dtype=tf.float32,
    scope="rnn_classification")
```

查看代码来获取如何使用 `CUDA` 加速等更多实现细节。

为了创建更加紧凑且长度固定的嵌入输入，我们累积了 LSTM 的输入。首先将序列中没有数据的位置清零。

```python
mask = tf.tile(
    tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
    [1, 1, tf.shape(outputs)[2]])
zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
outputs = tf.reduce_sum(zero_outside, axis=1)
```

### _add_fc_layers

输入被完整嵌入到全连接层并用于 softmax 层。

```python
tf.layers.dense(final_state, params.num_classes)
```

### 损失、预测及优化器

最后，我们需要添加损失、训练运算符以及预测方法给 `ModelFn`：

```python
cross_entropy = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits))
# 添加优化器
train_op = tf.contrib.layers.optimize_loss(
    loss=cross_entropy,
    global_step=tf.train.get_global_step(),
    learning_rate=params.learning_rate,
    optimizer="Adam",
    # 一些梯度截断可以提升训练开始时的稳定性
    clip_gradients=params.gradient_clipping_norm,
    summaries=["learning_rate", "loss", "gradients", "gradient_norm"])
predictions = tf.argmax(logits, axis=1)
return model_fn_lib.ModelFnOps(
    mode=mode,
    predictions={"logits": logits,
                 "predictions": predictions},
    loss=cross_entropy,
    train_op=train_op,
    eval_metric_ops={"accuracy": tf.metrics.accuracy(targets, predictions)})
```

### 训练并估计模型

为了训练和评估模型，我们依赖 `Estimator` API，同时使用 `Experiment` API 来训练及评估：

```python
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=output_dir,
      config=config,
      params=model_params)
  # 训练模型
  tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=get_input_fn(
          mode=tf.contrib.learn.ModeKeys.TRAIN,
          tfrecord_pattern=FLAGS.training_data,
          batch_size=FLAGS.batch_size),
      train_steps=FLAGS.steps,
      eval_input_fn=get_input_fn(
          mode=tf.contrib.learn.ModeKeys.EVAL,
          tfrecord_pattern=FLAGS.eval_data,
          batch_size=FLAGS.batch_size),
      min_eval_frequency=1000)
```

注意，本教程只是一个相对较小数据集的简单示例，可让你快速熟悉递归神经网络和估计器的 API。 如果你在大型数据集上进行尝试，这些模型会更加强大。

在对模型进行 1M 步训练时，你可以获得约70％的精确度。值得一提的是，这种精度已经足够好了，因为用户有足够的时间来调整他们的绘图。此外，游戏不仅仅只使用了排名第一的备选项，同时如果目标类别分数高于某个固定的阈值也会接受这个答案。