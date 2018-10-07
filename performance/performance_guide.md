# 性能指南

本指南包含了一些优化 TensorFlow 代码的最佳实践，它包含以下几节内容：

*   [通用最佳实践](#通用最佳实践) 涵盖多种模型类型和硬件的通用主题。
*   [GPU 上的优化](#gpu-上的优化) 针对 GPU 的相关技巧的细节。
*   [CPU 上的优化](#cpu-上的优化) 针对 CPU 的细节。

## 通用最佳实践

下面的几节内容为涵盖多种硬件和模型的最佳实践，它们是：

*   [输入管线的优化](#输入管线的优化)
*   [数据格式](#数据格式)
*   [通用的融合操作](#通用的融合操作)
*   [RNN 性能](#rnn-performance)
*   [从源码构建和安装](#从源码构建和安装)

### 输入管线的优化

典型的模型会从磁盘加载数据，然后处理并通过网络发送出去。比如，模型按照下列数据流过程来处理 JPEG 图像：从磁盘加载图像，将 JPEG 解码到一个张量中，裁剪和边缘垫值，以及可能的翻转和变形操作，然后按批次投入训练。这个数据流被称为输入管线。随着 GPU 和其它加速硬件运行得越来越快，数据预处理就成了性能的瓶颈。

确定输入管线是否为瓶颈可能会比较复杂。一种最简单的方法是在输入管线之后将模型简化为单个操作（平凡模型），并测量其每秒处理的样例数。如果整个模型和平凡模型之间的效率差异极小，则输入管线很有可能是瓶颈。以下是确定瓶颈问题的其它一些方法：

*   通过运行 `nvidia-smi -l 2` 来检查 GPU 是否已经被充分利用。如果 GPU 利用率没有接近 80-100%，则此输入管线可能是个瓶颈。
*   生成时间线，并检查它是否有大块的空白时间段（等待时间）。生成时间线的示例参见教程 [XLA JIT](../performance/xla/jit.md)。
*   检查 CPU 使用情况。有可能出现的情况是：管线已经优化，却仍然没有足够的 CPU 周期来处理这个管线。
*   估计所需的吞吐量，并验证所使用的磁盘能够达到吞吐量的水平。因为一些云解决方案提供的磁盘速度甚至低至 50 MB/秒，这比机械磁盘（150 MB/秒）、SATA SSD （500 MB/秒）、以及 PCIe SSD （2000+ MB/秒）都要慢。

#### CPU 上的预处理

将输入管线的操作放在 CPU 上可以显著提高性能。让 CPU 处理输入管线，可以使 GPU 专注于训练。为了确保预处理是在 CPU 上进行，可将预处理操作按如下方式包装一下：

```python
with tf.device('/cpu:0'):
  # 用于获得和处理图像或数据的函数
  distorted_inputs = load_and_distort_images()
```

如果使用 `tf.estimator.Estimator`，输入函数会自动在 CPU 上执行。

#### 使用 tf.data API

[tf.data API](../guide/datasets.md) 正在取代 `queue_runner` 作为构建输入管道的推荐 API。训练示例[ResNet example](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator/cifar10_main.py) ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) 说明了使用 `tf.estimator.Estimator` 时 `tf.data` 的 API 调用。

`tf.data` API 使用 C++ 多线程，相比于 Python 的 `queue_runner`，由于 `queue_runner` 受限于 Python 的多性能，`tf.data` API 则具有较低的开销。`tf.data` API 详细的性能指南请见[这儿](../performance/datasets_performance.md)。

虽然使用 `feed_dict` 的流数据提供了很高的灵活性，但是通常 `feed_dict` 并没有提供可伸缩的解决方案。如果只使用单个 GPU，则 `tf.data` API 和 `feed_dict` 之间的性能差异可以忽略不计。我们建议避免使用 `feed_dict` 来处理琐碎的示例。特别是，避免使用大量输入的 `feed_dict`：

```python
# feed_dict often results in suboptimal performance when using large inputs.
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

#### 融合解码和裁剪

如果输入为 JPEG 图像，且需要裁剪，则使用 `tf.image.decode_and_crop_jpeg` 加速预处理。`tf.image.decode_and_crop_jpeg` 只在裁剪窗口内解码图像数据。如果裁剪窗口比整个图像小得多，则会大大加快处理速度。对于 imagenet 数据，这种方法会将输入管线处理速度提升 30%。

使用示例：

```python
def _image_preprocess_fn(image_buffer):
    # image_buffer 1-D string Tensor representing the raw JPEG image buffer.

    # Extract image shape from raw JPEG image buffer.
    image_shape = tf.image.extract_jpeg_shape(image_buffer)

    # Get a crop window with distorted bounding box.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      image_shape, ...)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Decode and crop image.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    cropped_image = tf.image.decode_and_crop_jpeg(image, crop_window)
```

`tf.image.decode_and_crop_jpeg` 可运行在所有平台上。在其他平台上，由于使用了 `libjpeg` 和 `libjpeg-turbo`，所以在 Windows 上无法加速。

#### 使用大文件

加载大量的小文件会极大地影响 I/O 性能。一种获得最大的 I/O 吞吐量的方法是将输入数据预处理为更大的 `TFRecord` 文件（约 100MB 大小）。对于较小的数据集（200MB~1GB），最好的方法通常是将整个数据集加载到内存。资料 [下载和转换为 TFRecord 格式](https://github.com/tensorflow/models/tree/master/research/slim#downloading-and-converting-to-tfrecord-format) 中介绍了创建 `TFRecords` 的相关信息和脚本，而 [脚本](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py) 可用于将 CIFAR-10 数据集转化为 `TFRecords`。

### 数据格式

数据格式是指传递给指定操作的张量结构。下面的讨论专门针对表示图像的四维张量。在 TensorFlow 中，四维张量中的部分成员常用如下一些字母来表示：

*   N 表示一个训练批次中的图像数目。
*   H 表示垂直维度（高度方向）中的像素数。
*   W 表示水平维度（宽度方向）中的像素数。
*   C 表示通道数。比如，黑白或灰度图像为 1，RGB 图像为 3。

在 TensorFlow 中，有两种命名规范分别表示最常用的两种数据格式：

*   `NCHW` 或 `channels_first`
*   `NHWC` 或 `channels_last`

TensorFlow 默认采用 `NHWC`，而在 NVIDIA GPU 上使用 [cuDNN](https://developer.nvidia.com/cudnn) 时，`NCHW` 格式是最优选择。

最佳实践是构建同时支持两种数据格式的模型。这简化了 GPU 上的训练，并在 CPU 上运行推理。如果 TensorFlow 是通过 [Intel MKL](#tensorflow_with_intel_mkl-dnn) 优化编译的，许多操作，特别是与基于 CNN 的模型相关的操作，将会得到优化并支持 `NCHW`。如果不使用 MKL，有些操作在使用 `NCHW` 时无法在 CPU 上运行。

这里我们简要介绍一下这两种格式的历史。TensorFlow 最开始使用 `NHWC` 是因为它在 CPU 上稍微快一点。但长期以来，我们一直在编写工具，让计算图可以自动重写，从而让两种格式的切换变得透明化，来实现一些优化。我们发现，尽管 `NCHW` 在一般情况下效率是最高的，但有些 GPU 操作在使用 `NHWC` 时确实更快一些。

### 通用的融合操作

融合操作是将多个操作合并为单个内核，从而提高性能。TensorFlow 自带了大量的融合操作，而且 [XLA](../performance/xla/index.md) 会尽可能地创建融合操作，来自动地提高性能。下面，我们将挑选出一些融合操作，这些操作可以极大地提高性能，但往往会被忽视。

#### 融合批量标准化

融合批量标准化（Fused batch norm）是将批量标准化所需的多个操作合并为一个内核。批量标准化是一个开销很大的过程，对于一些模型而言，它会占用很大比例的操作时间。通过使用融合批量标准化，可以实现 12%-30% 的加速。

常用的批量标准化有两种，都支持融合。TensorFlow 1.3 版本中开始支持对核心函数 `tf.layers.batch_normalization` 添加融合参数。

```python
bn = tf.layers.batch_normalization(
    input_layer, fused=True, data_format='NCHW')
```

社区贡献（contrib）中的 `tf.contrib.layers.batch_norm` 函数则从 TensorFlow 1.0 起就加入融合支持。

```python
bn = tf.contrib.layers.batch_norm(input_layer, fused=True, data_format='NCHW')
```

### RNN 性能

有许多方法来指定 TensorFlow 中的 RNN 计算，同时需要在模型的灵活性和性能之间做出权衡。`tf.nn.rnn_cell.BasicLSTMCell` 是一个参考实现，在没有其他选择的情况下，作为最后的手段。

当使用一个单元，而不是完全融合的 RNN 层，可以选择使用 `tf.nn.static_rnn 或 `tf.nn.dynamic_rnn`。在运行时通常不应该有性能差异，但是大量的 unroll 数量会增加 `tf.nn.static_rnn` 的图形大小，并导致更长的编译时间。`tf.nn.dynamic_rnn` 的另一个优点是，它可以选择性地将内存从 GPU 切换到 CPU，以支持非常长的序列的训练。根据模型和硬件配置，这可能会带来性能成本。也可以并行运行 `tf.nn.dynamic_rnn` 和底层 `tf.while_loop` 的多个迭代。尽管这在 RNN 模型中很少有用，因为它们本质上是顺序的。

在 NVIDIA GPU 上，除非需要层归一化，否则应首选 `tf.contrib.cudnn_rnn`。它与 `tf.contrib.rnn.BasicLSTMCell` 和 `tf.contrib.rnn.LSTMBlockCell` 至少在一个数量级，并比 `tf.contrib.rnn.BasicLSTMCell` 少用 3-4 倍的内存。

如果你需要一次运行 RNN 的一个步骤，就像在增强学习中经常使用的策略一样，在 `tf.while_loop` 构造函数内，应该使用 `tf.contrib.rnn.LSTMBlockCell` 与您自己的环境交互循环。一次运行 RNN 的一个步骤并返回 Python 是可能的，但它会慢一些。

在 CPU，移动端设备，以及 GPU 上不支持 `tf.contrib.cudnn_rnn` 时，最快和最高效的内存选项是 `tf.contrib.rnn.LSTMBlockFusedCell`。

对于所有不太常见的单元类型，比如 `tf.contrib.rnn.NASCell`、`tf.contrib.rnn.PhasedLSTMCell`、`tf.contrib.rnn.UGRNNCell`、`tf.contrib.rnn.GLSTMCell`、`tf.contrib.rnn.Conv1DLSTMCell`、`tf.contrib.rnn.Conv2DLSTMCell` 和 `tf.contrib.rnn.LayerNormBasicLSTMCell` 等等，我们应该意识到，它们是在类似于 `tf.contrib.rnn.BasicLSTMCell` 的图形中实现的，且同样会遭受很差的性能和高内存使用。在使用这些单元之前，应该考虑这些权衡是否值得。例如，层标准化可以加速收敛速度，因为 cuDNN 比没有使用它时的最快收敛的 20 倍速度还快。

### 从源码构建和安装

默认情况下，TensorFlow 二进制程序已经覆盖了非常广泛的硬件种类，从而让每个人都能使用 TensorFlow。如果用 CPU 来做训练或推理，建议编译 TensorFlow 时启用所有针对 CPU 的优化。对 CPU 上训练和推理的加速的文档参见[编译器优化的对比](#编译器优化的对比)。

为安装 TensorFlow 的优化得最充分的版本，你需要从源码进行[构建和安装](../install/source.md)。如果需要在目标机器上构建支持不同硬件平台的 TensorFlow，你需要在交叉编译时针对目标平台启用最高级别的优化。下面的命令展示了使用 `bazel` 针对特定平台进行编译的示例。

```bash
# 此命令针对 Intel 的 Broadwell 处理器进行优化
bazel build -c opt --copt=-march="broadwell" --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

#### 环境、构建和安装技巧

*   `./configure` 命令是为了确定在构建中包含哪些计算能力。它不影响整体性能，但会影响初始启动。运行 TensorFlow 一次之后，编译的内核会被缓存到 CUDA 中。如果使用 docker 容器，这个数据将得不到缓存，因而每次 TensorFlow 启动时都会因此而变慢。最好的办法是将需要用到的 GPU 的[计算能力](http://developer.nvidia.com/cuda-gpus)都包含进来，比如 P100 为 6.0，Titan X (Pascal) 为 6.1，Titan X (Maxwell) 为 5.2，K80 为 3.7。
*   选择一个版本的 gcc ，要求能够支持目标 CPU 能提供的所有优化。推荐的最低的 gcc 版本为 4.8.3。在 OS X 上，更新到最新的 Xcode 版本，并使用 Xcode 自带的那个版本的 clang。
*   安装 TensorFlow 能够支持的最新的稳定版 CUDA 平台和 cuDNN 库。

## GPU 上的优化

本节介绍针对 GPU 的优化技巧，这和 [通用最佳实践](#通用最佳实践) 中的内容不同。如何在多 GPU 环境下获得最优的性能是一个有挑战性的任务。常用的方法是利用数据并行机制。基于数据并行的扩展需要将模型复制数份，它们被称之为“塔（tower）”，然后将每个“塔”置于一个 GPU 上。每个塔会对一个不同批次的数据进行操作，然后更新变量。这些变量即我们所说的参数，是需要由所有塔来共享的。那么每个塔是如何获得变量更新的？梯度计算又是如何影响模型的性能、扩展、以及收敛性的呢？本节后面的部分将概述模型的塔在多个 GPU 上是如何处理那些变量的。[高性能模型](../performance/performance_models.md)中则会更详细介绍一些更复杂的方法，用于在不同塔之间共享和更新变量。

如何最好地处理变量的更新与模型、硬件、以及硬件的配置方法等因素有关。比如，两个系统都用 NVIDIA Tesla P100s，但是一个使用的是 PCIe 而另一个却是 [NVLink](http://www.nvidia.com/object/nvlink.html)。在这种情况下，两者的最优方案可能就不一样了。对于真实世界的例子，请参考 [benchmark](../performance/benchmarks.md) 页面中关于多种平台上的最优设置的介绍。我们对几个平台和配置进行了基准测试，下面是摘要：

*   **Tesla K80**： 如果多个 GPU 位于同一个 PCI Express 根联合体上，且相互之间能够使用 [NVIDIA GPUDirect](https://developer.nvidia.com/gpudirect) 技术相通信，则将变量均匀地分布在这些 GPU 上进行训练是最好的方法。如果不能使用 GPUDirect，则变量放在 CPU 上是最好的办法。

*   **Titan X (Maxwell 和 Pascal)、 M40、P100、及类似型号**： 对于像 ResNet 和 InceptionV3 这样的模型，将变量放在 CPU 上是最优选择，但是对于变量很多的模型，比如 AlexNet 和 VGG，结合 `NCCL` 使用 GPU 会更好一些。

将变量放在哪个设备上有一个通用的方法，那就是编写一个方法来为每个操作确定放置的位置，然后在调用 `with tf.device():` 的时候用这个方法，而不要直接使用具体的设备名。考虑在两个 GPU 上训练一个模型的场景，假定其变量放在 CPU 上。那么我们需要用一个循环来为这每个 GPU 来生成和放置一个“塔”。一种惯用的放置方法是查看每个操作的类型，如果类型为 `Variable`、`VariableV2` 或 `VarHandleOp`，则此方法判断它们应该被放在 CPU 上，而其它所有操作被判定应该放在 GPU 上。所以，计算图的构建过程应该是：

*    在第一个循环中，为 `gpu:0` 创建一个模型的塔。在放置操作的过程中，我们的放置方法确定变量应放在 `cpu:0` 上，而所有其它操作应该放在 `gpu:0` 上。
*    在第二个循环中，`reuse` 设为 `True`，表示变量要被重用，然后为 `gpu:1` 生成一个“塔”。在放置这个塔上的操作时，那些已经被放在 `cpu:0` 上的变量会被重用，而所有其它的操作则被放在 `gpu:1` 上。

最后的结果是，所有的变量都被放在 CPU 上，而每个 GPU 上都有拷贝有一份模型中所有的计算操作。

下面的代码片断展示了两种不同的变量放置方法：一种是将变量放在 CPU 上，另一种是将变量均匀分布在各个 GPU 上。

```python

class GpuParamServerDeviceSetter(object):
  """用 with tf.device() 来将变量放置在负载最小的 GPU 上。

    这个类的通常用法是传入一个 GPU 设备列表变量 ps_devices，其值类似于 ['gpu:0', 'gpu:1','gpu:2']。
    当放置变量时，每个变量会被放在负载最小的 gpu 上。所有其它操作，即那些计算操作，将被放在 worker_device上。
  """

  def __init__(self, worker_device, ps_devices):
    """GpuParamServerDeviceSetter 的初始化函数
    参数：
      worker_device: 用于计算操作的设置
      ps_devices：一个设置列表，用于变量操作。每个变量被指定到最小负载的设备上
    """
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
      return self.worker_device

    # 从 ps_devices 中获得最小负载的设备
    device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name

def _create_device_setter(is_cpu_ps, worker, num_gpus):
  """创建设备设置器对象。"""
  if is_cpu_ps:
    # 如果 tf.train.replica_device_setter 支持在 CPU 上放置变量，所有放在一个 GPU 上，
    # 或放在 cluster_spec 中定义的服务器上（ps_servers）
    return tf.train.replica_device_setter(
        worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
  else:
    gpus = ['/gpu:%d' % i for i in range(num_gpus)]
    return ParamServerDeviceSetter(worker, gpus)

# 本方法是一个完整例子中摘出来的一段代码经修改而得到的
def _resnet_model_fn():
    # 当设置为 False 时，变量会被放置在最小负载 GPU 上，如果设置为 True，变量会被放置在 CPU 上
    is_cpu_ps = False

    # 遍历所有 GPU，为每个 GPU 生成一个模型的“塔”副本
    for i in range(num_gpus):
      worker = '/gpu:%d' % i
      # 创建一个设备设置器，用于确定操作的放置位置
      device_setter = _create_device_setter(is_cpu_ps, worker, FLAGS.num_gpus)
      # 在第一个循环中创建变量，在接下来的循环中，reuse 被设置为 True
      # 于是，那些“塔”会共享这些变量
      with tf.variable_scope('resnet', reuse=bool(i != 0)):
        with tf.name_scope('tower_%d' % i) as name_scope:
          # tf.device 为创建的每个操作调用 device_setter
          # device_setter 返回此操作将要放置的设备
          with tf.device(device_setter):
            # 创建“塔”
            _tower_fn(is_training, weight_decay, tower_features[i],
                      tower_labels[i], tower_losses, tower_gradvars,
                      tower_preds, False)

```
在不远的将来，上述代码将只会用于演示目的，因为我们会推出高层次接口来支持主流的设备放置方法，其使用也要容易地多。在我们逐步扩展 API 并更好地支持多 GPU 场景的同时，[示例](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator)也会持续更新。

## CPU 上的优化

只要[从源码构建](../install/source.md) TensorFlow，并启用目标 CPU 所支持的那些优化指令，包括 Intel® Xeon Phi™ 在内的 CPU 是可以实现最优性能的，

除了使用了最新的指令集，Intel® 还在 Intel® 深度神经网络数学核心库（Math Kernel Library，Intel® MKL-DNN）中加入了对 TensorFlow 的支持。虽然用词不完全准确，这些优化还是常被简称为 “MKL”，或“基于 MKL 的 TensorFlow”。[基于 Intel® MKL-DNN 的 TensorFlow](#tensorflow_with_intel_mkl_dnn) 中详细介绍了 MKL 优化。

下面两种配置是通过调整线程池来优化 CPU 性能：

*   `intra_op_parallelism_threads`：使用多线程来并行化计算的结点会将不同的计算单元分配到该池中的线程上
*   `inter_op_parallelism_threads`：所有待计算结点都由此线程池来调度

这些配置是通过 `tf.ConfigProto` 来设置的，如下面代码所示，将其作为 `config` 参数传递到 `tf.Session` 即可。对于这两种配置，如果都没有设置或设置为 0，则会默认使用逻辑 CPU 核心的数目。对于许多系统，包括 4 核的 CPU，以及包含70 多个逻辑核心的多 CPU 系统，测试都显示默认设置已经非常高效。另一种常用策略是将线程池的大小设置为物理核的数目，而非逻辑核的数目。

```python
  config = tf.ConfigProto()
  config.intra_op_parallelism_threads = 44
  config.inter_op_parallelism_threads = 44
  tf.Session(config=config)
```

在 [编译器优化对比](#comparing-compiler-optimizations) 中，介绍了不同编译器优化的测试结果。

### 在 TensorFlow 中使用 Intel® MKL DNN

通过使用 Intel® 深度神经网络数学核心库的优化指令，Intel® 在 Xeon 和 Xeon Phi™ 芯片中加入对 TensorFlow 的优化。这些优化也为消费级处理器提供了加速，比如 i5 和 i7 的 Intel 处理器。Intel 公司还发布了文章[在现代 Intel® 架构上的 TensorFlow* 优化](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture)，其中披露了实现的更多细节。

> 注意：TensorFlow 从 1.2 版本开始加入了对 MKL 的支持，但是目前只支持 Linux 平台。
> 而且，即使使用了 `--config=cuda`，也是无法使用 MKL 的。

除了显著地改善了基于 CNN 的模型的训练效率，用 MKL 编译的代码针对 AVX 和 AVX2 也进行了优化。因而，对于大部分现代处理器而言（2011年之后），一次编译就可同时满足优化和兼容性需求。

TensorFlow 可用下面的命令来在编译中加入 MKL 优化，不同的 TensorFlow 版本会有差异。

对于 1.3.0 版本以后的 TensorFlow 源码：

```bash
./configure
# 选择所需的选项
bazel build --config=mkl --config=opt //tensorflow/tools/pip_package:build_pip_package

```

对于从 1.2.0 到 1.3.0 之间的版本：

```bash
./configure
Do you wish to build TensorFlow with MKL support? [y/N] Y
Do you wish to download MKL LIB from the web? [Y/n] Y
# 剩余的选项用默认值即可

bazel build --config=mkl --copt="-DEIGEN_USE_VML" -c opt //tensorflow/tools/pip_package:build_pip_package

```

#### MKL 调参以实现性能最优

本节详细介绍不同的配置和环境变量，用来调节 MKL 实现最优性能。在调整这些不同的环境变量之前，需要确认模型正在使用的是 `NCHW` [数据格式](#数据格式)，即 `通道优先（channels_first）`格式。MKL 是针对 `NCHW` 来做优化的，至于 `NHWC`，Intel 对它的优化工作还在进行当中。

MKL 使用下列环境变量来调节性能：

*   KMP_BLOCKTIME - 设置线程在执行完一个并行计算区域之后进入睡眠之前的等待时间，单位是毫秒
*   KMP_AFFINITY - 启用运行时库来将线程绑定到物理处理单元上
*   KMP_SETTINGS - 启用（true）或禁用（false）程序执行过程中 OpenMP* 运行时库环境变量的打印
*   OMP_NUM_THREADS - 指定使用的线程数

关于 KMP 变量的更多详情参见 [Intel网站](https://software.intel.com/en-us/node/522775)，OMP 变量则参见 [gnu.org](https://gcc.gnu.org/onlinedocs/libgomp/Environment-Variables.html)。

虽然调节这些环境变量获益良多（后面会讨论到），但简化版的建议为：将 `inter_op_parallelism_threads` 设置为物理 CPU 核心数目，并设置下面的环境变量：

*   KMP_BLOCKTIME=0
*   KMP_AFFINITY=granularity=fine,verbose,compact,1,0

下面是用命令参数设置 MKL 变量的示例：

```bash
KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,verbose,compact,1,0 \
KMP_SETTINGS=1 python your_python_script.py
```

下面是用 python 的 `os.environ` 来设置 MKL 变量的示例：

```python
os.environ["KMP_BLOCKTIME"] = str(FLAGS.kmp_blocktime)
os.environ["KMP_SETTINGS"] = str(FLAGS.kmp_settings)
os.environ["KMP_AFFINITY"]= FLAGS.kmp_affinity
if FLAGS.num_intra_threads > 0:
  os.environ["OMP_NUM_THREADS"]= str(FLAGS.num_intra_threads)

```

不同的设置会让一些模型和硬件平台受益。下面，讨论了影响性能的每一个变量。

*   **KMP_BLOCKTIME**：MKL 中默认为 200ms，但这在我们的测试中并不是最优的。在我们的测试中，0 (0ms) 对于基于 CNN 的模型是一个不错的默认值。对于 AlexNet 模型，最优值为 30ms，而 GoogleNet 和 VGG11 都为 1ms。

*   **KMP_AFFINITY**：建议设置为 `granularity=fine,verbose,compact,1,0` 。

*   **OMP_NUM_THREADS**：默认值为物理核心数目。调整此参数时，如果其值超过核心数目，则会对某些模型在 Intel® Xeon Phi™ (Knights Landing) 芯片上的性能产生影响。关于 Intel 优化的详情，参见 [现代 Intel® 架构上的 TensorFlow* 优化](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture)一文。 

*   **intra_op_parallelism_threads**：推荐设置为物理核心数目。默认值为 0，该值设置的是逻辑核心数目，这对于某些架构而言，是一个可行的选项。这个变量的值应该和 `OMP_NUM_THREADS` 保持一致。

*   **inter_op_parallelism_threads**：推荐设置为套接字数目。默认为 0，意为逻辑核心数目。

### 编译器优化的对比

下面的内容中整理了在不同平台、不同类型 CPU、以及不同编译器优化的情况下的训练和推理时的性能测试结果。我们测试的模型包括 ResNet-50 ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) 和 InceptionV3 ([arXiv:1512.00567](https://arxiv.org/abs/1512.00567))。

对于每个测试，当用到 MKL 优化时，环境变量 KMP_BLOCKTIME 都被设置为 0 (0ms)，而 KMP_AFFINITY 被设置为 `granularity=fine,verbose,compact,1,0`。

#### InceptionV3 的推理

**环境**

*   实例类型：AWS EC2 m4.xlarge
*   CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz (Broadwell)
*   数据集: ImageNet
*   TensorFlow 版本： 1.2.0 RC2
*   测试脚本： [tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/blob/mkl_experiment/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)

**每批次样本数目：1**

MKL 测试所执行的命令：

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=inception3 --data_format=NCHW \
--batch_size=1 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

|     优化      |    数据格式  |  图像数目/秒 <br> (每步时间)  |  Intra 线程数  |   Inter 线程数 |
| ------------ | ----------- | ------------ | ------------- | ------------- |
| AVX2         | NHWC        | 7.0 (142ms)  | 4             | 0             |
| MKL          | NCHW        | 6.6 (152ms)  | 4             | 1             |
| AVX          | NHWC        | 5.0 (202ms)  | 4             | 0             |
| SSE3         | NHWC        | 2.8 (361ms)  | 4             | 0             |

**每批次样本数目：32**

MKL 测试所执行的命令：

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=inception3 --data_format=NCHW \
--batch_size=32 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

|     优化      |    数据格式  |  图像数目/秒 <br> (每步时间)  |  Intra 线程数  |   Inter 线程数 |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| MKL          | NCHW        | 10.3 (3,104ms)         | 4             | 1             |
| AVX2         | NHWC        | 7.5 (4,255ms) | 4             | 0             |
| AVX          | NHWC        | 5.1 (6,275ms) | 4             | 0             |
| SSE3         | NHWC        | 2.8 (11,428ms)| 4             | 0             |

#### 推理 ResNet-50

**环境**

*   实例类型： AWS EC2 m4.xlarge
*   CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz (Broadwell)
*   数据集: ImageNet
*   TensorFlow 版本： 1.2.0 RC2
*   测试脚本： [tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/blob/mkl_experiment/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)

**每批次样本数目：1**

MKL 测试所执行的命令：

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=resnet50 --data_format=NCHW \
--batch_size=1 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

|     优化      |    数据格式  |  图像数目/秒 <br> (每步时间)  |  Intra 线程数  |   Inter 线程数 |
| ------------ | ----------- | ------------ | ------------- | ------------- |
| AVX2         | NHWC        | 8.8 (113ms)  | 4             | 0             |
| MKL          | NCHW        | 8.5 (120ms)  | 4             | 1             |
| AVX          | NHWC        | 6.4 (157ms)  | 4             | 0             |
| SSE3         | NHWC        | 3.7 (270ms)  | 4             | 0             |

**每批次样本数目：32**

MKL 测试所执行的命令：

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=resnet50 --data_format=NCHW \
--batch_size=32 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

|     优化      |    数据格式  |  图像数目/秒 <br> (每步时间)  |  Intra 线程数  |   Inter 线程数 |
| ------------ | ----------- | ------------- | ------------- | ------------- |
| MKL          | NCHW        | 12.4 (2,590ms)      | 4             | 1             |
| AVX2         | NHWC        | 10.4 (3,079ms)| 4             | 0             |
| AVX          | NHWC        | 7.3 (4,4416ms)| 4             | 0             |
| SSE3         | NHWC        | 4.0 (8,054ms) | 4             | 0             |

#### 训练 InceptionV3

**环境**

*   实例类型： Dedicated AWS EC2 r4.16xlarge (Broadwell)
*   CPU: Intel Xeon E5-2686 v4 (Broadwell) Processors
*   数据集： ImageNet
*   TensorFlow 版本： 1.2.0 RC2
*   测试脚本： [tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/blob/mkl_experiment/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)

MKL 测试所执行的命令：

```bash
python tf_cnn_benchmarks.py --device=cpu --mkl=True --kmp_blocktime=0 \
--nodistortions --model=resnet50 --data_format=NCHW --batch_size=32 \
--num_inter_threads=2 --num_intra_threads=36 \
--data_dir=<path to ImageNet TFRecords>
```

| 优化 | 数据格式 | 图像数目/秒   | Intra 线程数 | Inter 线程数 |
------------ | ----------- | ---------- | ------------- | -------------
MKL          | NCHW        | 20.8       | 36            | 2
AVX2         | NHWC        | 6.2        | 36            | 0
AVX          | NHWC        | 5.7        | 36            | 0
SSE3         | NHWC        | 4.3        | 36            | 0

另外，我们还让 ResNet 和 [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 在此配置上以一种自组织（ad hoc）方式运行。只不过测试次数还不足以发布一份明确的结果。不过，不完整的测试结果很大程度上与上表结果类似，即 MKL 的效率为 AVX2 三倍多。
