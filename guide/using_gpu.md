# 使用（多个）GPU

## 支持的设备

在一个典型的系统上，有多个计算设备。在 TensorFlow 中，支持的设备类型是“CPU”和“GPU”。 它们都是字符串形式。
例如：

*   `"/cpu:0"`：你的机器上的 CPU。
*   `"/device:GPU:0"`：你的机器上的 GPU（如果有的话）。
*   `"/device:GPU:1"`：你的机器上的第二块 GPU ，以此类推。

如果某个 TensorFlow 的操作同时有 CPU 和 GPU 的实现，当它被分配给设备（以执行）时，GPU 将被优先考虑。 例如，`matmul` 有 CPU 和 GPU 的内核实现，在同时具备“cpu：0”和“gpu：0“设备的系统上，将选择“gpu：0”来执行“matmul”。

## 设备配置信息日志记录

为了解你的操作和张量被分配给了哪些设备，请创建 session ，并将 log_device_placement 配置选项置为 True。

```python
# 创建一个 graph。
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# 创建一个 session ，并将 log_device_placement 设置为 True。
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 执行这个操作。
print(sess.run(c))
```

将会看到以下输出:

```
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/device:GPU:0
a: /job:localhost/replica:0/task:0/device:GPU:0
MatMul: /job:localhost/replica:0/task:0/device:GPU:0
[[ 22.  28.]
 [ 49.  64.]]

```

## 手动分配设备

如果你希望某个特定操作在你选择的设备上运行，而非自动选择，可以使用 `tf.device` 创建设备的上下文，如此一来，该上下文内部的所有操作都将使用你所指定的设备运行。

```python
# 创建一个 graph。
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# 创建一个 session ，并将 log_device_placement 设置为 True。
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 执行这个操作。
print(sess.run(c))
```

可以看到 `a` 和 `b` 当前被分配给了 `cpu:0`。由于 `Matmul` 操作没有被指定设备，TensorFlow 运行时会基于当前操作和可用设备进行选择，还会在设备间自动复制张量（如果要求的话）。



```
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/cpu:0
a: /job:localhost/replica:0/task:0/cpu:0
MatMul: /job:localhost/replica:0/task:0/device:GPU:0
[[ 22.  28.]
 [ 49.  64.]]
```

## 允许 GPU 显存增长


默认情况下，Tensorflow 会使用所有 GPU 上的几乎所有的显存（取决于系统环境变量 [`CUDA_VISIBLE_DEVICES`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)）去运行程序。这样做是为了通过减少[内存碎片](https://en.wikipedia.org/wiki/Fragmentation_\(computing\))来更有效利用设备上相对宝贵的GPU显存资源。

在某些情况下，进程仅分配一部分可用显存或视进程需要再行增加显存使用量这种做法是可取的。TensorFlow 在 Session 上提供了两个 Config 选项来设置。

第一个选项是 `allow_growth`，尝试仅分配尽可能多的支持运行的 GPU 显存：它一开始分配很少的内存，当 Sessions 运行并需要更多的 GPU 显存时，拓展 Tensorflow 程序所需要的 GPU 显存区域。注意，我们不会释放显存，因为这会导致更加严重的内存碎片问题。可以在 ConfigProto 打开这个选项：

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
```

第二个选项是 `pre_process_gpu_memory_fraction`，它决定了每个可见的 GPU 应该被分配多大比例的显存。例如，对于每个 GPU，你想让 TensorFlow 仅仅分配总显存的 40%，就这么做：

```python
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
```

如果您想真正限制可用于 TensorFlow 进程的 GPU 显存用量，这是非常有用的。

## 在一个多 GPU 机器上使用单个 GPU

如果你的机器上有不止一个 GPU ，Tensorflow 将默认使用 ID 编号最小的那个。如果你想在别的 GPU 上运行，你需要明确指定 GPU 的 ID：

```python
# 创建一个 graph。
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# 创建一个 session ，并将 log_device_placement 设置为 True。
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 执行这个操作。
print(sess.run(c))
```

如果你指定的设备不存在，你会得到一个错误 ：`InvalidArgumentError` 。

```
InvalidArgumentError: Invalid argument: Cannot assign a device to node 'b':
Could not satisfy explicit device specification '/device:GPU:2'
   [[{{node b}} = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [3,2]
   values: 1 2 3...>, _device="/device:GPU:2"]()]]
```

如果你想让 Tensorflow 自动选择现有且受支持的设备来运行操作，以防指定的设备不存在，你可以在创建 session 时在配置选项中将 `allow_soft_placement` 设置为 `True`。

```python
# 创建一个 graph。
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# 创建一个 session ，并将 allow_soft_placement 和 log_device_placement 设置为 True。
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# 执行这个操作。
print(sess.run(c))
```

## 使用多个 GPU

如果你想在多个 GPU 上运行 Tensorflow ，可以采用 multi-tower 的方式构建模型，其中每个 tower 分配给不同的 GPU 。
例如：

```python
# 创建一个 graph。
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# 创建一个 session ，并将 log_device_placement 设置为 True。
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 执行这个操作。
print(sess.run(sum))
```

将会看到以下输出：

```
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K20m, pci bus
id: 0000:02:00.0
/job:localhost/replica:0/task:0/device:GPU:1 -> device: 1, name: Tesla K20m, pci bus
id: 0000:03:00.0
/job:localhost/replica:0/task:0/device:GPU:2 -> device: 2, name: Tesla K20m, pci bus
id: 0000:83:00.0
/job:localhost/replica:0/task:0/device:GPU:3 -> device: 3, name: Tesla K20m, pci bus
id: 0000:84:00.0
Const_3: /job:localhost/replica:0/task:0/device:GPU:3
Const_2: /job:localhost/replica:0/task:0/device:GPU:3
MatMul_1: /job:localhost/replica:0/task:0/device:GPU:3
Const_1: /job:localhost/replica:0/task:0/device:GPU:2
Const: /job:localhost/replica:0/task:0/device:GPU:2
MatMul: /job:localhost/replica:0/task:0/device:GPU:2
AddN: /job:localhost/replica:0/task:0/cpu:0
[[  44.   56.]
 [  98.  128.]]
```

作为一个优秀示例，[cifar10 教程](../tutorials/images/deep_cnn.md) 演示了如何使用多个 GPU 进行训练。
