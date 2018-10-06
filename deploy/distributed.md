# 分布式 TensorFlow

本文演示了怎样创建一个 TensorFlow 集群（cluster），以及怎样向集群提交计算图（graph）。我们假设你已经对基础的 TensorFlow 编程所需要用到的[基本概念](../guide/low_level_intro.md)有所了解。

## 你好，分布式 TensorFlow ！

要查看一个简单的 TensorFlow 集群，请执行以下操作：

```shell
# 以单进程“集群”模式启动一个 TensorFlow 服务器
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server()
>>> sess = tf.Session(server.target)  # 在服务器上创建一个会话
>>> sess.run(c)
'Hello, distributed TensorFlow!'
```

`tf.train.Server.create_local_server` 方法使用进程内服务器创建了一个单进程集群。

## 创建一个集群

<div class="video-wrapper">
  <iframe class="devsite-embedded-youtube-video" data-video-id="la_M6bCV91M"
          data-autohide="1" data-showinfo="0" frameborder="0" allowfullscreen>
  </iframe>
</div>

TensorFlow “集群”是一组参与分布式执行 TensorFlow 计算图的“任务（Task）”集合。每个任务都与一个 TensorFlow 服务器（Server） 相关联，TensorFlow 服务器中包含一个可以用来创建会话（sessions）的 `Master`，和一个在计算图中执行命令的 `Worker`。一个集群同样可以被分为一个或多个“作业（Job）”，每个作业又包含一个或多个任务。（译者注：集群由任务组成，任务被包含在特定作业中）

要创建一个群集，我们在群集中为每个任务启动一个 TensorFlow 服务器。通常每个任务运行在不同的机器上，但是这里我们在一台机器上运行多个任务（例如，控制不同的 GPU 设备）。 我们在每个任务中都做如下操作：

1. 在集群中**创建一个描述所有任务的 `tf.train.ClusterSpec`**。它对每个任务而言都应该是相同的。

2. **创建一个 `tf.train.Server`**，将 `tf.train.ClusterSpec` 传给构造函数，并用工作名称标识本地任务和任务索引。


### 创建一个 `tf.train.ClusterSpec` 来描述集群

群集规范（ClusterSpec）是一个将作业名称映射到网络地址列表地址的字典。把该字典传递给 `tf.train.ClusterSpec` 构造函数。例如：

<table>
  <tr><th>构造 <code>tf.train.ClusterSpec</code> </th><th>可用的任务</th>
  <tr>
    <td><pre>
tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
</pre></td>
<td><code>/job:local/task:0<br/>/job:local/task:1</code></td>
  </tr>
  <tr>
    <td><pre>
tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222",
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})
</pre></td><td><code>/job:worker/task:0</code><br/><code>/job:worker/task:1</code><br/><code>/job:worker/task:2</code><br/><code>/job:ps/task:0</code><br/><code>/job:ps/task:1</code></td>
  </tr>
</table>

### 在每个任务中创建一个 `tf.train.Server` 实例

一个 `tf.train.Server` 对象包含一套本地设备，一套与 `tf.train.ClusterSpec` 中其他任务相连的连接，以及一个可以用来执行分布式计算的 `tf.Session`。 每个 TensorFlow 服务器都是特定命名作业的成员，并拥有一份该作业中的任务索引。TensorFlow 服务器可以与集群中其他服务器通信。

例如，启动一个运行在 `localhost：2222` 和 `localhost：2223` 两台服务器上的集群，在本地机器的两个不同进程上运行以下代码：

```python
# 任务 0 中:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)
```
```python
# 任务 1 中:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
```

**注意：** 手动指定这些集群规范可能很乏味，特别是对于大型集群。我们正在开发可编程的任务启动工具，例如类似 [Kubernetes](http://kubernetes.io) 的集群管理器。如果你希望 Tensorflow 支持某种特定的集群管理器，请提出一个 [GitHub issue](https://github.com/tensorflow/tensorflow/issues)。

## 指定模型中的分布式设备

要将操作放在特定的进程上，可以使用同一个 `tf.device` 函数来指定，它同样被用来指定操作是在 CPU 还是 GPU 上执行。 例如

```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7.example.com:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)
```

在上面的例子中，变量是在 `ps` 作业中的两个任务上创建的，模型的计算密集部分是在 `worker` 作业中创建的。TensorFlow 将在作业之间插入适当的数据传输（正向传递时从 `ps` 到 `worker`，反向传递时从 `worker` 到 `ps`）。

## 重复训练

一种通用训练的配置，也被称为“并行数据”，包含了使用不同 mini-batch 来训练相同模型的 `Worker` 作业中的多个任务，更新 `ps` 作业中一个或多个任务里的共享参数。所有任务通常在不同的机器上运行。在 TensorFlow 中有很多方法可以指定任务分配的结构，我们正在开发简化指定复制模型工作的库。可能的方法包括：

* **图内复制** 在这种方法中，客户端构建一个包含一组参数（在 `tf.Variable` 节点上固定到 `/job:ps`）的 `tf.Graph`；以及模型的计算密集型部分的多个副本，每个副本固定对应到 `/job:worker` 中不同的任务上。

* **图间复制** 在这种方法中，每个 `/job:worker` 任务都对应一个独立的客户端，客户端通常与 worker 任务在同一进程中。每个客户端会构建一个相似的、带参数的图（这些参数像以往一样，通过 `tf.train.replica_device_setter` 来映射到相同任务 `/job:ps`）；和一个模型中的计算密集型部分的单一副本，对应到 `/job:worker` 中的本地任务。

* **异步训练** 在这种方法中，图的每个副本都有一个没有独立训练循环，不做协调就可以执行。它是兼容的以上两种形式的复制。

* **同步训练** 在这种方法中，所有的副本读取到相同的值赋给当前的参数，并行计算梯度，然后将它们一起应用。它与图内复制（例如：像 [CIFAR-10 multi-GPU trainer](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py) 一样使用梯度平均和多 GPU 图间复制），图间复制（使用 `tf.train.SyncReplicasOptimizer`）。

### 总结：示例训练程序

以下代码显示了分布式训练程序的框架，实现**图间复制**和**异步训练**。它包括参数服务器和 `Worker` 任务的代码。

```python
import argparse
import sys

import tensorflow as tf

FLAGS = None


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # 从参数服务器和工作主机创建一个集群
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # 创建并启动本地任务的服务器
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # 默认情况下将操作分配给本地 Worker
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # 建立模型...
      loss = ...
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # StopAtStepHook 在运行给定步骤后处理停止
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # MonitoredTrainingSession 负责会话初始化
    # 从检查点恢复，保存到检查点，一旦完成或报错就关闭
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # 异步运行训练
        # 有关如何执行同步训练的更多信息，请参见 `tf.train.SyncReplicasOptimizer`
        # mon_sess.run 在被抢占 PS 的情况下处理 AbortedError
        mon_sess.run(train_op)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # 用于定义 tf.train.ClusterSpec 的标志
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```

要启动两个参数服务器和两个 `Worker` 的训练，请使用下面的命令行脚本（假设脚本被称为 `trainer.py`）

```shell
# On ps0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=1
```

## Glossary

**客户端**

客户端通常是一个程序，用来构建 TensorFlow 计算图和创建用于与集群交互的会话 `tensorflow::Session`。通常用 Python 或 C++ 编写。一个客户端进程可以直接与多个 TensorFlow 服务器交互（参阅上面的“重复训练”），一台服务器可为多个客户端服务。


**集群**

一个 TensorFlow 集群包含一个或多个“作业”，每个“作业”分为一个个列表，列表由一个或多个“任务”组成。集群通常专用于特定的高级用途，比如训练神经网络，并行使用多台机器。一个集群由 `tf.train.ClusterSpec` 对象定义。

**作业**

一份作业包括一份“任务”清单，通常用于一个共同的目的。例如，名为 `ps`（即 parameter server，参数服务器）的作业通常包括存储和更新变量的节点; 而名为 `worker` 的作业通常包括执行计算密集型任务的无状态节点。工作中的任务通常运行在不同的机器上。这套工作角色是灵活的：例如，`Worker` 可能会保持某种状态。

**主服务**

提供远程调用控制一组分布式设备的 RPC 服务，并作为会话目标。 主服务实现了 `tensorflow::Session` 接口，负责协调一个或多个 `worker服务`。所有的 TensorFlow 服务器都实现了 Master 服务。

**任务**

任务对应于特定的 TensorFlow 服务器，并且通常对应于到一个进程。一个任务属于一个特定的“作业”，并在该作业列表的索引中被唯一标识。

**TensorFlow 服务器**

运行着 `tf.train.Server` 实例的进程，是集群的成员，并对外提供 `master 服务` 和 `worker 服务`。


**Worker 服务**

一个使用本地设备执行 TensorFlow 计算图中部分命令的 RPC 服务。Worker 服务实现了 [worker_service.proto](https://www.tensorflow.org/code/tensorflow/core/protobuf/worker_service.proto)。所有的 TensorFlow 服务器都实现了 Worker 服务。

