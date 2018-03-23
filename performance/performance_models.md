# 高性能模型

本文及相关
[脚本](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)
说明了如何构建能应对多种系统类型及网络拓扑的高可用模型。本文的技术利用了一些 TensorFlow Python 的底层组件。其中的大部分技术将来将被整合进高层次的 API 里。

## 输入管道

@{$performance_guide$Performance Guide} 解释了如何识别可能的输入管道问题和最佳实践。我们发现像类似采用 [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 训练 ImageNet 这种使用大量输入并每秒处理大量采样的场景下，采用 @{tf.FIFOQueue} 和 @{tf.train.queue_runner} 不能充分利用目前的 GPU 计算资源。
这是因为底层实现采用的 Python 进程引入的额外开销太大导致的。

我们在
[脚本](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) 中采用的另一种方式是采用 TensorFlow 原生的并行机制来构建的输入管道。我们的实现由3个阶段构成：

*   I/O 读取： 从硬盘选择并读取图像。
*   图像处理： 将图像记录解码成图像，预处理并组织成 mini-batch 。
*   CPU-to-GPU 数据转移：将图像从 CPU 转移到 GPU。

每个阶段的关键步骤可以采用 `data_flow_ops.StagingArea` 和其他阶段并行执行。 `StagingArea` 是类似于 @{tf.FIFOQueue} 的队列操作。不同之处在于 `StagingArea` 不保证先进先出的顺序，但提供了能在 CPU 和 GPU 上并行执行其他阶段的简单功能。将输入管道拆分为能并行执行的3个阶段是可扩展的，能充分发挥大量多核环境的优势。本章节后续将详细阐述这三个阶段以及使用 `data_flow_ops.StagingArea` 的细节。

### 并行 I/O 读取

`data_flow_ops.RecordInput` 用于处理并行从磁盘读取。对于包含 TFRecords 记录的一系列输入文件，`RecordInput` 将持续使用后台进程去读取记录。这些记录将被放入它自身的内部空间；当载入超过它一半能力的数据量后，它将产生输出张量。

这个操作有它自己的由 I/O 时间控制且消耗最少 CPU 的内部进程，这使它能平缓地与模型的其他部分并行执行。

### 并行镜像处理

从 `RecordInput` 读取图像后，它们被当做张量传递给图像处理管道。为了更容易解释图像处理管道，假设输入管道是面向 256 个批处理大小的 8 核GPU（每个 GPU 32个批处理大小）。

256 条记录被独立并行地读取和处理。它起始于图中 256 个独立的 `RecordInput` 读操作。每个读操作之后是独立并行执行的一系列相同的图像前置处理操作。图像前置处理操作包括对于图像的解码、变形、大小缩放等操作。

图像经过预处理之后，他们被连结成 8 个张量，每个张量有 32 位大小。Rather than using @{tf.concat} for this
purpose, which is implemented as a single op that waits for all the inputs to be
ready before concatenating them together, @{tf.parallel_stack} is used.
@{tf.parallel_stack} allocates an uninitialized tensor as an output, and each
input tensor is written to its designated portion of the output tensor as soon
as the input is available.

When all the input tensors are finished, the output tensor is passed along in
the graph. This effectively hides all the memory latency with the long tail of
producing all the input tensors.

### Parallelize CPU-to-GPU Data Transfer

Continuing with the assumption that the target is 8 GPUs with a batch size of
256 (32 per GPU). Once the input images are processed and concatenated together
by the CPU, we have 8 tensors each with a batch-size of 32.

TensorFlow enables tensors from one device to be used on any other device
directly. TensorFlow inserts implicit copies to make the tensors available on
any devices where they are used. The runtime schedules the copy between devices
to run before the tensors are actually used. However, if the copy cannot finish
in time, the computation that needs those tensors will stall and result in
decreased performance.

In this implementation, `data_flow_ops.StagingArea` is used to explicitly
schedule the copy in parallel. The end result is that when computation starts on
the GPU, all the tensors are already available.

### Software Pipelining

With all the stages capable of being driven by different processors,
`data_flow_ops.StagingArea` is used between them so they run in parallel.
`StagingArea` is a queue-like operator similar to @{tf.FIFOQueue} that offers
simpler functionalities that can be executed on both CPU and GPU.

Before the model starts running all the stages, the input pipeline stages are
warmed up to prime the staging buffers in between with one set of data.
During each run step, one set of data is read from the staging buffers at
the beginning of each stage, and one set is pushed at the end.

For example: if there are three stages: A, B and C. There are two staging areas
in between: S1 and S2. During the warm up, we run:

```
Warm up:
Step 1: A0
Step 2: A1  B0

Actual execution:
Step 3: A2  B1  C0
Step 4: A3  B2  C1
Step 5: A4  B3  C2
```

After the warm up, S1 and S2 each have one set of data in them. For each step of
the actual execution, one set of data is consumed from each staging area, and
one set is added to each.

Benefits of using this scheme:

*   All stages are non-blocking, since the staging areas always have one set of
    data after the warm up.
*   Each stage can run in parallel since they can all start immediately.
*   The staging buffers have a fixed memory overhead. They will have at most one
    extra set of data.
*   Only a single`session.run()` call is needed to run all stages of the step,
    which makes profiling and debugging much easier.

## Best Practices in Building High-Performance Models

Collected below are a couple of additional best practices that can improve
performance and increase the flexibility of models.

### Build the model with both NHWC and NCHW

Most TensorFlow operations used by a CNN support both NHWC and NCHW data format.
On GPU, NCHW is faster. But on CPU, NHWC is sometimes faster.

Building a model to support both data formats keeps the model flexible and
capable of operating optimally regardless of platform. Most TensorFlow
operations used by a CNN support both NHWC and NCHW data formats. The benchmark
script was written to support both NCHW and NHWC. NCHW should always be used
when training with GPUs. NHWC is sometimes faster on CPU. A flexible model can
be trained on GPUs using NCHW with inference done on CPU using NHWC with the
weights obtained from training.

### Use Fused Batch-Normalization

The default batch-normalization in TensorFlow is implemented as composite
operations. This is very general, but often leads to suboptimal performance. An
alternative is to use fused batch-normalization which often has much better
performance on GPU. Below is an example of using @{tf.contrib.layers.batch_norm}
to implement fused batch-normalization.

```python
bn = tf.contrib.layers.batch_norm(
          input_layer, fused=True, data_format='NCHW'
          scope=scope)
```

## Variable Distribution and Gradient Aggregation

During training, training variable values are updated using aggregated gradients
and deltas. In the benchmark script, we demonstrate that with the flexible and
general-purpose TensorFlow primitives, a diverse range of high-performance
distribution and aggregation schemes can be built.

Three examples of variable distribution and aggregation were included in the
script:

*   `parameter_server` where each replica of the training model reads the
    variables from a parameter server and updates the variable independently.
    When each model needs the variables, they are copied over through the
    standard implicit copies added by the TensorFlow runtime. The example
    [script](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)
    illustrates using this method for local training, distributed synchronous
    training, and distributed asynchronous training.
*   `replicated` places an identical copy of each training variable on each
    GPU. The forward and backward computation can start immediately as the
    variable data is immediately available. Gradients are accumulated across all
    GPUs, and the aggregated total is applied to each GPU's copy of the
    variables to keep them in sync.
*   `distributed_replicated` places an identical copy of the training parameters
    on each GPU along with a master copy on the parameter servers. The forward
    and backward computation can start immediately as the variable data is
    immediately available. Gradients are accumulated across all GPUs on each
    server and then the per-server aggregated gradients are applied to the
    master copy. After all workers do this, each worker updates its copy of the
    variable from the master copy.

Below are additional details about each approach.

### Parameter Server Variables

The most common way trainable variables are managed in TensorFlow models is
parameter server mode.

In a distributed system, each worker process runs the same model, and parameter
server processes own the master copies of the variables. When a worker needs a
variable from a parameter server, it refers to it directly. The TensorFlow
runtime adds implicit copies to the graph to make the variable value available
on the computation device that needs it. When a gradient is computed on a
worker, it is sent to the parameter server that owns the particular variable,
and the corresponding optimizer is used to update the variable.

There are some techniques to improve throughput:

*   The variables are spread among parameter servers based on their size, for
    load balancing.
*   When each worker has multiple GPUs, gradients are accumulated across the
    GPUs and a single aggregated gradient is sent to the parameter server. This
    reduces the network bandwidth and the amount of work done by the parameter
    servers.

For coordinating between workers, a very common mode is async updates, where
each worker updates the master copy of the variables without synchronizing with
other workers. In our model, we demonstrate that it is fairly easy to introduce
synchronization across workers so updates for all workers are finished in one
step before the next step can start.

The parameter server method can also be used for local training, In this case,
instead of spreading the master copies of variables across parameters servers,
they are either on the CPU or spread across the available GPUs.

Due to the simple nature of this setup, this architecture has gained a lot of
popularity within the community.

This mode can be used in the script by passing
`--variable_update=parameter_server`.

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" alt="parameter_server mode in distributed training"
   src="../images/perf_parameter_server_mode_doc.png">
</div>

### Replicated Variables

In this design, each GPU on the server has its own copy of each variable. The
values are kept in sync across GPUs by applying the fully aggregated gradient to
each GPU's copy of the variable.

The variables and data are available at the start of training, so the forward
pass of training can start immediately. Gradients are aggregated across the
devices and the fully aggregated gradient is then applied to each local copy.

Gradient aggregation across the server can be done in different ways:

*   Using standard TensorFlow operations to accumulate the total on a single
    device (CPU or GPU) and then copy it back to all GPUs.
*   Using NVIDIA® NCCL, described below in the NCCL section.

This mode can be used in the script by passing `--variable_update=replicated`.

### Replicated Variables in Distributed Training

The replicated method for variables can be extended to distributed training. One
way to do this like the replicated mode: aggregate the gradients fully across
the cluster and apply them to each local copy of the variable. This may be shown
in a future version of this scripts; the scripts do present a different
variation, described here.

In this mode, in addition to each GPU's copy of the variables, a master copy is
stored on the parameter servers. As with the replicated mode, training can start
immediately using the local copies of the variables.

As the gradients of the weights become available, they are sent back to the
parameter servers and all local copies are updated:

1.  All the gradients from the GPU on the same worker are aggregated together.
2.  Aggregated gradients from each worker are sent to the parameter server that
    owns the variable, where the specified optimizer is used to update the
    master copy of the variable.
3.  Each worker updates its local copy of the variable from the master. In the
    example model, this is done with a cross-replica barrier that waits for all
    the workers to finish updating the variables, and fetches the new variable
    only after the barrier has been released by all replicas. Once the copy
    finishes for all variables, this marks the end of a training step, and a new
    step can start.

Although this sounds similar to the standard use of parameter servers, the
performance is often better in many cases. This is largely due to the fact the
computation can happen without any delay, and much of the copy latency of early
gradients can be hidden by later computation layers.

This mode can be used in the script by passing
`--variable_update=distributed_replicated`.


<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" alt="distributed_replicated mode"
   src="../images/perf_distributed_replicated_mode_doc.png">
</div>

#### NCCL

In order to broadcast variables and aggregate gradients across different GPUs
within the same host machine, we can use the default TensorFlow implicit copy
mechanism.

However, we can instead use the optional NCCL (@{tf.contrib.nccl}) support. NCCL
is an NVIDIA® library that can efficiently broadcast and aggregate data across
different GPUs. It schedules a cooperating kernel on each GPU that knows how to
best utilize the underlying hardware topology; this kernel uses a single SM of
the GPU.

In our experiment, we demonstrate that although NCCL often leads to much faster
data aggregation by itself, it doesn't necessarily lead to faster training. Our
hypothesis is that the implicit copies are essentially free since they go to the
copy engine on GPU, as long as its latency can be hidden by the main computation
itself. Although NCCL can transfer data faster, it takes one SM away, and adds
more pressure to the underlying L2 cache. Our results show that for 8-GPUs, NCCL
often leads to better performance. However, for fewer GPUs, the implicit copies
often perform better.

#### Staged Variables

We further introduce a staged-variable mode where we use staging areas for both
the variable reads, and their updates. Similar to software pipelining of the
input pipeline, this can hide the data copy latency. If the computation time
takes longer than the copy and aggregation, the copy itself becomes essentially
free.

The downside is that all the weights read are from the previous training step.
So it is a different algorithm from SGD. But it is possible to improve its
convergence by adjusting learning rate and other hyperparameters.

## Executing the script

This section lists the core command line arguments and a few basic examples for
executing the main script
([tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)).

> Note: `tf_cnn_benchmarks.py` uses the config `force_gpu_compatible`,
> which was introduced after TensorFlow 1.1. Until TensorFlow 1.2 is released
> building from source is advised.

#### Base command line arguments

*   **`model`**: Model to use, e.g. `resnet50`, `inception3`, `vgg16`, and
    `alexnet`.
*   **`num_gpus`**: Number of GPUs to use.
*   **`data_dir`**: Path to data to process. If not set, synthetic data is used.
    To use ImageNet data use these
    [instructions](https://github.com/tensorflow/models/tree/master/research/inception#getting-started)
    as a starting point.
*   **`batch_size`**: Batch size for each GPU.
*   **`variable_update`**: The method for managing variables: `parameter_server`
    ,`replicated`, `distributed_replicated`, `independent`
*   **`local_parameter_device`**: Device to use as parameter server: `cpu` or
    `gpu`.

#### Single instance examples

```bash
# VGG16 training ImageNet with 8 GPUs using arguments that optimize for
# Google Compute Engine.
python tf_cnn_benchmarks.py --local_parameter_device=cpu --num_gpus=8 \
--batch_size=32 --model=vgg16 --data_dir=/home/ubuntu/imagenet/train \
--variable_update=parameter_server --nodistortions

# VGG16 training synthetic ImageNet data with 8 GPUs using arguments that
# optimize for the NVIDIA DGX-1.
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \
--batch_size=64 --model=vgg16 --variable_update=replicated --use_nccl=True

# VGG16 training ImageNet data with 8 GPUs using arguments that optimize for
# Amazon EC2.
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \
--batch_size=64 --model=vgg16 --variable_update=parameter_server

# ResNet-50 training ImageNet data with 8 GPUs using arguments that optimize for
# Amazon EC2.
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \
--batch_size=64 --model=resnet50 --variable_update=replicated --use_nccl=False

```

#### Distributed command line arguments

*   **`ps_hosts`**: Comma separated list of hosts to use as parameter servers
    in the format of ```<host>:port```, e.g. ```10.0.0.2:50000```.
*   **`worker_hosts`**: Comma separated list of hosts to use as workers in the
    format of ```<host>:port```, e.g. ```10.0.0.2:50001```.
*   **`task_index`**: Index of the host in the list of `ps_hosts` or
    `worker_hosts` being started.
*   **`job_name`**: Type of job, e.g `ps` or `worker`

#### Distributed examples

Below is an example of training ResNet-50 on 2 hosts: host_0 (10.0.0.1) and
host_1 (10.0.0.2). The example uses synthetic data. To use real data pass the
`--data_dir` argument.

```bash
# Run the following commands on host_0 (10.0.0.1):
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \
--batch_size=64 --model=resnet50 --variable_update=distributed_replicated \
--job_name=worker --ps_hosts=10.0.0.1:50000,10.0.0.2:50000 \
--worker_hosts=10.0.0.1:50001,10.0.0.2:50001 --task_index=0

python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \
--batch_size=64 --model=resnet50 --variable_update=distributed_replicated \
--job_name=ps --ps_hosts=10.0.0.1:50000,10.0.0.2:50000 \
--worker_hosts=10.0.0.1:50001,10.0.0.2:50001 --task_index=0


# Run the following commands on host_1 (10.0.0.2):
python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \
--batch_size=64 --model=resnet50 --variable_update=distributed_replicated \
--job_name=worker --ps_hosts=10.0.0.1:50000,10.0.0.2:50000 \
--worker_hosts=10.0.0.1:50001,10.0.0.2:50001 --task_index=1

python tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 \
--batch_size=64 --model=resnet50 --variable_update=distributed_replicated \
--job_name=ps --ps_hosts=10.0.0.1:50000,10.0.0.2:50000 \
--worker_hosts=10.0.0.1:50001,10.0.0.2:50001 --task_index=1

```
