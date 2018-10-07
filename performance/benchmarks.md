# 基准

## 概述

在多个平台上对图像分类模型进行测试，为 TensorFlow 社区创建了一个参考点。在 [方法](#methodology) 章节中会详细说明如何执行测试，并给出使用的脚本链接。

## 图像分类模型的结果

InceptionV3 ([arXiv:1512.00567](https://arxiv.org/abs/1512.00567))、ResNet-50
([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))、ResNet-152
([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))、VGG16
([arXiv:1409.1556](https://arxiv.org/abs/1409.1556)) 和
[AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 使用 [ImageNet](http://www.image-net.org/) 数据集测试。这些测试运行在 Google 计算云引擎，亚马逊计算云（Amazon EC2）和 NVIDIA® DGX-1™。大部分测试都使用了合成和真实的数据。使用 `tf.Variable` 对合成数据进行测试，数据集设置为 ImageNet 中每个模型所需的数据的同一形状。我们认为，对平台进行基准测试时，包含真实数据是很重要的。在底层硬件和框架上对准备数据加载测试是为了进行实际训练。为了将磁盘 I/O 作为变量移除，我们从合成数据开始，并设置一个基线。然后使用真实的数据来验证 TensorFlow 的输入管道和底层磁盘 I/O 是否使计算单元饱和。

### 使用 NVIDIA® DGX-1™ (NVIDIA® Tesla® P100) 训练

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="https://www.tensorflow.org/images/perf_summary_p100_single_server.png">
</div>

细节和其他结果参见 [Details for NVIDIA® DGX-1™ (NVIDIA®
Tesla® P100)](#details_for_nvidia_dgx-1tm_nvidia_tesla_p100)。

### 使用 NVIDIA® Tesla® K80 训练

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="https://www.tensorflow.org/images/perf_summary_k80_single_server.png">
</div>

细节和其他结果参见 [Details for Google Compute Engine
(NVIDIA® Tesla® K80)](#details_for_google_compute_engine_nvidia_tesla_k80) 和
[Details for Amazon EC2 (NVIDIA® Tesla®
K80)](#details_for_amazon_ec2_nvidia_tesla_k80)。

### 使用 NVIDIA® Tesla® K80 分布式训练

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="https://www.tensorflow.org/images/perf_summary_k80_aws_distributed.png">
</div>

细节和其他结果参见 [Details for Amazon EC2 Distributed
(NVIDIA® Tesla® K80)](#details_for_amazon_ec2_distributed_nvidia_tesla_k80)。

### 合成和真实数据训练比较

**NVIDIA® Tesla® P100**

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/perf_summary_p100_data_compare_inceptionv3.png">
  <img style="width:35%" src="https://www.tensorflow.org/images/perf_summary_p100_data_compare_resnet50.png">
</div>

**NVIDIA® Tesla® K80**

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/perf_summary_k80_data_compare_inceptionv3.png">
  <img style="width:35%" src="https://www.tensorflow.org/images/perf_summary_k80_data_compare_resnet50.png">
</div>

## NVIDIA® DGX-1™ (NVIDIA® Tesla® P100) 的详细资料

### 环境配置

*   **Instance type**: NVIDIA® DGX-1™
*   **GPU:** 8x NVIDIA® Tesla® P100
*   **OS:** Ubuntu 16.04 LTS with tests run via Docker
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Benchmark GitHub hash:** 9165a70
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** Local SSD
*   **DataSet:** ImageNet
*   **Test Date:** May 2017

下表列出了每个模型的批处理大小和优化器。除了表中列出的批处理大小之外，InceptionV3、ResNet-50、ResNet-152 和 VGG16 测试的批次大小为 32。这些结果在 *其他结果* 章节。

Options            | InceptionV3 | ResNet-50 | ResNet-152 | AlexNet | VGG16
------------------ | ----------- | --------- | ---------- | ------- | -----
Batch size per GPU | 64          | 64        | 64         | 512     | 64
Optimizer          | sgd         | sgd       | sgd        | sgd     | sgd

用于每个模型的配置。

Model       | variable_update        | local_parameter_device
----------- | ---------------------- | ----------------------
InceptionV3 | parameter_server       | cpu
ResNet50    | parameter_server       | cpu
ResNet152   | parameter_server       | cpu
AlexNet     | replicated (with NCCL) | n/a
VGG16       | replicated (with NCCL) | n/a

### 结果

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="https://www.tensorflow.org/images/perf_summary_p100_single_server.png">
</div>

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/perf_dgx1_synth_p100_single_server_scaling.png">
  <img style="width:35%" src="https://www.tensorflow.org/images/perf_dgx1_real_p100_single_server_scaling.png">
</div>

**训练合成数据**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | AlexNet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 142         | 219       | 91.8       | 2987    | 154
2    | 284         | 422       | 181        | 5658    | 295
4    | 569         | 852       | 356        | 10509   | 584
8    | 1131        | 1734      | 716        | 17822   | 1081

**训练真实数据**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | AlexNet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 142         | 218       | 91.4       | 2890    | 154
2    | 278         | 425       | 179        | 4448    | 284
4    | 551         | 853       | 359        | 7105    | 534
8    | 1079        | 1630      | 708        | N/A     | 898

从上图表可以看出，由于最大输入的限制，AlexNet 模型没有使用 8 个 GPU 来训练数据。

### 其他结果

以下是批处理大小为 32 的结果。

**训练合成数据**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | VGG16
---- | ----------- | --------- | ---------- | -----
1    | 128         | 195       | 82.7       | 144
2    | 259         | 368       | 160        | 281
4    | 520         | 768       | 317        | 549
8    | 995         | 1485      | 632        | 820

**训练真实数据**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | VGG16
---- | ----------- | --------- | ---------- | -----
1    | 130         | 193       | 82.4       | 144
2    | 257         | 369       | 159        | 253
4    | 507         | 760       | 317        | 457
8    | 966         | 1410      | 609        | 690

## Google Compute Engine (NVIDIA® Tesla® K80) 的详细资料

### 环境配置

*   **Instance type**: n1-standard-32-k80x8
*   **GPU:** 8x NVIDIA® Tesla® K80
*   **OS:** Ubuntu 16.04 LTS
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Benchmark GitHub hash:** 9165a70
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** 1.7 TB Shared SSD persistent disk (800 MB/s)
*   **DataSet:** ImageNet
*   **Test Date:** May 2017

下表列出了每个模型的批处理大小和优化器。除了表中列出的批处理大小之外，InceptionV3 和 ResNet-50 测试的批次大小为 32。这些结果在 *其他结果* 章节。

Options            | InceptionV3 | ResNet-50 | ResNet-152 | AlexNet | VGG16
------------------ | ----------- | --------- | ---------- | ------- | -----
Batch size per GPU | 64          | 64        | 32         | 512     | 32
Optimizer          | sgd         | sgd       | sgd        | sgd     | sgd

每个模型所用的配置中， variable_update 和 parameter_server 配置相同，local_parameter_device 和 cpu 配置相同。

### 结果

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/perf_gce_synth_k80_single_server_scaling.png">
  <img style="width:35%" src="https://www.tensorflow.org/images/perf_gce_real_k80_single_server_scaling.png">
</div>

**训练合成数据**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | AlexNet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 30.5        | 51.9      | 20.0       | 656     | 35.4
2    | 57.8        | 99.0      | 38.2       | 1209    | 64.8
4    | 116         | 195       | 75.8       | 2328    | 120
8    | 227         | 387       | 148        | 4640    | 234

**训练真实数据**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | AlexNet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 30.6        | 51.2      | 20.0       | 639     | 34.2
2    | 58.4        | 98.8      | 38.3       | 1136    | 62.9
4    | 115         | 194       | 75.4       | 2067    | 118
8    | 225         | 381       | 148        | 4056    | 230

### 其他结果

**训练合成数据**

GPUs | InceptionV3 (batch size 32) | ResNet-50 (batch size 32)
---- | --------------------------- | -------------------------
1    | 29.3                        | 49.5
2    | 55.0                        | 95.4
4    | 109                         | 183
8    | 216                         | 362

**训练真实数据**

GPUs | InceptionV3 (batch size 32) | ResNet-50 (batch size 32)
---- | --------------------------- | -------------------------
1    | 29.5                        | 49.3
2    | 55.4                        | 95.3
4    | 110                         | 186
8    | 216                         | 359

## Amazon EC2 (NVIDIA® Tesla® K80) 的详细资料

### 环境配置

*   **Instance type**: p2.8xlarge
*   **GPU:** 8x NVIDIA® Tesla® K80
*   **OS:** Ubuntu 16.04 LTS
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Benchmark GitHub hash:** 9165a70
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** 1TB Amazon EFS (burst 100 MiB/sec for 12 hours, continuous 50
    MiB/sec)
*   **DataSet:** ImageNet
*   **Test Date:** May 2017

下表列出了每个模型的批处理大小和优化器。除了表中列出的批处理大小之外，InceptionV3 和 ResNet-50 测试的批次大小为 32。这些结果在 *其他结果* 章节。

Options            | InceptionV3 | ResNet-50 | ResNet-152 | AlexNet | VGG16
------------------ | ----------- | --------- | ---------- | ------- | -----
Batch size per GPU | 64          | 64        | 32         | 512     | 32
Optimizer          | sgd         | sgd       | sgd        | sgd     | sgd

用于每个模型的配置。

Model       | variable_update           | local_parameter_device
----------- | ------------------------- | ----------------------
InceptionV3 | parameter_server          | cpu
ResNet-50   | replicated (without NCCL) | gpu
ResNet-152  | replicated (without NCCL) | gpu
AlexNet     | parameter_server          | gpu
VGG16       | parameter_server          | gpu

### 结果

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/perf_aws_synth_k80_single_server_scaling.png">
  <img style="width:35%" src="https://www.tensorflow.org/images/perf_aws_real_k80_single_server_scaling.png">
</div>

**训练合成数据**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | AlexNet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 30.8        | 51.5      | 19.7       | 684     | 36.3
2    | 58.7        | 98.0      | 37.6       | 1244    | 69.4
4    | 117         | 195       | 74.9       | 2479    | 141
8    | 230         | 384       | 149        | 4853    | 260

**训练真实数据**

GPUs | InceptionV3 | ResNet-50 | ResNet-152 | AlexNet | VGG16
---- | ----------- | --------- | ---------- | ------- | -----
1    | 30.5        | 51.3      | 19.7       | 674     | 36.3
2    | 59.0        | 94.9      | 38.2       | 1227    | 67.5
4    | 118         | 188       | 75.2       | 2201    | 136
8    | 228         | 373       | 149        | N/A     | 242

由于我们的 EFS 没有提供足够的吞吐量，在上面的图表中我们排除了使用 8 个 GPU 来训练 AlexNet 模型的统计。

### 其他结果

**训练合成数据**

GPUs | InceptionV3 (batch size 32) | ResNet-50 (batch size 32)
---- | --------------------------- | -------------------------
1    | 29.9                        | 49.0
2    | 57.5                        | 94.1
4    | 114                         | 184
8    | 216                         | 355

**训练真实数据**

GPUs | InceptionV3 (batch size 32) | ResNet-50 (batch size 32)
---- | --------------------------- | -------------------------
1    | 30.0                        | 49.1
2    | 57.5                        | 95.1
4    | 113                         | 185
8    | 212                         | 353

## Amazon EC2 Distributed (NVIDIA® Tesla® K80) 的详细资料

### 环境配置

*   **Instance type**: p2.8xlarge
*   **GPU:** 8x NVIDIA® Tesla® K80
*   **OS:** Ubuntu 16.04 LTS
*   **CUDA / cuDNN:** 8.0 / 5.1
*   **TensorFlow GitHub hash:** b1e174e
*   **Benchmark GitHub hash:** 9165a70
*   **Build Command:** `bazel build -c opt --copt=-march="haswell" --config=cuda
    //tensorflow/tools/pip_package:build_pip_package`
*   **Disk:** 1.0 TB EFS (burst 100 MB/sec for 12 hours, continuous 50 MB/sec)
*   **DataSet:** ImageNet
*   **Test Date:** May 2017

下表列出了每个模型的批处理大小和优化器。除了表中列出的批处理大小之外，InceptionV3 和 ResNet-50 测试的批次大小为 32。这些结果在 *其他结果* 章节。

Options            | InceptionV3 | ResNet-50 | ResNet-152
------------------ | ----------- | --------- | ----------
Batch size per GPU | 64          | 64        | 32
Optimizer          | sgd         | sgd       | sgd

用于每个模型的配置。

Model       | variable_update        | local_parameter_device | cross_replica_sync
----------- | ---------------------- | ---------------------- | ------------------
InceptionV3 | distributed_replicated | n/a                    | True
ResNet-50   | distributed_replicated | n/a                    | True
ResNet-152  | distributed_replicated | n/a                    | True

为了简化服务器安装，EC2 实例（p2.8xlarge）运行了 worker 服务器和 parameter 服务器。使用的 worker 服务器和 parameter 服务器数量相等，但有如下例外：

*   InceptionV3: 8 instances / 6 parameter servers
*   ResNet-50: (batch size 32) 8 instances / 4 parameter servers
*   ResNet-152: 8 instances / 4 parameter servers

### 结果

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:80%" src="https://www.tensorflow.org/images/perf_summary_k80_aws_distributed.png">
</div>

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:70%" src="https://www.tensorflow.org/images/perf_aws_synth_k80_distributed_scaling.png">
</div>

**训练合成数据**

GPUs | InceptionV3 | ResNet-50 | ResNet-152
---- | ----------- | --------- | ----------
1    | 29.7        | 52.4      | 19.4
8    | 229         | 378       | 146
16   | 459         | 751       | 291
32   | 902         | 1388      | 565
64   | 1783        | 2744      | 981

### 其他结果

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:50%" src="https://www.tensorflow.org/images/perf_aws_synth_k80_multi_server_batch32.png">
</div>

**训练合成数据**

GPUs | InceptionV3 (batch size 32) | ResNet-50 (batch size 32)
---- | --------------------------- | -------------------------
1    | 29.2                        | 48.4
8    | 219                         | 333
16   | 427                         | 667
32   | 820                         | 1180
64   | 1608                        | 2315

## 方法

该 [脚本](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) 在各种平台上运行，以生成上述结果。

为了创建尽可能重复的结果，每个测试运行 5 次，然后将时间取平均值。在给定的平台上，GPU 是在默认状态下运行的。对于 NVIDIA® Tesla® K80 来说这意味着不使用 [GPU
Boost](https://devblogs.nvidia.com/parallelforall/increase-performance-gpu-boost-k80-autoboost/)。对于每个测试，需要完成 10 次预热，然后再平均完成 100 次测试。
