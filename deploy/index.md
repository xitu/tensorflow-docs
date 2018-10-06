# 部署

本章主要介绍如何在真实世界中部署模型，包含以下文档：

  * [分布式 TensorFlow](../deploy/distributed.md)，这一节解释了如何创建一个 TensorFlow 服务器集群。
  * [如何在 Hadoop 上运行 Tensorflow](../deploy/hadoop.md)，内容如题所示。
  * [如何在 S3 文件系统上运行 TensorFlow](../deploy/s3.md)，这一节说明了如何在 S3 文件系统上运行 TensorFlow。
  * 整个文档都以 [TensorFlow Serving](/serving)为基础。TensorFlow Serving 是一个开源、灵活的高性能机器学习模型服务系统，专门为生产环境所设计。TensorFlow Serving 还提供了开箱即用的整合 TensorFlow 模型的方法。[TensorFlow Serving 的源码](https://github.com/tensorflow/serving)部署于 Github。

[TensorFlow Extended（TFX）](/tfx)是一个 TensorFlow 端到端的机器学习平台。是由 Google 实施的，我们已经开放了一些 TFX 库，系统的其余部分也将会推出。
