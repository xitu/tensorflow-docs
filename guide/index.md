# TensorFlow 指南

本节中的文档深入介绍了 TensorFlow 的工作原理。节点内容如下：

## 高级 APIs

  * [Keras](../guide/keras.md)，用于构建和训练深度学习模型的 TensorFlow 高级 API。
  * [Eager Execution](../guide/eager.md)，一个用于命令行式动态编写 TensorFlow 代码的 API，就像你使用 Numpy 一样。
  * [导入数据](../guide/datasets.md)，简单的输入管道，将您的数据引入 TensorFlow 程序。
  * [估算器](../guide/estimators.md)，一个提供完全打包模型并且可以用于大规模培训和生产的高级 API。

## 估算器

* [预设估算器](../guide/premade_estimators.md)，预设估算器的基础知识。
* [检查点](../guide/checkpoints.md)，保存训练进度和从中断处恢复。
* [特征列](../guide/feature_columns.md)，在不改变模型的情况下处理各种输入数据类型。
* [估算器的数据集](../guide/datasets_for_estimators.md)，使用 `tf.data` 输入数据。
* [创建自定义估算器](../guide/custom_estimators.md)，编写自定义估算器。

## 加速

  * [使用 GPUs](../guide/using_gpu.md) 解释了 TensorFlow 如何为设备分配操作以及如何手动更改排列。
  * [使用 TPUs](../guide/using_tpu.md) 解释了如何修改 `Estimator` 程序以在 TPU 上运行。

## 低级 APIs

  * [简介](../guide/low_level_intro.md)，介绍了如何在高级 API 之外使用 TensorFlow 的基础知识。
  * [张量](../guide/tensors.md)，解释了如何创建，操作和访问张量（Tensors）—— TensorFlow 中的基本对象。
  * [变量](../guide/variables.md)，详细说明了如何在程序中表示共享的持久状态。
  * [图和会话](../guide/graphs.md)，即：
      * 数据流图，它是 TensorFlow 表示的计算作为操作之间的依赖关系。
      * 会话，这是 TensorFlow 在一个或多个本地或远程设备上运行数据流图的机制。
    如果您使用低级 TensorFlow API 进行编程，则此单元至关重要。如果您使用高级 TensorFlow API（例如估算器或 Keras）进行编程，则高级 API 会为您创建和管理图和会话，但理解图和会话仍然会有所帮助。
  * [保存于恢复](../guide/saved_model.md)，解释了如何保存和恢复变量和模型。

## 机器学习概念

  * [嵌入](../guide/embedding.md)，介绍了嵌入的概念，提供了在 TensorFlow 中训练嵌入（Embeddings）的简单示例，并解释了如何使用 TensorBoard Embeddings 投影查看嵌入。

## 调试

  * [TensorFlow 调试器](../guide/debugger.md)，解释了如何使用 TensorFlow 调试器（tfdbg）。

## TensorBoard

TensorBoard 是一个在机器学习各个方面进行可视化的实用程序。下面的内容介绍了如何使用TensorBoard：

  * [TensorBoard：可视化学习](../guide/summaries_and_tensorboard.md)，介绍 TensorBoard。
  * [TensorBoard：图可视化](../guide/graph_viz.md)，解释了如何可视化计算图。
  * [TensorBoard 直方图仪表板](../guide/tensorboard_histograms.md)演示了如何使用 TensorBoard 的直方图仪表板。

## 其它

  * [TensorFlow 版本兼容性](../guide/version_compat.md)，解释了向后兼容性保证和非保证。
  * [常见问题](../guide/faq.md)，包含常见的关于 TensorFlow 的问题。
