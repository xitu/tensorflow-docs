# 开发者指南

这个部分的文档会深入介绍 TensorFlow 的工作细节中。这些模块分别如下：

## 顶层 API

* @{$programmers_guide/estimators}，介绍了一种能够大大简化机器学习（ML）编程的 TensorFlow API。
* @{$programmers_guide/datasets}，说明了如何设置读取数据送入 TensorFlow 程序的数据管道。

## 底层 API

* @{$programmers_guide/low_level_intro}，介绍怎样在顶层 API 之外使用 TensorFlow 的基础内容。
* @{$programmers_guide/tensors}，介绍怎样创建，操作和使用 Tensor（TensorFlow 的基本对象）。
* @{$programmers_guide/variables}，详细说明了如何在程序中表示共享的持久状态。
* @{$programmers_guide/graphs}，解释了：
  * 数据流图，将计算表示为操作间的依赖关系的 TensorFlow 中的表现方式。
  * 会话，用于跨一个或多个本地或远程设备运行数据流图的 TensorFlow 中的机制。如果你是使用 TensorFlow 底层 API 编程，这一单元是必不可少的。如果你是使用 TensorFlow 顶层 API（比如 Estimators or Keras）编程，顶层 API 会创建和管理流图和会话，但是理解流图和会话仍然有益于你。
* @{$programmers_guide/saved_model}，解释了如何保存和恢复变量和模型。
* @{$using_gpu} 解释了 TensorFlow 如何将操作分配分配给各个设备和如何手动改变编排。

## ML 概念

* @{$programmers_guide/embedding}，介绍了嵌入（embedding）的概念，提供了一个 TensorFlow 的训练嵌入的样例，并说明了如何使用 TensorBoard Embedding Projector 查看嵌入。

## 调试

* @{$programmers_guide/debugger}，说明了如何使用 TensorFlow 调试器（tfdbg）。

## TensorBoard

TensorBoard 是一个用于可视化展现机器学习中不同方面的工具。下面的指南说明了如何使用 TensorBoard：

* @{$programmers_guide/summaries_and_tensorboard}，介绍 TensorBoard。
* @{$programmers_guide/graph_viz}，说明如何可视化计算流图。
* @{$programmers_guide/tensorboard_histograms}，演示如何使用 TensorBoard 直方图仪表盘。

## 其他

* @{$programmers_guide/version_compat}，向后兼容性说明。
* @{$programmers_guide/faq}，TensorFlow 常见问题列表
