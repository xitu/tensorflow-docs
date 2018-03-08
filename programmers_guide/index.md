# 开发者指南

这个单元的文档会深入到编写 TensorFlow 应用的细节中。针对 TensorFlow 1.3，我们认真的修订了这份文档，这份文档现在主要包括以下内容：

  * @{$programmers_guide/estimators$Estimators}，介绍了一个能够简化机器学习程序的 TensorFlow 高级 API。
  * @{$programmers_guide/tensors$Tensors}，解释怎样创建、操作和使用 Tensor（TensorFlow 的基础对象）。
  * @{$programmers_guide/variables$Variables}，详细说明了怎样展现程序中共享的、持久化的状态。
  * @{$programmers_guide/graphs$Graphs and Sessions}，解释了：
      * 数据流图，TensorFlow 将计算过程作为操作之间依赖关系的展现。
      * 会话，TensorFlow 用来在若干台设备（本地或远程）上运行数据流图表的机制。
    如果你使用底层的 TensorFlow API 编写程序，那么此部分内容对你来说将非常重要。如果你在使用 TensorFlow 高级 API （比如 Estimators 或 Keras） 编写程序，那么这些高级 API 会帮你管理流图和会话，但是理解流图和会话是非常有益的。
  * @{$programmers_guide/saved_model$Saving and Restoring}，解释了如何保存和恢复你的变量和模型。
  * @{$programmers_guide/datasets$Input Pipelines}，解释了怎么搭建数据输入总线，从而读取数据集到你的 TensorFlow 程序。
  * @{$programmers_guide/embedding$Embeddings}，介绍了嵌入的概念，提供了一个在 TensorFlow 训练一个嵌入的样例，并演示了怎样在 TensorBoard Embedding Projector 中查看嵌入。
  * @{$programmers_guide/debugger$Debugging TensorFlow Programs}，讲解了怎样使用 TensorFlow 调试器（tfdbg）。
  * @{$programmers_guide/version_compat$TensorFlow Version Compatibility}，介绍了向后兼容性的支持情况
  * @{$programmers_guide/faq$FAQ}，涵盖了一些有关 TensorFlow 的常见问题。（除了移除了一些过时内容外，我们没有针对 TensorFlow 1.3 版本对这个文档进行修订。）
