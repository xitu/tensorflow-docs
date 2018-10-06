# 扩展

本节将介绍开发者如何向 TensorFlow 添加功能。在开始之前，请先阅读以下架构概述：

  * [TensorFlow 架构](../extend/architecture.md)

以下指南介绍了如何扩展 TensorFlow 的特定功能：

  * [添加一个新操作（Op）](../extend/adding_an_op.md)，介绍了如何创建您自己的操作。
  * [添加一个定制的文件系统插件](../extend/add_filesys.md)，介绍了如何添加对您自己的共享或分布式文件系统的支持。
  * [自定义数据读取器](../extend/new_data_formats.md)，详细说明了如何添加对您自己的文件与记录格式的支持。

Python 是现在唯一一个 TensorFlow 承诺 API 稳定的语言。不过 TensorFlow 也为 C++、Java、Go 和 [JavaScript](https://js.tensorflow.org)（包括
[Node.js](https://github.com/tensorflow/tfjs-node)），提供了功能支持；此外，社区为 [Haskell](https://github.com/tensorflow/haskell) 和 [Rust](https://github.com/tensorflow/rust) 也提供了支持。如果你想为其它的语言创建或开发 TensorFlow 的功能，请阅读以下指南：

  * [其他语言的 TensorFlow](../extend/language_bindings.md)

如需创建与 TensorFlow 模型格式兼容的工具，请阅读以下指南：

  * [TensorFlow 模型文件的工具开发者指南](../extend/tool_developers/index.md)
