# 扩展

本节将介绍开发者如何向 TensorFlow 添加功能。在开始之前，请先阅读以下架构概述：

  * @{$architecture$TensorFlow Architecture}

以下指南介绍了如何扩展 TensorFlow 的特定功能：

  * @{$adding_an_op$Adding a New Op}，介绍了如何创建您自己的操作。
  * @{$add_filesys$Adding a Custom Filesystem Plugin}，介绍了如何添加对您自己的共享或分布式文件系统的支持。
  * @{$new_data_formats$Custom Data Readers}，详细说明了如何添加对您自己的文件与记录格式的支持。

Python 是现在唯一一个 TensorFlow 承诺 API 稳定的语言。不过 TensorFlow 也为 C++、Java 和 Go 提供了功能支持；此外，社区为 [Haskell](https://github.com/tensorflow/haskell) 和 [Rust](https://github.com/tensorflow/rust) 提供了支持。如果你想为其它的语言创建或开发 TensorFlow 的功能，请阅读以下指南：

  * @{$language_bindings$TensorFlow in Other Languages}

如需创建与 TensorFlow 模型格式兼容的工具，请阅读以下指南：

  * @{$tool_developers$A Tool Developer's Guide to TensorFlow Model Files}
