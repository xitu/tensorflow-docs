# 在其他语言中绑定 TensorFlow

## 背景

本文旨在说明如何在其他编程语言中创建或开发具有 TensorFlow 功能的程序提供指导。它描述了 TensorFlow 的特性及使得它们在通过其它编程实现相同功能的推荐方法。

Python 是 TensorFlow 支持的第一种且支持特性最多的客户端语言。随着越来越多的功能被移植到 TensorFlow 内核（C++ 实现）并通过 [C API] 公开，因此客户端语言应该使用语言的[外部函数接口](https://en.wikipedia.org/wiki/Foreign_function_interface)（FFI）调用 [C API] 从而提供 TensorFlow 功能。

## 概述

在一个编程语言中提供 TensorFlow 的功能可以分解为下面几个广泛的类别：

-   *运行一个预定义 Graph*：给定一个 `GraphDef` 或 `MetaGraphDef` 的协议消息，能够创建一个会话，执行查询并获得张量结果。 这对于想要在预先训练的模型上运行推断的移动应用或服务器来说足够了。
-   *Graph 构造*：每个定义的 TensorFlow 操作符至少有一个函数将操作添加到图形中。理想情况下这些函数会自动生成，以便在操作符定义被修改时保持同步。
-   *梯度（即自动微分）*：给定给定一个图和一系列输入输出操作符，将操作添加到到图中，计算输出与输入的损失函数的偏微分；并允许能够对图中特定的操作符自定义梯度函数。
-   *函数*：定义一个可以在 `GraphDef` 的多个位置调用的子图，并定义一个 `GraphDef` 内的 `FunctionDefLibrary` 中的 `FunctionDef` 。
-   *控制流*：构造用户特定子图的「If」和「While」操作。理想状态下，这些控制流能与梯度共同工作（见上）。
-   *神经网络库*：创建与训练一个神经网络的一系列组件（尽可能提供分布式环境支持）。虽然在其他语言中提供这种封装会非常方便，但目前还没有计划支持 Python 以外的语言。这些库通常是对以上功能的封装。

最低限度内，一个语言的绑定必须支持运行预定义的图，当然这也意味着需要支持图的构造。 TensorFlow Python API 提供了所有这些功能。

## 当前状态

新的语言支持应该建立在 [C API] 之上。 但是，正如下表所示，并不是所有的功能都可以在 C 中使用。在 [C API] 中提供更多的功能是一个正在进行的项目。

| 特性               | Python                                   | C                                        |
| :--------------- | :--------------------------------------- | :--------------------------------------- |
| 运行一个预定义 Graph    | `tf.import_graph_def`, `tf.Session`      | `TF_GraphImportGraphDef`, `TF_NewSession` |
| Graph 构造及运算符函数生成 | 支持                                       | 支持 (C API 提供客户端语言支持)                     |
| 自动微分             | `tf.gradients`                           |                                          |
| 函数               | `tf.python.framework.function.Defun`     |                                          |
| 控制流              | `tf.cond`, `tf.while_loop`               |                                          |
| 神经网络库            | `tf.train`, `tf.nn`, `tf.contrib.layers`, `tf.contrib.slim` |                                          |

## 推荐方法

### 运行一个预定义 `Graph`

一个语言的绑定应该定义下面的这些类：

-   `Graph`： 代表 TensorFlow 计算的一个图。由操作符组成（在客户端语言中由`Operation`表示）并用于 C API 中的 `TF_Graph` 。主要用于创建新 `Operation` 对象和启动 `Session` 时的参数。并同时支持通过运算符的图的遍历（`TF_GraphNextOperation`）、按名称查找操作（`TF_GraphOperationByName`）以及从 `GraphDef` 协议消息（C API 中的`TF_GraphToGraphDef` 和 `TF_GraphImportGraphDef`）进行转换。
-   `Operation`：表示一个图中的计算节点，对应于 C API 中的 `TF_Operation`。
-   `Output`：表示图中操作的某个输出，包含一个 `DataType`（和张量最终的形状）。可作为一个输入参数传递给一个函数，并将其添加到一个图中，或传递给一个 `Session` 的 `Run()` 方法来获取输出张量。对应于 C API 中的 `TF_Output`。
-   `Session`：表示 TensorFlow 运行时的特定实例的客户端。 它的主要工作为使用 `Graph` 及一些选项，然后调用图的 `Run()` 方法。对应于 C API 中的 `TF_Session`。
-   `Tensor`：表示包含所有相同 `DataType` 的元素的 N 维（矩形）数组。获取数据输出输出 `Session` 的 `Run()` 调用。 对应于 C API 中的 `TF_Tensor`。
-   `DataType`：TensorFlow 支持的所有可能张量类型的枚举。对应于 C API 中的`TF_DataType`，在 Python API 中通常称为 `dtype`。

### Graph 的构造

TensorFlow 具有许多不同的 op，并且不会永远不变。因此我们建议为给图添加新 op 编写生成函数，而不是单独手写（找到出生成器的写法的最好方法是手写几个函数）。生成函数所需的信息包含在 `OpDef` 协议消息中。

有几种方法可以获得注册一系列 `OpDef` ：

-   在 C API 中的 `TF_GetAllOpList` 会检索所有注册的 `OpDef` 协议消息。 这可以用来为客户端语言编写生成器。这便要求客户端语言具有协议缓冲区并支持解释 `OpDef` 消息。
-   C++ 函数 `OpRegistry::Global() -> GetRegisteredOps()` 返回所有已注册 `OpDef`（在[`tensorflow/core/framework/ op.h`] 中定义）的相同列表。 这可以用来在 C++中编写生成器（对没有协议缓冲区支持的语言非常有用）。
-   该列表的 ASCII 序列化版本通过自动化过程定期检入[ `tensorflow/core/ops/ops.pbtxt`]。

`OpDef` 包含以下内容：

-   驼峰法命名的 Op 名称。对于生成的函数而言，遵循语言习惯。例如，如果语言使用 snake_case，则应使用这种习惯而不是 CamelCase 作为 op 的函数名称。
-   输入和输出的列表。如 @{$adding_an_op$Adding an op} 中输入和输出部分所描述的哪样，这些类型可能会通过引用属性而变为多态。
-   属性列表及其默认值（如果有的话）。需要注意的是某些默认参数的类型（从输入中）推导情况、可选参数（如果有默认值）以及实参（没有默认值）。
-   op 的文档以及输入、输出和非推断属性。
-   运行时使用的一些其他字段，可由代码生成器忽略。

`OpDef` 可以转换成一个函数的文本，使用 `TF_OperationDescription` C API（包含在语言的 FFI 中）将该操作添加到图中：

-   Start with `TF_NewOperation()` to create the `TF_OperationDescription*`. 从 `TF_NewOperation()` 开始创建 `TF_OperationDescription*`。
-   每个输入调用 `TF_AddInput()` 或 `TF_AddInputList()` 一次（取决于输入是否具有列表类型）。
-   调用 `TF_SetAttr *()` 函数来设置不需要推导的属性。如果不想覆盖默认值，可以跳过默认值的属性。
-   在有必要时设置可选项：
    -   `TF_SetDevice()`：将操作强制设定到一个特殊设备上。
    -   `TF_AddControlInput()`：在此操作开始运行之前添加另一个操作完成的要求
    -   `TF_SetAttrString("_ kernel")` 用来设置内核标签（很少使用）
    -   `TF_ColocateWith()` 将一个操作与另一个操作合并
-   完成后调用 `TF_FinishOperation()`。这个调用会在图不能被修改后将操作添加到图中。

现有示例运行代码生成器作为构建过程的一部分（使用 Bazel genrule）。 或者，代码生成器可由自动化 cron 进程运行，并检查结果。这会在生成的代码和存储库中的 `OpDef` 之间产生分歧。但是，对于预计会提前生成代码的语言（例如 Go 里的 `go get` 和 Rust 里的 `cargo ops`）而言，这是非常有用的。 另一方面，对于某些语言来说，代码可以从 [`tensorflow/core/ops/ops.pbtxt`] 动态生成。

#### 处理常量

如果用户可以为输入参数提供常量，那么调用代码将更加简洁。生成的代码应该将这些常量转换并添加到图的操作中，从而将其用作实例化为 op 的输入。

#### 可选参数

如果语言支持一个函数拥有可选参数（比如 Python 中带有默认值的关键字参数），则可以将操作名、设备、输入控制等用于可选属性。在某些语言中，这些可选参数可以使用动态作用域 （类似于Python中的「with」）。 如果没有这些功能，则可以尝试使用「建造者模式」，类似于 TensorFlow API的 C++ 版本的做法。

#### 作用域命名

使用 Scope 层次结构一类的方法来支持图操作作用域命名是个不错的选择，尤其是考虑到 TensorBoard 需要它来暂时强大的图的表示。现有的 Python 和 C++ API 使用了不同的方法：在 Python 中，名称的「目录」部分（直到最后一个「/」）都来自 `with` 代码区块。事实上，这里有一个局部的线程堆栈，在这个作用域下定义了命名的层次结构。在Python中，名称的最后一个组件由用户显示提供（使用可选的 `name` 关键字参数）或者默认为需要添加的 op 类型的名称。在 C ++ 中，名称的「目录」部分存储在显式的 `Scope` 对象中。`NewSubScope()` 方法会添加到该名称的所在的位置并返回一个新的 `Scope`。该名称的最后一个组件是使用 `WithOpName()` 方法设置的，与 Python 相同，默认为添加的 op 类型的名称。`Scope` 对象显式传递从而指定上下文的名称。

#### 封装

将生成的函数对某些操作私有也是有意义的，这使得封装函数可以做一些额外的逻辑。同时也提供了一个逃生舱来支持生成代码范围之外的功能。

封装器的一个用途是支持 `SparseTensor` 输入和输出。 `SparseTensor` 是一个由 3 个密集张量构成的元组：索引、值和形状。值向量的大小为 [n]，形状向量大小为 [rank]，索引矩阵的大小为 [n, rank]。 有一些稀疏操作使用这个三元组来表示单个稀疏张量。

使用封装的另一个原因是某些 op 需要记录状态。有几个 op （例如变量）具有用于在特定状态下操作的几个伴随 op。Python API 具有用于这些操作的类，构造函数在这些类中创建操作，该类上的方法则是将操作添加到到图中并操作其状态。

#### 其他考虑

-   解决 op 函数重命名与语言关键字（或其他会引起问题的符号，如代码中引用库函数或变量的名字）冲突可以考虑准备一个关键字列表。
-   将 `Const` 操作添加到图的函数通常是一个包装器，因为生成的函数通常会有冗余的 `DataType` 输入。

### 梯度、函数及控制流

目前，除了 Python 之外，其他语言并没有提供梯度、函数及控制流操作（if 和 while）。我们会在 [C API] 提供必要支持后更新。

[C API]: https://www.tensorflow.org/code/tensorflow/c/c_api.h

[`tensorflow/core/ops/ops.pbtxt`]: https://www.tensorflow.org/code/tensorflow/core/ops/ops.pbtxt
[`tensorflow/python/BUILD`]: https://www.tensorflow.org/code/tensorflow/python/BUILD
[`tensorflow/core/framework/op.h`]: https://www.tensorflow.org/code/tensorflow/core/framework/op.h
