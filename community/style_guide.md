# TensorFlow 代码风格指南

为了 TensorFlow 用户以及开发者能够提高代码可读性、减少报错并且提升一致性，我们为此提供了这份 TensorFlow 的代码风格指南。

[TOC]

## Python 代码风格指南

除了要使用两空格缩进这一点，通常就遵循 [PEP8 Python 代码风格指南](https://www.python.org/dev/peps/pep-0008/)即可。


## Python 2 以及 Python 3 的兼容

* 所有代码都必须兼容 Python 2 和 Python 3。

* 每个 Python 文件都应该包含如下几行代码：

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
```

* 使用 `six` 来编写可兼容的代码（例如 `six.moves.range`）。

## Bazel 构建规则

TensorFlow 使用 Bazel 来构建系统并执行下面的依赖：

* 每一个 BUILD 文件头部都应该包含这些代码：

```
# Description:
#   <...>

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])
```

* 在所有的 Python BUILD 目标中（库文件和测试用例）加入下面这行代码：

```
srcs_version = "PY2AND3",
```

## Tensor

* 批处理操作通常假设 Tensor 的第一个维度是批处理维度。

* 在大多数模型中**最后一个维度**是通道的数量。

* 除了第一个和最后一个之外的维度通常组成「空间」维度：代表序列长度或是图片尺寸。

* 尽量使用 Tensor 的重载运算符，而不是 TensorFlow 函数。比如说，尽量使用 `**`、`+`、`/`、`*`、`-`、`and/or` 来代替 `tf.pow`、`tf.add`、`tf.div`、`tf.mul`、`tf.subtract` 和 `tf.logical*`，除非需要特定的操作名称。

## Python 处理函数

这里的 **Python 处理函数** 是一种在输入的 tensors 和参数后，会创建一部分 graph 并且输出返回的 tensors 的函数。

* 第一个参数应该传入 tensors，后面的参数再传入一些基本的 python 参数。最后一个参数是默认值为 `None` 的 `name` 参数。如果这个处理函数需要保存一些 `Tensor` 来用于收集 Graph 的话，那么在 `name` 参数前加上要收集的参数名称即可。

* Tensor 参数应该是单个的 tensor 变量，也可以是个可迭代的 tensors 变量。例如说“Tensor 必须是单个 tensor 变量要不就是个 Tensors 数组”就太宽泛了。想了解更多可以查看 `assert_proper_iterable`。

* 如果使用 C++ 的处理函数，需要调用 `convert_to_tensor` 把 non-tensor 输入值转换为 tensors 用来当做处理函数的参数。要注意的是这个参数依然被描述为 `Tensor` 文档中具体的 dtype 对象。

* 每个 Python 处理函数都应该有个类似下面的 `name_scope`。它要作为 `name` 参数传入，这是处理函数的一个默认的变量名也是一个包含输入 tensors 的列表。

* 处理函数应该包含一个通用的 Python 函数注释，包括传入参数以及返回值的声明用于解释每个值的类型和含义。这段说明中应当规定好参数的 shapes、 dtypes、以及 ranks。

[查看文档详细信息](../community/documentation.md)

* 为了提升可用性，示例部分应该包含一个含有处理函数输入与输出的用例。

* 尽量避免直接使用 `tf.Tensor.eval` 或 `tf.Session.run`。例如，要编写依赖于 Tensor 的值的逻辑，就应该使用 [TensorFlow 流控制](https://www.tensorflow.org/api_guides/python/control_flow_ops)。或者，只在动态执行（`tf.executing_eagerly()`）打开时使用。

示例：

    def my_op(tensor_in, other_tensor_in, my_param, other_param=0.5,
              output_collections=(), name=None):
      """My operation that adds two tensors with given coefficients.

      Args:
        tensor_in: `Tensor`, input tensor.
        other_tensor_in: `Tensor`, same shape as `tensor_in`, other input tensor.
        my_param: `float`, coefficient for `tensor_in`.
        other_param: `float`, coefficient for `other_tensor_in`.
        output_collections: `tuple` of `string`s, name of the collection to
                            collect result of this op.
        name: `string`, name of the operation.

      Returns:
        `Tensor` of same shape as `tensor_in`, sum of input values with coefficients.

      Example:
        >>> my_op([1., 2.], [3., 4.], my_param=0.5, other_param=0.6,
                  output_collections=['MY_OPS'], name='add_t1t2')
        [2.3, 3.4]
      """
      with tf.name_scope(name, "my_op", [tensor_in, other_tensor_in]):
        tensor_in = tf.convert_to_tensor(tensor_in)
        other_tensor_in = tf.convert_to_tensor(other_tensor_in)
        result = my_param * tensor_in + other_param * other_tensor_in
        tf.add_to_collection(output_collections, result)
        return result

用法：

    output = my_op(t1, t2, my_param=0.5, other_param=0.6,
                   output_collections=['MY_OPS'], name='add_t1t2')

## 网络层

记得使用 `tf.keras.layers`，而非 `tf.layers`。

关于如何继承网络层的详细信息请看 `tf.keras.layers` 和 [Keras 指南](../guide/keras.md#custom_layers)。
