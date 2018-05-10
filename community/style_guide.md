# TensorFlow 代码风格指南

为了 TensorFlow 用户以及开发者能够提高代码可读性、减少报错并且提升一致性，我们为此提供了这份 TensorFlow 的代码风格指南。

[TOC]

## Python 代码风格指南

除了要使用两空格缩进这一点，通常就遵循
[PEP8 Python 代码风格指南](https://www.python.org/dev/peps/pep-0008/) 即可。


## Python 2 以及 Python 3 的兼容

* 所有代码都必须兼容 Python 2 和 Python 3 。

* 每个 Python 文件都应该包含如下几行代码：

```
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
```

* 使用 `six` 来编写可兼容的代码 (例如 `six.moves.range`).


## Bazel 构建规则

TensorFlow 使用 Bazel 来构建系统并执行下面的依赖:

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

* 每一个 BUILD 文件尾部都应该包含这些代码：

```
filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
```

* 在创建新的 BUILD 文件时，把下面这一行加入到 `all_opensource_files` 目标内的 `tensorflow/BUILD` 文件中。

```
"//tensorflow/<directory>:all_files",
```

* 在所有的 python BUILD 目标中 （库文件和测试用例） 加入下面这行代码：

```
srcs_version = "PY2AND3",
```


## Tensor

* 在假设 Tensor 的第一维度是 batche 维度的情况下对 batches 的操作。


## Python operations

A *Python operation* is a function that, given input tensors and parameters,
creates a part of the graph and returns output tensors.

* The first arguments should be tensors, followed by basic python parameters.
 The last argument is `name` with a default value of `None`.
 If operation needs to save some `Tensor`s to Graph collections,
 put the arguments with names of the collections right before `name` argument.

* Tensor arguments should be either a single tensor or an iterable of tensors.
 E.g. a "Tensor or list of Tensors" is too broad. See `assert_proper_iterable`.

* Operations that take tensors as arguments should call `convert_to_tensor`
 to convert non-tensor inputs into tensors if they are using C++ operations.
 Note that the arguments are still described as a `Tensor` object
 of a specific dtype in the documentation.

* Each Python operation should have a `name_scope` like below. Pass as
 arguments `name`, a default name of the op, and a list of the input tensors.

* Operations should contain an extensive Python comment with Args and Returns
 declarations that explain both the type and meaning of each value. Possible
 shapes, dtypes, or ranks should be specified in the description.
 @{$documentation$See documentation details}

* For increased usability include an example of usage with inputs / outputs
 of the op in Example section.

Example:

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

Usage:

    output = my_op(t1, t2, my_param=0.5, other_param=0.6,
                   output_collections=['MY_OPS'], name='add_t1t2')


## Layers

A *Layer* is a Python operation that combines variable creation and/or one or many
other graph operations. Follow the same requirements as for regular Python
operation.

* If a layer creates one or more variables, the layer function
 should take next arguments also following order:
  - `initializers`: Optionally allow to specify initializers for the variables.
  - `regularizers`: Optionally allow to specify regularizers for the variables.
  - `trainable`: which control if their variables are trainable or not.
  - `scope`: `VariableScope` object that variable will be put under.
  - `reuse`: `bool` indicator if the variable should be reused if
             it's present in the scope.

* Layers that behave differently during training should take:
  - `is_training`: `bool` indicator to conditionally choose different
                   computation paths (e.g. using `tf.cond`) during execution.

Example:

    def conv2d(inputs,
               num_filters_out,
               kernel_size,
               stride=1,
               padding='SAME',
               activation_fn=tf.nn.relu,
               normalization_fn=add_bias,
               normalization_params=None,
               initializers=None,
               regularizers=None,
               trainable=True,
               scope=None,
               reuse=None):
      ... see implementation at tensorflow/contrib/layers/python/layers/layers.py ...

