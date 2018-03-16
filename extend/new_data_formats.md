# 自定义数据读取

预备知识：

*   熟悉 C++ 。
*   必须已经下载安装源码 @{$install_sources$downloaded TensorFlow source}，并能够构建它。

我们将支持一种文件格式的任务划分为两个部分：

*   文件格式：我们使用 **读取** 操作从一个文档中读取 **记录**(可以是任意字符串)。
*   记录格式：我们使用解码器或解析操作将字符串记录转换成 TensorFlow 可用的 tensor 。

例如，要读取 [CSV 文件](https://en.wikipedia.org/wiki/Comma-separated_values),我们使用 @{tf.TextLineReader$a 读取文本文件}，然后再使用 @{tf.decode_csv$a 操作逐行解析 CSV 数据}。

[TOC]

## 为一种文件格式编写读取

`Reader` 用于从文件中读取记录。TensorFlow 中已经有一些预建好的读取操作样例：

*   @{tf.TFRecordReader} ([source in `kernels/tf_record_reader_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/tf_record_reader_op.cc))
*   @{tf.FixedLengthRecordReader} ([source in `kernels/fixed_length_record_reader_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/fixed_length_record_reader_op.cc))
*   @{tf.TextLineReader} ([source in `kernels/text_line_reader_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/text_line_reader_op.cc))

这些操作暴露的都是一样的接口，唯一的区别在于构造函数不同。最重要的方法是 `read`。它需要一个队列参数，用于在需要文件名时读取文件名（例如：当 `read` 操作第一次运行，或者从文件读取最后一条记录时）。它会产生两个标量 tensor：一个字符串键和一个字符串值。

要创建新的名为 `SomeReader` 的读取操作，你需要：

1.  在 C++ 中，定义[`tensorflow::ReaderBase`](https://www.tensorflow.org/code/tensorflow/core/framework/reader_base.h) 的子类 `SomeReader`。
2.  在 C++ 中，使用 `"SomeReader"` 注册一个新的读取操作和内核。
3.  在 Python 中，定义一个 @{tf.ReaderBase} 的子类 `SomeReader`。

你可以把所有的 C++ 代码放到 `tensorflow/core/user_ops/some_reader_op.cc` 这个文档中。用于读取文件的代码会运行在 C++ `ReaderBase` 类的子类中，这个类被定义在 [`tensorflow/core/kernels/reader_base.h`](https://www.tensorflow.org/code/tensorflow/core/framework/reader_base.h)。你需要实现以下方法：

*   `OnWorkStartedLocked`：打开下一个文件
*   `ReadLocked`：读取记录或者抛出 EOF /错误
*   `OnWorkFinishedLocked`：关闭当前文件，并
*   `ResetLocked`：清除状态，例如在抛出一个错误后

这些以“Locked”结尾的方法在调用前，由于 `ReaderBase` 保证获取到互斥锁，因此你通常不必担心线程安全（尽管这种方式只会保护此类中的成员，而不是全局状态）。

对于 `OnWorkStartedLocked` 方法，要打开的文件名是 `current_work()` 方法的返回值。`ReadLocked` 声明如下：

```c++
Status ReadLocked(string* key, string* value, bool* produced, bool* at_end)
```

如果 `ReadLocked` 成功地从文件中读取到了一个记录，则返回：

*   `*key`：记录标识符，可以使用它再次查找此记录。你可以将 `current_work()` 返回的文件名包含其中，并附上一个记录编码或者其他任何内容。
*   `*value`：记录的内容。
*   `*produced`：设定为 `true`。

如果你到达了文件结尾（EOF），设置 `*at_end` 为 `true`。在任何情况下，返回 `Status::OK()`。如果出现错误，只需使用 [`tensorflow/core/lib/core/errors.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h) 中的一个助手函数返回它，而不用修改其他参数。

下一步，你将创建实际的读取操作。如果你熟悉 @{$adding_an_op$the adding an op how-to} ，将会有所帮助。主要步骤有：

*   注册操作。
*   定义并注册一个 `OpKernel`。

要注册这个操作，你需要使用在 [`tensorflow/core/framework/op.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op.h) 中定义的 `REGISTER_OP` 调用。读取操作不接受任何输入，并且总会有一个 `resource` 类型的输出。它应有字符串类型的 `container` 和 `shared_name` 属性。你可以选择为配置或在 `Doc` 中包含文档定义其他属性。有关示例，请参见 [`tensorflow/core/ops/io_ops.cc`](https://www.tensorflow.org/code/tensorflow/core/ops/io_ops.cc) ，例如：

```c++
#include "tensorflow/core/framework/op.h"

REGISTER_OP("TextLineReader")
    .Output("reader_handle: resource")
    .Attr("skip_header_lines: int = 0")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc(R"doc(
A Reader that outputs the lines of a file delimited by '\n'.
)doc");
```

读取操作可以使用 `ReaderOpKernel`（[`tensorflow/core/framework/reader_op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/reader_op_kernel.h)）中的快捷方式来定义一个 `OpKernel`。定义完类后，你需要使用 `REGISTER_KERNEL_BUILDER(...)` 注册它。一个没有参数的样例：

```c++
#include "tensorflow/core/framework/reader_op_kernel.h"

class TFRecordReaderOp : public ReaderOpKernel {
 public:
  explicit TFRecordReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    Env* env = context->env();
    SetReaderFactory([this, env]() { return new TFRecordReader(name(), env); });
  }
};

REGISTER_KERNEL_BUILDER(Name("TFRecordReader").Device(DEVICE_CPU),
                        TFRecordReaderOp);
```

一个有参数的样例：

```c++
#include "tensorflow/core/framework/reader_op_kernel.h"

class TextLineReaderOp : public ReaderOpKernel {
 public:
  explicit TextLineReaderOp(OpKernelConstruction* context)
      : ReaderOpKernel(context) {
    int skip_header_lines = -1;
    OP_REQUIRES_OK(context,
                   context->GetAttr("skip_header_lines", &skip_header_lines));
    OP_REQUIRES(context, skip_header_lines >= 0,
                errors::InvalidArgument("skip_header_lines must be >= 0 not ",
                                        skip_header_lines));
    Env* env = context->env();
    SetReaderFactory([this, skip_header_lines, env]() {
      return new TextLineReader(name(), skip_header_lines, env);
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("TextLineReader").Device(DEVICE_CPU),
                        TextLineReaderOp);
```

最后一步是添加 Python 封装。你可以通过 @{$adding_an_op#building_the_op_library$compiling a dynamic library} 完成，或者，如果你正在从源代码构建 TensorFlow，则添加到 `user_ops.py`。然后你需要在 [`tensorflow/python/user_ops/user_ops.py`](https://www.tensorflow.org/code/tensorflow/python/user_ops/user_ops.py) 中引入 `tensorflow.python.ops.io_ops` 并添加一个 [`io_ops.ReaderBase`](https://www.tensorflow.org/code/tensorflow/python/ops/io_ops.py) 的子类。

```python
from tensorflow.python.framework import ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import io_ops

class SomeReader(io_ops.ReaderBase):

    def __init__(self, name=None):
        rr = gen_user_ops.some_reader(name=name)
        super(SomeReader, self).__init__(rr)


ops.NotDifferentiable("SomeReader")
```

你可以在 [`tensorflow/python/ops/io_ops.py`](https://www.tensorflow.org/code/tensorflow/python/ops/io_ops.py) 中查看一些样例。

## 为记录格式编写操作

通常来说，这是一个普通的运算，它采取标量字符串记录作为输入，并遵循 @{$adding_an_op$the instructions to add an Op} 。你可以选择将标量字符串键值作为输入，并将其包含在用于报告错误格式数据的错误信息中。这样用户就可以更简单的追踪到错误数据的源头。

用于解码记录的操作示例：

*   @{tf.parse_single_example} (和 @{tf.parse_example})
*   @{tf.decode_csv}
*   @{tf.decode_raw}

需要注意的是，使用多种操作来解码特定记录格式可能很有用。例如，在 [a `tf.train.Example` protocol buffer](https://www.tensorflow.org/code/tensorflow/core/example/example.proto) 中你可能需要将一个图像保存成一个字符串。根据图像的格式，你可能需要和 @{tf.parse_single_example} 操作类似的输出并调用 @{tf.image.decode_jpeg}、@{tf.image.decode_png} 或者 @{tf.decode_raw}。通常会使用`tf.decode_raw` 的输出再调用 @{tf.slice} 和 @{tf.reshape} 提取片段。
