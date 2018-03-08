# 编写数据读写器

预备知识：

*   熟悉 C++ 。
*   必须已经掌握 @{$install_sources$downloaded TensorFlow source}，并能够构建它。

我们将这个编写数据读写器以支持一种文件格式的任务分为两个部分

*   文件格式：我们使用 **读写器** 操作从一个文档中读取 **记录**(它可以是任意的字符串)。
*   记录格式：我们通过 TensorFlow 使用解码器或解析操作将一个字符记录转换成 tensor 可用的对象。

例如，为了读入一个 [CSV file](https://en.wikipedia.org/wiki/Comma-separated_values),我们使用 @{tf.TextLineReader$a Reader for text files} 然后再使用 @{tf.decode_csv$an Op that parses CSV data from a line of text}。

[TOC]

## 为一种文件格式编写一个读写器

`Reader` 用于从文件中读取记录。TensorFlow 中已经有一些预建好的读写器操作样例：

*   @{tf.TFRecordReader} ([source in `kernels/tf_record_reader_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/tf_record_reader_op.cc))
*   @{tf.FixedLengthRecordReader} ([source in `kernels/fixed_length_record_reader_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/fixed_length_record_reader_op.cc))
*   @{tf.TextLineReader} ([source in `kernels/text_line_reader_op.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/text_line_reader_op.cc))

这些读写器暴露的都是一样的接口，只有它们的构建函数不相同。最重要的方法是 `read`。它需要一个队列参数，用于在需要文件名时读取文件名（例如：当 `read` 操作第一次运行，或者前一次 `read` 从一个文件中读取最后一条记录）。它会产生两个纯量的 tensor：一个字符类型的键和一个字符类型的值。

要创建一个叫 `SomeReader` 新的读写器，你需要：

1.  在 C++中，定义一个 [`tensorflow::ReaderBase`](https://www.tensorflow.org/code/tensorflow/core/framework/reader_base.h) 的子类 `SomeReader`。
2.  在 C++ 中，使用 `"SomeReader"` 注册一个新的读写器操作和内核。
3.  在 Python 中，定义一个 @{tf.ReaderBase} 的子类 `SomeReader`。

可以把所有的 C++ 代码放到 `tensorflow/core/user_ops/some_reader_op.cc` 这个文档中。用于读取文件的代码会运行在 C++ `ReaderBase` 类的子类中，这个类被定义在 [`tensorflow/core/kernels/reader_base.h`](https://www.tensorflow.org/code/tensorflow/core/framework/reader_base.h)。你需要实现下面这些方法：

*   `OnWorkStartedLocked`：打开下一个文件
*   `ReadLocked`：读取一个记录或者抛出一个文件结尾（EOF）/错误
*   `OnWorkFinishedLocked`：关闭当前文件
*   `ResetLocked`：清除当前状态，例如在抛出一个错误后

由于 `ReaderBase` 需要在调用这些方法之前获取一个互斥锁，所以它们的名称都以 “Locked” 结尾，因此你通常不用担心线程的安全性（尽管这种方式只会保护此类中的成员，而不是全局状态）。

对于 `OnWorkStartedLocked` 方法，要打开的文件件名是 `current_work()` 方法返回的值。`ReadLocked` 有这样的特征码：

```c++
Status ReadLocked(string* key, string* value, bool* produced, bool* at_end)
```

如果 `ReadLocked` 成功的从一个文件中读取到了一个记录，它会被装填进：

*   `*key`：对于记录的一个标识，用于使用它时再次找出这个记录。你可以将 `current_work()` 返回的文件名包含其中，并附加上一个记录编码或者任何其他内容。
*   `*value`：记录的内容。
*   `*produced`：设定为 `true`。

如果你到达了文件结尾（EOF），设定 `*at_end` 为 `true`。在任何情况下，返回 `Status::OK()`。如果有一个错误，直接使用 [`tensorflow/core/lib/core/errors.h`](https://www.tensorflow.org/code/tensorflow/core/lib/core/errors.h) 中的一个助手函数返回它，并且不需要定义其他参数。

下一步，你要创建真实的读写器操作。如果你熟悉 @{$adding_an_op$the adding an op how-to} 会简单很多，主要步骤是：

*   注册这个操作。
*   定义并注册一个 `OpKernel`。

要注册这个操作，你需要使用一个 [`tensorflow/core/framework/op.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op.h)  中定义的一个 `REGISTER_OP` 调用。读写器操作从不需要任何输入参数并且总会输出一个 `resource` 类型对象。它应有字符型的 `container` 和 `shared_name` 属性。你也可以定义用于配置的附加属性或在 `Doc` 属性中包含文件。例如，查看 [`tensorflow/core/ops/io_ops.cc`](https://www.tensorflow.org/code/tensorflow/core/ops/io_ops.cc) 了解，再例如：

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

读写器可以使用 `ReaderOpKernel` 中的缩写来定义一个 `OpKernel`，它被定义在 [`tensorflow/core/framework/reader_op_kernel.h`](https://www.tensorflow.org/code/tensorflow/core/framework/reader_op_kernel.h)。在定义完类之后， 你需要使用 `REGISTER_KERNEL_BUILDER(...)` 注册它。
一个没有参数的样例：

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

最后一步是添加 Python 封装。你既可以使用 @{$adding_an_op#building_the_op_library$compiling a dynamic library} 完成，或者在你用源代码构建 TensorFlow 时添加进 `user_ops.py`。然后你需要在 [`tensorflow/python/user_ops/user_ops.py`](https://www.tensorflow.org/code/tensorflow/python/user_ops/user_ops.py) 中引入 `tensorflow.python.ops.io_ops` 并添加一个 [`io_ops.ReaderBase`](https://www.tensorflow.org/code/tensorflow/python/ops/io_ops.py) 的子类。

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

## 为一种记录格式编写一个操作

通常来说获得一个纯量字符记录作为输入并跟着 @{$adding_an_op$the instructions to add an Op} 是一个普通的操作。你可以选择使用纯量字符串键值作为输入，并将其包含在用于报告错误格式数据的错误信息中。这样用户就可以更简单的追踪到错误数据的源头。

对于解析记录有用的操作样例：

*   @{tf.parse_single_example} (and @{tf.parse_example})
*   @{tf.decode_csv}
*   @{tf.decode_raw}

需要注意的是，使用多种操作解析一种详细的记录格式是很有用的。例如，在 [a `tf.train.Example` protocol buffer](https://www.tensorflow.org/code/tensorflow/core/example/example.proto) 中你可能需要将一个图像保存成一个字符串。根据图像的格式，你可能需要和 @{tf.parse_single_example} 操作类似的输出并调用 @{tf.image.decode_jpeg}、@{tf.image.decode_png} 或者 @{tf.decode_raw}。通常会使用`tf.decode_raw` 的输出再调用 @{tf.slice} 和 @{tf.reshape} 提取片段。
