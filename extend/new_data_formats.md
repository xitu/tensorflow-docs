# 读取自定义文件和记录格式

前提：

*   对 C++ 有一定程度的了解。
*   已经[下载 TensorFlow 源码](../install/source.md)，并能够运行。

我们将支持自定义文件格式分为两个任务：

*   文件格式：使用 `tf.data.Dataset` 阅读器来从文件中读取原始**记录**（通常以零阶字符串张量（scalar string tensors）表示，也可能有其他结构）。
*   记录格式：使用解码器或者解析操作将一个字符串记录转换成 TensorFlow 可用的张量（tensor）。

例如，要重新实现 `tf.contrib.data.make_csv_dataset` 函数，我们可以使用 `tf.data.TextLineDataset` 来提取数据，并使用 `tf.data.Dataset.map` 和 `tf.decode_csv` 来从数据集中的每一行文本中解析 CSV 数据。要读取一个 [CSV 文件](https://en.wikipedia.org/wiki/Comma-separated_values)，我们可以使用 `tf.data.TextLineDataset`，然后 `tf.data.Dataset.map` 一个从数据集中的文本逐行解析 CSV 数据的 `tf.decode_csv`。

[TOC]

## 为文件格式编写一个数据集

`tf.data.Dataset` 表示一系列**元素**，即文件中独立的记录。TensorFlow中内置了几种『阅读器』数据集：

*  `tf.data.TFRecordDataset`（[源自 `kernels/data/reader_dataset_ops.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/data/reader_dataset_ops.cc)）
*  `tf.data.FixedLengthRecordDataset`（[源自 `kernels/data/reader_dataset_ops.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/data/reader_dataset_ops.cc)）
*  `tf.data.TextLineDataset`（[源自 `kernels/data/reader_dataset_ops.cc`](https://www.tensorflow.org/code/tensorflow/core/kernels/data/reader_dataset_ops.cc)）

每个实现包含了三个相关的类：

* 一个 `tensorflow::DatasetOpKernel` 的子类 （如 `TextLineDatasetOp`），这个类的 `MakeDataset()` 方法告诉 TensorFlow 怎样根据一个操作的输入和属性生成一个数据集的对象。

* 一个 `tensorflow::GraphDatasetBase` 的子类（如 `TextLineDatasetOp::Dataset`），表示数据集的**不可变性**定义，这个类的 `MakeIteratorInternal()` 方法告诉 TensorFlow 怎样在数据集上生成迭代器对象。

* 一个 `tensorflow::DatasetIterator<Dataset>` 的子类（如 `TextLineDatasetOp::Dataset::Iterator`），表示特定数据集上的迭代器的**可变性**，这个类的 `GetNextInternal()` 方法告诉 TensorFlow 怎样获取迭代器的下一个元素。

其中最重要的方法是 `GetNextInternal()`，因为它定义了怎样从文件中实际读取记录，并用一个或多个 `Tensor` 对象来表示它们。

创建一个新的阅读器数据集叫做（比方说）`MyReaderDataset`，你需要：

1. 在 C++ 中定义 `tensorflow::DatasetOpKernel`、`tensorflow::GraphDatasetBase` 和 `tensorflow::DatasetIterator<Dataset>` 的子类来实现读取逻辑。
2. 在 C++ 中注册一个新的名叫 `"MyReaderDataset"` 的阅读器操作和内核。
3. 在 Python 中定义一个名叫 `MyReaderDataset` 的 `tf.data.Dataset` 的子类。

你可以把所有 C++ 代码放到一个文件里面，比如 `my_reader_dataset_op.cc`。并且你最好熟读[如何添加一个新操作（Op）](../extend/adding_an_op.md)。下文的框架可以给你提供一点参考：

```c++
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace myproject {
namespace {

using ::tensorflow::DT_STRING;
using ::tensorflow::PartialTensorShape;
using ::tensorflow::Status;
 class MyReaderDatasetOp : public tensorflow::DatasetOpKernel {
 public:

  MyReaderDatasetOp(tensorflow::OpKernelConstruction* ctx)
      : DatasetOpKernel(ctx) {
    // 用 `ctx->GetAttr()` 解析并验证定义数据集的属性，并把它们存在成员变量中。
  }

  void MakeDataset(tensorflow::OpKernelContext* ctx,
                   tensorflow::DatasetBase** output) override {
    // 用 `ctx->input()` 或者通用函数 `ParseScalarArgument<T>(ctx, &arg)` 解析并验证定义数据集的输入张量。

    // 创建数据集对象，并根据属性或输入张量传入（已经验证的）参数。
    *output = new Dataset(ctx);
  }

 private:
  class Dataset : public tensorflow::GraphDatasetBase {
   public:
    Dataset(tensorflow::OpKernelContext* ctx) : GraphDatasetBase(ctx) {}

    std::unique_ptr<tensorflow::IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<tensorflow::IteratorBase>(new Iterator(
          {this, tensorflow::strings::StrCat(prefix, "::MyReader")}));
    }

    // 记录结构：每个记录用一个零阶字符串张量表示。
    //
    // 数据集的元素有固定数量的组件，每个组件有不同的类型和形状；重写以下两个方法来自定义数据集的这方面的设置。
    const tensorflow::DataTypeVector& output_dtypes() const override {
      static auto* const dtypes = new tensorflow::DataTypeVector({DT_STRING});
      return *dtypes;
    }
    const std::vector<PartialTensorShape>& output_shapes() const override {
      static std::vector<PartialTensorShape>* shapes =
          new std::vector<PartialTensorShape>({{}});
      return *shapes;
    }

    string DebugString() const override { return "MyReaderDatasetOp::Dataset"; }

   protected:
    // 可选：数据集的 `GraphDef` 序列化。
    //
    // 如果你想保存这个数据集（和它上面的所有迭代器）的实例，实现以下这个方法。
    Status AsGraphDefInternal(DatasetGraphDefBuilder* b,
                              tensorflow::Node** output) const override {
      // 使用 `b->AddScalar()` 和 `b->AddVector()` 来从这个对象的成员变量构建代表输入张量的节点。
      std::vector<tensorflow::Node*> input_tensors;
      TF_RETURN_IF_ERROR(b->AddDataset(this, input_tensors, output));
      return Status::OK();
    }

   private:
    class Iterator : public tensorflow::DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
          : DatasetIterator<Dataset>(params), i_(0) {}

      // 读取逻辑的实现。
      //
      // 这个文件中的示例实现产生十次 『MyReader!』 字符串。总的来讲有以下三种情况：
      // 1. 如果成功读取一个元素，在 `*out_tensors` 中将它储存为一个或多个张量，设置 `*end_of_sequence = false` 并返回 `Status::OK()`。
      // 2. 如果到达输入的结尾，设置 `*end_of_sequence = true` 并返回 `Status::OK()`。
      // 3. 如果发生了一个错误，通过 "tensorflow/core/lib/core/errors.h" 中的帮助函数返回一个错误状态。
      Status GetNextInternal(tensorflow::IteratorContext* ctx,
                             std::vector<tensorflow::Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        // 注意：`GetNextInternal()` 可能会被并发调用，所以推荐用一个互斥量来保护迭代器的状态。
        tensorflow::mutex_lock l(mu_);
        if (i_ < 10) {
          // 创建一个零阶字符串张量并把它添加到输出中。
          tensorflow::Tensor record_tensor(ctx->allocator({}), DT_STRING, {});
          record_tensor.scalar<string>()() = "MyReader!";
          out_tensors->emplace_back(std::move(record_tensor));
          ++i_;
          *end_of_sequence = false;
        } else {
          *end_of_sequence = true;
        }
        return Status::OK();
      }

     protected:
      // 可选：迭代器的状态序列化。
      //
      // 如果你想保存和恢复这个迭代器的实例，实现以下两个方法。
      Status SaveInternal(tensorflow::IteratorStateWriter* writer) override {
        tensorflow::mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("i"), i_));
        return Status::OK();
      }
      Status RestoreInternal(tensorflow::IteratorContext* ctx,
                             tensorflow::IteratorStateReader* reader) override {
        tensorflow::mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("i"), &i_));
        return Status::OK();
      }

     private:
      tensorflow::mutex mu_;
      int64 i_ GUARDED_BY(mu_);
    };
  };
};

// 为 MyReaderDataset 注册操作定义。
//
// 数据集操作通常只有一个类型为 `variant` 的输出，代表结构化的 `Dataset` 对象。
//
// 在这里添加定义数据集的属性和输入张量。
REGISTER_OP("MyReaderDataset")
    .Output("handle: variant")
    .SetIsStateful()
    .SetShapeFn(tensorflow::shape_inference::ScalarShape);

// 为 MyReaderDataset 注册核心实现。
REGISTER_KERNEL_BUILDER(Name("MyReaderDataset").Device(DEVICE_CPU),
                        MyReaderDatasetOp);

}  // 命名空间
}  // 命名空间 myproject
```

最后一步是编译 C++ 代码并添加一个 Python 封装器。完成这一步最简单的方法是[编译一个动态库](../extend/adding_an_op.md#build_the_op_library)（比方说叫做 `"my_reader_dataset_op.so"`），然后创建一个继承 `tf.data.Dataset` 的 Python 子类来封装它。以下是一个 Python 示例程序：

```python
import tensorflow as tf

# 假设文件在当前工作目录下。
my_reader_dataset_module = tf.load_op_library("./my_reader_dataset_op.so")

class MyReaderDataset(tf.data.Dataset):

  def __init__(self):
    super(MyReaderDataset, self).__init__()
    # 将输入属性或张量作为类的成员变量创建。

  def _as_variant_tensor(self):
    # 为数据集操作构建图形节点
    #
    # 当你在这个数据集或者由它衍生出来的数据集上创建一个迭代器时，
    # 这个方法会被调用。
    return my_reader_dataset_module.my_reader_dataset()

  # 以下属性定义了元素的结构：一个零阶 `tf.string` 张量。
  # 如果你修改了元素的结构，也需要修改这些属性
  # 来与 `MyReaderDataset::Dataset` 中
  # 的 `output_dtypes()` 和 `output_shapes()` 匹配。
  @property
  def output_types(self):
    return tf.string

  @property
  def output_shapes(self):
    return tf.TensorShape([])

  @property
  def output_classes(self):
    return tf.Tensor

if __name__ == "__main__":
  # 创建一个 MyReaderDataset 并打印它的元素。
  with tf.Session() as sess:
    iterator = MyReaderDataset().make_one_shot_iterator()
    next_element = iterator.get_next()
    try:
      while True:
        print(sess.run(next_element))  # 打印十次 『MyReader!』。
    except tf.errors.OutOfRangeError:
      pass
```

你可以在 [`tensorflow/python/data/ops/dataset_ops.py`](https://www.tensorflow.org/code/tensorflow/python/data/ops/dataset_ops.py) 中找到一些 `Dataset` 封装类的例子。

## 为记录格式写一个操作

通常来说这就是一个普通的以一个零阶字符串记录作为输入的操作，所以照着[添加一个新操作（Op）的说明](../extend/adding_an_op.md)做就行了。你可以选择一个零阶字符串键作为输入，并加入到错误信息中，用来报告格式非法的数据。这样用户可以轻而易举地找出脏数据来源于哪里。

列几个对解码记录有帮助的操作：

*  `tf.parse_single_example`（和 `tf.parse_example`）
*  `tf.decode_csv`
*  `tf.decode_raw`

注意：使用多个操作来解码一个特定的记录格式很有效。举个例子，你可能有一张图片以字符串的形式存在[一个 `tf.train.Example` 的 protocol buffer](https://www.tensorflow.org/code/tensorflow/core/example/example.proto) 中。根据图像格式的不同，你可以从一个 `tf.parse_single_example` 操作选择对应的输出，然后调用 `tf.image.decode_jpeg`、`tf.image.decode_png` 或者 `tf.decode_raw`。常见的做法是获取 `tf.decode_raw` 的输出，然后用 `tf.slice` 和 `tf.reshape` 来提取切片。
