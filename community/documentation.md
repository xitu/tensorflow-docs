# 编写 TensorFlow 文档

我们欢迎社区对 TensorFlow 文档做贡献。这份文档说明了你可以怎样为 TensorFlow 文档做出贡献，特别地，这份文档对以下内容进行了说明：

* 文档位于什么地方。
* 怎样进行格式一致的编辑。

你可以在 https://www.tensorflow.org 上查看 TensorFlow 文档，也可以在 [`site/en`](https://github.com/tensorflow/docs/tree/master/site/en) 文件夹下对应的路径中查看和编辑原始文件。

我们正在将这些文档发布到 GitHub 上，以便每个人都可以为之做贡献。所有经过核对编入 `tensorflow/docs/site/en` 的内容之后都会尽快地发布到 [tensorflow.org](https://www.tensorflow.org)。

我们非常欢迎通过不同的形式重新发布 TensorFlow 文档，但我们不大可能允许让别的格式的文档（或者其他的文档生成工具）进入我们的代码仓库。如果你想以另外的格式重新发布我们的文档，请确保包含以下内容：

* 这种格式的文档对应的 API 版本（例如 r1.0、master 等等）
* 这份文档是从哪次提交或者哪个版本产生的
* 从哪里（即 https://www.tensorflow.org）可以找到最新版本的文档
* Apache 2.0 开源许可协议

## 版本与分支

在 [tensorflow.org](https://www.tensorflow.org) 网站的根目录下有  Tensorflow 最新稳定版的文档。如果你要用 `pip` 命令来安装 TensorFlow，你应该阅读这份文档。

默认的 TnesorFlow pip 包是由 [TensorFlow 主仓库](https://github.com/tensorflow/tensorflow/)的代码构建而成的。

此外，为了快速迭代更新，本网站的文档由 [`docs/master` 分支](https://github.com/tensorflow/docs/blob/master/site/en/)构建。

老版本的文档可以在名为 `rX.X` 的分支中查看。所有老版本分支都是在新版本发布时创建的。比如我们会在 `r1.11` 发布时创建 `r1.10` 分支。

少数情况下，有一些新的特性我们没有能够及时加入本网站中，此时文档会在一个特征分支中进行完善，并在完成时尽快并入主分支。

## API 文档

以下几种参考文档是由代码中的注释自动生成的：

- C++ API 参考文档
- Java API 参考文档
- Python API 参考文档

如果想修改参考文档，你需要编辑对应代码的注释。由于这些参考文档的内容需要与默认的安装版本一直，因此它们仅会随着最新发布的版本进行更新。

Python 的 API 文档是由 tensorflow 的主代码库生成的，使用 bazel 对 `//tensorflow/tools/docs:generate` 进行构建：

```sh
bazel run //tensorflow/tools/docs:generate -- --output_dir=/tmp/master_out
```

C++ 的 API 文档是由 doxygen 产生的 XML 文件生成的，不过此工具暂未开源。

## Markdown 与 Notebook

TensorFlow 的文档是由 Markdown（`.md`）或者 Notebook（`.ipynb`）编写的。除少数情况外，TensorFlow 文档遵循[标准 Markdown 规则](https://daringfireball.net/projects/markdown/)。

这一节介绍标准 Markdown 语法规则和可编辑的 TensorFlow 文档中使用的 Markdown 语法规则之间的主要差异。

### Markdown 中的数学公式

在编辑 Markdown 文件时，你可以在 TensorFlow 中使用 MathJax，但是需要注意以下几点：

- MathJax 可以在 [tensorflow.org](https://www.tensorflow.org) 上正确地渲染
- MathJax 可能在 [github](https://github.com/tensorflow/tensorflow) 上无法正确地渲染。

在写 MathJax 的时候，你可以使用 <code>&#36;&#36;</code> 和 `\\(`、`\\)` 将数学公式包起来。<code>&#36;&#36;</code> 会导致会换行，所以在文本行内使用 `\\(`、`\\)`。

### Markdown 中的链接

链接可以分为几种类型：

- 指向文档库中其它文档的链接
- 指向 API 文档的链接
- 其它的链接

对于指向文档库中其它文档的链接，使用相对链接路径，比如：`[Eager Basics](../tutorials/eager/eager_basics.ipynb)` 会产生 [Eager Basics](../tutorials/eager/eager_basics.ipynb) 链接。这些链接在 github 与 tensorflow.org 中都能正常使用。

API 链接会在文档网站发布时进行转换。生成指向 Python API 的链接仅需在反引号内输入完整的符号路径即可：\`tf.data.Dataset\` 会产生 `tf.data.Dataset` 链接。生成指向 C++ API 的链接需要使用命名空间路径，比如 \`tensorflow::Tensor\` 会产生 `tensorflow::Tensor` 链接。

对于其它不在 `tensorflow/docs` 仓库中，指向 tensorflow.org 的链接（包括 [ecosystem](https://tensorflow.org/ecosystem) 中的全部链接），直接用标准 markdown 语法填入完整的 URL 即可。

对于指向源码的链接，使用以 `https://www.tensorflow.org/code/` 开头的链接，后面接上以 github 仓库根目录开始的文件名即可。

以上的 URL 指南可以确保 [tensorflow.org](https://www.tensorflow.org/) 网站根据用户正在浏览的文档版本正确地指向相应源码。在指向源码的链接中不要加入任何 url 参数。

## 操作（op）文档样式指南

模块级别的、很长的、描述性的文档应该在 `docs_src/api_guides/python` 的 API 指南中。

在一般情况下，类（class）和操作（op）的文档会按照顺序提供以下信息：

* 用一个简短的句子描述操作（op）作用。
* 简短描述传递参数给操作（op）时会发生什么。
* 一个显示操作（op）如何工作的示例（最好有伪代码）。
* 要求、注意事项、重要说明（如果有的话）。
* 对操作（op）构造函数的输入、输出、Attrs 或其他参数的描述。

上述每一项都在[下面](#文档字符串章节的描述)有更详细的描述。

请使用 Markdown 来写文档。基本语法请参考[此文档](https://daringfireball.net/projects/markdown/)。你也可以使用 [MathJax](https://www.mathjax.org) 符号来写公式（见上文有关限制）。

### 关于代码的写法

在文本中书写以下内容时需要用反引号包起来：

* 参数名（例如 `input`、`x`、`tensor`）
* 返回的张量名（例如 `output`、`idx`、`out`）
* 数据类型（例如 `int32`、`float`、`uint8`）
* 文本中引用的其他操作（op）名（例如 `list_diff()`、`shuffle()`）
* 类（class）名（例如使用 `Tensor`，实际上你用它来表示一个 `Tensor` 对象；如果你只是要解释一个操作（op）要对一个张量、一幅图做什么，或者是要解释某种一般的操作，不要大写或使用反单引号。
* 文件名（例如 `image_ops.py` 或 `/path-to-your-data/xml/example-name`）
* 数学表达式或条件（例如 `-1-input.dims() <= dim <= input.dims()`）

用三个反单引号样例代码和伪代码示例包起来。并使用 `# ==>` 而非等号来表达一个操作（op）将返回什么。例如：

    ```
    # 'input' is a tensor of shape [2, 3, 5]
    (tf.expand_dims(input, 0))  # ==> [1, 2, 3, 5]
    ```

如果你要提供一个 Python 的示例代码，请添加 Python 风格标签以确保使用合适的语法高亮：

    ``` python
    # some Python code
    ```

Markdown 中关于示例代码反单引号的两点说明：

1. 如果有必要的话，你可以使用反单引号来美观地显示除 Python 外的其他语言，[这里](https://github.com/google/code-prettify#how-do-i-specify-the-language-of-my-code)有一份可用语言的完整列表。

2. Markdown 也允许你使用四个空格的缩进来指示一段代码示例。但是，请一定不要同时使用四个空格缩进和反单引号。只使用其中的一个。

### 张量维度

当你提及一般的张量（tensor）时，不要大写这个词的首字母。当你在提及作为参数提供给操作（op）或由操作 (op) 返回的特定对象时，你应该使用 Tensor 这个词，并在其周围添加反单引号，因为你提及的是一个 `Tensor` 对象。

不要使用 `Tensors` 这个词来表示多个 Tensor 对象，除非你真的在谈论一个 `Tensors` 对象。更好的说法是“许多 `Tensor` 对象（a list of `Tensor` objects）”。

使用术语“维度”来表示张量的大小。如果您需要指定具体的大小，请使用以下约定：

- 标量表示“0 维张量”
- 向量表示“1 维张量”
- 矩阵表示“2 维张量”
- 张量表示“3 维张量”或“n 维张量”。“秩”这个词真正有用的时候再用，不然就用维度。永远不要用“阶数”这个词来描述张量的大小。

使用“形状”（shape）这个词来具体说明张量的维度，并用一对方括号来展示一个张量的形状。例如：

    如果 `input` 是一个形状为 `[3, 4, 3]` 的三维张，则这个操作会返回一个形状为 `[6, 8, 6]` 的三维张量。

### C++ 中定义的操作（op）

所有在 C++ 中定义的操作（并且可以通过其他语言访问）必须用 `REGISTER_OP` 声明进行记录。C++ 文件中的文档字符串会自动进行处理，为输入类型、输出类型和 Attr 类型以及默认值添加相关信息。

例如：

```c++
REGISTER_OP("PngDecode")
  .Input("contents: string")
  .Attr("channels: int = 0")
  .Output("image: uint8")
  .Doc(R"doc(
Decodes the contents of a PNG file into a uint8 tensor.

contents: PNG file contents.
channels: Number of color channels, or 0 to autodetect based on the input.
  Must be 0 for autodetect, 1 for grayscale, 3 for RGB, or 4 for RGBA.
  If the input has a different number of channels, it will be transformed accordingly.
image:= A 3-D uint8 tensor of shape `[height, width, channels]`.
  If `channels` is 0, the last dimension is determined from the png contents.
)doc");
```

会输出以下 Markdown：

    ### tf.image.png_decode(contents, channels=None, name=None) {#png_decode}

    Decodes the contents of a PNG file into a uint8 tensor.

    #### Args:

    *  **contents**: A string Tensor. PNG file contents.
    *  **channels**: An optional int. Defaults to 0.
       Number of color channels, or 0 to autodetect based on the input.
       Must be 0 for autodetect, 1 for grayscale, 3 for RGB, or 4 for RGBA. If the input has a different number of channels, it will be transformed accordingly.
    *  **name**: A name for the operation (optional).

    #### Returns:
    A 3-D uint8 tensor of shape `[height, width, channels]`.  If `channels` is 0, the last dimension is determined from the png contents.

大多数的参数描述都是被自动添加的。需要注意的是，文档生成器会自动为所有的输入、属性和输出添加名称和类型。在上面的例子中，`contents: A string Tensor.` 是自动添加的。你应该使用一些额外的文本来让描述更加自然。

对于输入和输出，你可以在额外添加的文字前加等号来避免生成器自动添加名称和类型。在上面的例子中，描述中名为 `image` 的输出需要在开头加上 `=`，以免生成器自动在文字 `A 3-D uint8 Tensor...` 前添加 `A uint8 Tensor.`。你不能通过这种方式来避免添加属性的名称、类型和默认值，因此要小心书写这些额外的文本。

### Python 中定义的操作（op）

如果你的操作（op）是在 `python/ops/*.py` 文件中定义的，则需要为所有参数和输出（返回值）张量提供描述文本。文档成器不会对 Python 中定义的操作（op）的任何文本进行自动生成，所以你写什么就会得到什么。

你应该遵守常见的 Python 文档字符串的约定，但您应该在文档字符串中使用 Markdown 语法格式。

以下是一个简单的例子：

    def foo(x, y, name="bar"):
      """Computes foo.

      Given two 1-D tensors `x` and `y`, this operation computes the foo.

      Example:

      ```
      # x is [1, 1]
      # y is [2, 2]
      tf.foo(x, y) ==> [3, 3]
      ```
      Args:
        x: A `Tensor` of type `int32`.
        y: A `Tensor` of type `int32`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `int32` that is the foo of `x` and `y`.

      Raises:
        ValueError: If `x` or `y` are not of type `int32`.
      """

## 文档字符串章节的描述

本节会详细介绍文档字符串中的每个元素。

### 描述操作（op）是做什么的短句

示例：

```
拼接张量。
```

```
从左向右水平翻转一幅图像。
```

```
计算两个序列之间的 Levenshtein 距离。
```

```
将一些张量保存到一个文件中。
```

```
从一个张量中提取一些切片。
```

### 对传递参数给这个操作（op）后会发生什么的简要描述

示例：

    给定一个数值类型的张量输入，这个操作会返回一个相同类型和大小但是值会按照 `seq_dim` 维度逆序的张量。向量 `seq_lengths` 确定维度 0（通常是 batch 对应的维度）内的每个索引都逆序了哪些元素。

    这一操作会返回一个类型为 `dtype`、维度为 `shape` 的张量，并且所有的元素都被置为 0。


### 对操作（op）进行举例说明

好的代码示例既简短也易于理解，通常包含一个简短的代码段用以阐明示例在说明什么。当操作（op）是对一个 Tensor 的形状进行操作时，在示例中分别写出操作前和操作后的张量是非常有用的。

`squeeze()` 操作有一个非常好的伪代码示例：

    # 't' 是一个形状为 [1, 2, 1, 3, 1, 1] 的张量
    shape(squeeze(t)) ==> [2, 3]

`tile()` 操作在描述性文本方面提供了一个很好的样例：

    例如，用 `[2]` 对 `[a, b, c, d]` 进行 `tile()` 操作会产生 `[a b c d a b c d]`。

最好使用 Python 来书写示例代码。永远不要将它们放到 C++ 的操作文件中，并且也要避免将它们放到 Python 的操作文档中。我们推荐尽可能地将示例代码放入 [API 指南](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/docs_src/api_guides)中。或者，将它们添加到调用操作构造函数的模块或者类的文档字符串中。

以下为 `api_guides/python/math_ops.md` 模块级文档字符串的示例：

    ## 分割

    TensorFlow 有多种操作方式让你在张量分割时进行各种数学运算。
    ...
    特别地，对矩阵张量的分割操作是一个矩阵的各行到分割片段的映射。


    例如：

    ```python
    c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])
    tf.segment_sum(c, tf.constant([0, 0, 1]))
      ==>  [[0 0 0 0]
            [5 6 7 8]]
    ```

### 要求、注意事项、重要说明

示例：

```
这个操作要求 `-1-input.dims() <= dim <= input.dims()`
```

```
说明：如果对这个张量求值会产生错误。它的值必须使用 `Session.run()`、`Tensor.eval()` 或 `Operation.run()` 的可选参数 `feed_dict` 导入 。
```

### 对参数和输出（返回）张量的描述

要保证描述言简意赅。在参数部分无须介绍这个操作的工作原理。

如果这个操作对输入张量或输出张量的维度有强制要求，需要在此处提及。请记住，对于 C++ 操作，张量的类型是自动添加的，例如“A ..type .. Tensor”或“一个类型为 {...list of types...} 的 Tensor”。在这种情况下，如果操作对张量的维度有约束，则可以添加诸如“必须为 4 维”的文本，或者在开头用 `=`（为了防止添加张量类型）进行描述，例如写上“一个 4 维的 float 张量”。

例如，以下描述了两种对一个 C++ 操作中 image 参数进行记录的方式（注意 “=” 号）：

```
image: 要进行调整尺寸的图像。必须为 4 维。
```

```
image:= 要进行调整尺寸的图像。必须为一个 4 维的 `float` 型张量。
```

在文档中，这些会被渲染成如下所示的 Markdown 格式

```
image: 要进行调整尺寸的图像。是一个 `float` 型张量，必须为 4 维。
```

```
image: 要进行调整尺寸的图像。是一个 4 维的`float` 型张量。
```

### 可选参数说明（“attrs”）

文档生成器总是会描述所有 attr 的类型及其默认值（如果有的话）。由于 C++ 和 Python 生成的文档的描述互不相同，因此不能用等号来覆写。

请在进行额外的 attr 描述时使用合适的语句，以便将它加在类型和默认值之后也能流畅阅读。首先展示类型和默认值，最后是附加说明。因此，最好使用完整的句子进行描述。

这里有一个 `image_ops.cc` 中的例子：

    REGISTER_OP("DecodePng")
        .Input("contents: string")
        .Attr("channels: int = 0")
        .Attr("dtype: {uint8, uint16} = DT_UINT8")
        .Output("image: dtype")
        .SetShapeFn(DecodeImageShapeFn)
        .Doc(R"doc(
    Decode a PNG-encoded image to a uint8 or uint16 tensor.

    The attr `channels` indicates the desired number of color channels for the decoded image.

    Accepted values are:

    *   0: Use the number of channels in the PNG-encoded image.
    *   1: output a grayscale image.
    *   3: output an RGB image.
    *   4: output an RGBA image.

    If needed, the PNG-encoded image is transformed to match the requested number of color channels.

    contents: 0-D.  The PNG-encoded image.
    channels: Number of color channels for the decoded image.
    image: 3-D with shape `[height, width, channels]`.
    )doc");

上述文本将在 `api_docs/python/tf/image/decode_png.md` 中生成如下的 Args 描述：

    #### Args:

    * **`contents`**: A `Tensor` of type `string`. 0-D.  The PNG-encoded image.
    * **`channels`**: An optional `int`. Defaults to `0`. Number of color channels for the decoded image.
    * **`dtype`**: An optional `tf.DType` from: `tf.uint8, tf.uint16`. Defaults to `tf.uint 8`.
    * **`name`**: A name for the operation (optional).
    
