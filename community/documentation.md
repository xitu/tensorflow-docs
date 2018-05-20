# 编写 TensorFlow 文档

我们欢迎社区对 TensorFlow 文档做贡献。这份文档说明了你可以怎样为 TensorFlow 文档做出贡献，特别地，这份文档对以下内容进行了说明：

* 文档位于什么地方。
* 怎样进行格式一致的编辑。
* 在提交之前，如何构建并测试你对文档做的改动。

你可以在 https://www.tensorflow.org 上查看 TensorFlow 文档，也可以查看并编辑它在 [GitHub](https://www.tensorflow.org/code/tensorflow/docs_src/) 上的原始文件。我们正在将这些文档发布到 GitHub 上，以便每个人都可以为之做贡献。所有经过核对编入 `tensorflow/docs_src` 的内容之后都会尽快地发布到 [https://www.tensorflow.org](https://www.tensorflow.org)。

我们非常欢迎通过不同的形式重新发布 TensorFlow 文档，但我们不大可能允许让别的格式的文档（或者其他的文档生成工具）进入我们的代码仓库。如果你想以另外的格式重新发布我们的文档，请确保包含以下内容：

* 这种格式的文档对应的 API 版本（例如 r1.0、master 等等）
* 这份文档是从哪次提交或者哪个版本产生的
* 从哪里（即 https://www.tensorflow.org）可以找到最新版本的文档
* Apache 2.0 开源许可协议

## 关于版本的说明

在 tensorflow.org 网站的根目录下有针对 Tensorflow 最新的稳定版二进制文档。如果你在用 `pip` 命令来安装 TensorFlow，你应该阅读这份文档。


但是，大多数的开发者都是向 GitHub 的 master 分支里的文档做贡献，这份文档会在 [tensorflow.org/versions/master](https://www.tensorflow.org/versions/master) 不定时地发布。

如果你想对出现在网站的根目录中的文档做一些改动，你需要将文档的变动提交到当前的稳定版二进制分支 (和/或 [cherrypick](https://stackoverflow.com/questions/9339429/what-does-cherry-picking-a-commit-with-git-mean) 分支)。

## 参考文档与非参考文档

下面的 reference documentation 是由代码中的注释自动生成的：

- C++ API 参考文档
- Java API 参考文档
- Python API 参考文档

如果想修改参考文档，你需要编辑对应的代码注释。

非参考文档 (例如 TensorFlow 的安装指南) 是由人们撰写的。

这份文档位于 [`tensorflow/docs_src`](https://www.tensorflow.org/code/tensorflow/docs_src/) 目录。`docs_src` 的每个子目录下包含一系列相关 TensorFlow 文档。例如，TensorFlow 的安装指南全部位于 `docs_src/install` 目录中。

C++ 文档是通过文档生成工具从 XML 文件生成的；但是这些工具现在还没有开源。

## Markdown

可编辑的 TensorFlow 文档都是用 Markdown 编写的。除了一些例外的情况，TensorFlow使用[标准 Markdown 语法规则](https://daringfireball.net/projects/markdown/)。


这一节介绍标准 Markdown 语法规则和可编辑的 TensorFlow 文档中使用的 Markdown 语法规则之间的主要差异。

### Markdown 中的数学公式

在编辑 Markdown 文件时，你可以在 TensorFlow 中使用 MathJax，但是需要注意以下几点：

- MathJax 可以在 [tensorflow.org](https://www.tensorflow.org) 上正确地渲染
- MathJax 不能在 [github](https://github.com/tensorflow/tensorflow) 上正确地渲染

在写 MathJax 的时候，你可以使用 <code>&#36;&#36;</code> 和 `\\(`、`\\)` 将数学公式包起来。<code>&#36;&#36;</code> 会导致会换行，所以在文本行内使用 `\\(`、`\\)`。

### Markdown 中的链接

链接可以分为几种类型：

- 指向同一文件中不同部分的链接
- 指向 tensorflow.org 网站外其他 URL 地址的链接
- 从一个 Markdown 文件 (或者代码注释) 指向 tensorflow.org 网站内其他文件的链接。

对于前两种链接类型，你可以使用标准的 Markdown 链接，但要把链接全部都放在一行，而不是把它拆分成多行。例如：

- `[text](link)    # 好的链接`
- `[text]\n(link)  # 不好的链接`
- `[text](\nlink)  # 不好的链接`

对于最后一种类型的链接 (指向 tensorflow.org 网站内其他文件的链接)，请使用一种特殊的链接参数化机制，以使作者不改动链接也能移动和重新组织文件。

这种参数化机制具体如下，使用：

<!-- 注意：t&#64; 的使用是一种 hack，所以我们没有把它转换成符号 -->
- <code>&#64;{tf.symbol}</code> 链接到 Python 符号参考页面。需要注意，类成员没有自己的页面，但是这种语法仍然奏效，因为 <code>&#64;{tf.MyClass.method}</code> 会链接到`tf.MyClass` 页面合适的部分。

- <code>&#64;{tensorflow::symbol}</code> 链接到C++ 符号页面。

- <code>&#64;{$doc_page}</code> 链接到另一个文档页面（非 API 参考）。为了链接到

    - `red/green/blue/index.md`，使用 <code>&#64;{$blue}</code> 或者 <code>&#64;{$green/blue}</code>，

    - `foo/bar/baz.md`，使用 <code>&#64;{$baz}</code> 或者 <code>&#64;{$bar/baz}</code>。

    应该首选较短的这个，这样我们就可以在不破坏这些引用的情况下移动页面。主要的例外是 Python API 指南应该使用 <code>&#64;{$python/<guide-name>}</code> 引用以避免产生歧义。

- <code>&#64;{$doc_page#anchor-tag$link-text}</code> 链接到该文档的另一个链接标记并使用不同的链接文本（在默认情况下，链接文本是目标页面的标题）。


    如果只要重写链接文本，可以忽略 `#anchor-tag`。

如果要链接到源代码，使用以`https://www.tensorflow.org/code/` 开头的链接，然后跟上它在 github 根目录下的文件名。例如，你现在正在读的这个文件的
链接应该写作 `https://www.tensorflow.org/code/tensorflow/docs_src/community/documentation.md`。

这种 URL 的命名方式可以确保 [tensorflow.org](https://www.tensorflow.org/) 能够将链接转到你正在阅读的相应版本文档的代码。一定不要在源代码的 URL 中添加链接参数。

## 生成文档和预览链接

在构建文档之前，你必须先完成以下步骤来安装环境：


1. 如果你的机器没有安装 bazel，现在就装上它。如果你用的是 Linux，使用下面的命令来安装 bazel：

        $ sudo apt-get install bazel  # Linux

    如果你用的是 Mac OS, 可以去[这个页面](https://bazel.build/versions/master/docs/install.html#mac-os-x) 找 bazel 的安装说明。

2. 切换至 TensorFlow 源代码中 `tensorflow` 文件夹所在的目录。

3. 运行 `configure` 脚本，按照提示，根据你的系统选择合适的指令。

        $ ./configure

然后，切换到包含 `docs_src` 的 `tensorflow` 目录 (`cd tensorflow`)。运行下面的命令来编译 TensorFlow 并在 `/tmp/tfdocs` 目录中生成文档：

    bazel run tools/docs:generate -- \
              --src_dir="$(pwd)/docs_src/" \
              --output_dir=/tmp/tfdocs/

注意：你必须将 `src_dir` 和 `output_dir` 设置为绝对文件路径。

## 生成 Python API 文档

操作 (op)、类 (class)、功能函数 (utility functions) 都是在 Python 模块中定义的，例如 `image_ops.py`。Python 中模块都包含一个模块文档字符串。例如：

```python
"""Image processing and decoding ops."""
```

文档生成器会把模块文档字符串放在针对该模块生成的 Markdown 文件的开头，在这个例子中就是 [tf.image](https://www.tensorflow.org/api_docs/python/tf/image)。

有时候需要在模块文件的开头列出一个模块的所有成员，这时可以在每个成员前添加 `@@`。`@@member_name` 语法已经被废弃了，不会生成任何文档。但是根据模块的[密封方式](#密封模块)，有可能需要将一个模块的内容元素标记为公开的。被调用的操作 (op)、函数 (function)或类 (class) 不必在同一个文件中进行定义。本文档接下来的几个部分将讨论密封以及如何向公共文档添加元素。


新的文档系统会自动记录公共符号，但不包括下面这些：

- 以下划线开头的私有符号。
- `object` 中原本定义的符号或 protobuf 的 `Message`。
- 一些类的成员，例如被动态创建但通常有没有有用的文档的 `__base__`、`__class__`。

只有上层模块（目前只有 `tf` 和 `tfdbg`）需要被手动添加到生成脚本中。

### 密封模块

因为文档生成器会遍历所有可见的符号，并深入它能找到的任何东西，所以它会记录意外暴露出来的符号。如果一个模块只会暴露那些有意作为开放 API 的一部分的符号，我们就称其为**密封的**。由于 Python 有宽松的导包机制和可见性的传统，那些写得很幼稚的 Python 代码会无意中暴露出很多实现细节的模块。密封不合理的模块可能会暴露出其他没有封装好的模块，这往往会导致文档生成器生成失败。**这种失败是符合预期的行为。** 它确保我们的 API 定义良好，并允许我们无需担心意外中断用户，就能更改实施细节（包括哪些模块被导入到哪里）。


如果一个模块被意外导入了，它通常会中断文档生成器(`generate_test`)。这是你需要密封你的模块的一个明显标志。然而即使文档成功生成了，其中也可能会出现一些不需要的符号。检查生成的文档以确保所有被记录的符号都是符合预期的。如果一个地方出现了不该出现的符号，你有下面几个选项可以对它们进行处理：

- 私有符号和导入
- `remove_undocumented` 过滤器
- 一个可遍历的黑名单。

下面我们会详细地讨论这些选项。

#### 私有符号和导入

使 API 密封符合要求的最简单的方法就是将非公开的符号私有化（通过预先加下划线 `_`）。文档生成器会遵守私有符号，这同样也适用于模块。如果唯一的问题是
文档中显示少量导入的模块（或打破了生成器），你可以简单地在导入时重命名它们，例如：`import sys as _sys`。

因为 Python 会将所有的文件视为模块，所以这也适用与文件。
如果你的文件中包含下面两个文件/模块：

    module/__init__.py
    module/private_impl.py

在 `module` 被导入之后，有可能可以访问
`module.private_implit`。将 `private_impl.py` 重命名为 `_private_impl.py`
可以解决这个问题。如果重命名模块比较尴尬，请继续阅读。

#### 使用 `remove_undocumented` 过滤器

密封模块的另一种方式是从 API 中拆分出来你的实现。
为此，请考虑使用 `remove_undocumented`，其中包含允许的符号列表，
并从模块中删除其他所有内容。
例如，以下代码段演示了如何将 `remove_undocumented` 一个模块的 `__init__.py` 文件中：


__init__.py:

    # 只有在定义了 __all__ 的文件中使用 * 导入
    from tensorflow.some_module.some_file import *

    # 否则就直接导入符号
    from tensorflow.some_module.some_other_file import some_symbol

    from tensorflow.python.util.all_util import remove_undocumented

    _allowed_symbols = [‘some_symbol’, ‘some_other_symbol’]

    remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)

该 `@@member_name` 语法已经被废弃了，但它仍然作为 `remove_undocumented` 的指示器在文档中的某些地方存在，
用于指示这些符号是公开的。
所有的 `@@` 最终都将被删除。但是，如果你看到它们，
请不要随意删除，因为我们的某些系统仍在使用它们。


#### 遍历黑名单

如果上述的方法都失败了，你可以在遍历黑名单中添加条目 
`generate_lib.py`。**这个列表中几乎所有的条目都是对它存在的目的的一种滥用；尽量避免向这个列表中添加东西。**


遍历黑名单将合法的模块名称（不带前缀 `tf.`）
映射到不向下遍历到的局部名称。例如，
以下条目将从遍历中排除 `some_module`。

    { ...
      ‘contrib.my_module’: [‘some_module’]
      ...
    }

这意味着文档生成器将显示 `some_module` 存在，
但它不会枚举其内容。

这个黑名单最初是为了确保用于平台抽象的系统模块（mocks、
flags 等等）可以被记录下来，
而无需记录其内部空间。对于 contrib 而言，它超出此目的的用途
是一个可以接受的捷径，但对于 core tensorflow 来说并不是。

## 操作 (op) 文档样式指南

关于模块的很长的、描述性模块级文档应该在 
`docs_src/api_guides/python` 的 API 指南中。

在理想情况下，对于类 (class) 和 操作 (op)，你应该按照演示的顺序，提供以下信息：


* 一个描述操作 (op) 作用的简短的句子。
* 一个关于当传递参数给操作 (op) 时会发生什么的简短描述。
* 一个显示操作 (op) 如何工作的示例（最好有伪代码）。
* 要求、注意事项、重要说明（如果有的话）。
* 对操作 (op) 构造函数的输入、输出、Attrs 或其他参数的描述。


每一项都在[下面](#文档字符串章节的描述)
有更详细的描述。

用 Markdown 来写。基本语法参考在
[这里](https://daringfireball.net/projects/markdown/).
你可以使用 [MathJax](https://www.mathjax.org) 符号来写公式 (见上文有关
限制)。

### 关于代码的写法

下面这些东西用于文本中间时需要用反单引号包起来：

* 参数名（例如 `input`、`x`、`tensor`）
* 返回的张量名（例如 `output`、`idx`、`out`）
* 数据类型（例如 `int32`、`float`、`uint8`）
* 文本中引用的其他操作 (op) 名（例如 `list_diff()`、`shuffle()`）
* 类 (class) 名（例如使用 `Tensor`，实际上你用它来表示一个 `Tensor` 对象；
  如果你只是要解释一个操作 (op) 要对一个张量、一幅图做什么，或者是要解释某种一般的操作，不要大写或使用反单引号。

* 文件名（例如 `image_ops.py` 或
`/path-to-your-data/xml/example-name`）
* 数学表达式或条件（例如 `-1-input.dims() <= dim <=
  input.dims()`）

用三个反单引号样例代码和伪代码示例包起来。
还要用 `==>` 而不是一个等号用于表示一个操作 (op) 返回什么。
例如：

    ```
    # 'input' 是形状为 [2, 3, 5] 的一个张量
    (tf.expand_dims(input, 0)) ==> [1, 2, 3, 5]
    ```

如果你要提供一个 Python 的示例代码，添加 Python 风格标签以确保完成
合适的语法高亮：

    ```python
    # 一些 Python 代码
    ```

Markdown 中关于示例代码反单引号的两点说明：

1. 如果有必要的话，你可以使用反单引号来美观地显示除 Python 外的其他语言，
[这里](https://github.com/google/code-prettify#how-do-i-specify-the-language-of-my-code)有一份可用语言的完整列表。

2. Markdown 也允许你使用四个空格的缩进来指示一段代码示例。
   但是，一定不要同时使用四个空格缩进和反单引号。
   只使用其中的一个。

### 张量维度

当你在谈论一般的张量 (tensor) 时，不要大写这个词的首字母。
当你在谈论作为
参数提供给操作 (op) 或由操作 (op) 返回的特定对象时，你应该使用 Tensor 这个词，并在其周围
添加反单引号，因为你在谈论一个 `Tensor` 对象。

不要使用 `Tensors` 这个词来表示多个 Tensor 对象，除非你
真的在谈论一个 `Tensors` 对象。更好的说法是“许多 `Tensor` 对象 (a list of `Tensor` objects)”。


使用术语“维度”来表示张量的大小。如果您需要
指定具体的大小，请使用以下约定：

- 标量表示“0维张量”
- 向量表示“1维张量”
- 矩阵表示“2维张量”
- 张量表示“3维张量”或“n维张量”
  “秩”这个词真正有用的时候再用，不然就用维度。
  永远不要用“阶数”这个词来描述张量的大小。

使用“形状”这个词来具体说明张量的维度，并用一对方括号来展示一个张量的形状。
例如：

    如果 `input` 是一个形状为 `[3, 4, 3]` 的三维张量，
    这个操作会返回一个形状为 `[6, 8, 6]` 的三维张量。

### C++ 中定义的操作 (op)

所有在 C++ 中定义的操作（并且可以通过其他语言访问）必须用 `REGISTER_OP` 声明来记录。
C++ 文件中的文档字符串经过处理会
自动为输入类型、输出类型和 Attr 类型以及默认值添加一些信息。


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
      If the input has a different number of channels, it will be transformed
      accordingly.
    image:= A 3-D uint8 tensor of shape `[height, width, channels]`.
      If `channels` is 0, the last dimension is determined
      from the png contents.
    )doc");
    ```

会输出下面的 Markdown 结果：

    ### tf.image.png_decode(contents, channels=None, name=None) {#png_decode}

    Decodes the contents of a PNG file into a uint8 tensor.

    #### Args:

    *  <b>contents</b>: A string Tensor. PNG file contents.
    *  <b>channels</b>: An optional int. Defaults to 0.
       Number of color channels, or 0 to autodetect based on the input.
       Must be 0 for autodetect, 1 for grayscale, 3 for RGB, or 4 for RGBA.  If the
       input has a different number of channels, it will be transformed accordingly.
    *  <b>name</b>: A name for the operation (optional).

    #### Returns:
    A 3-D uint8 tensor of shape `[height, width, channels]`.  If `channels` is
    0, the last dimension is determined from the png contents.

大多数的参数描述都是被自动添加的。特别的是，
文档生成器会自动为所有的输入、属性和输出添加名称和类型。
在上面的例子中，`<b>contents</b>: A string Tensor.` 是被自动添加的。
你应该在那个描述之后多写点额外的文字让它顺畅自然一些。


对于输入和输出，你可以在额外的文字前加等号来
避免自动添加名称和类型。
在上面的例子中，名称为 `image` 的输出描述以 `=` 开头以避免在文字 `A 3-D uint8 Tensor...` 前添加 `A uint8 Tensor.`。
你不能通过这种方式来避免添加属性的名称、类型和默认值，
所以要小心写这些文本。


### Python 中定义的操作 (op)

如果你的操作 (op) 是在 `python/ops/*.py` 文件中定义的，则需要
为所有参数和输出（返回值）张量提供文本。文档成器
不会自动生成 Python 中定义的操作 (op) 的任何文本，所以你写什么
就会得到什么。

你应该遵守常见的 Python 文档字符串约定，但您
应该在文档字符串中使用 Markdown 语法格式。

这是一个简单的例子：

    def foo(x, y, name="bar"):
      """计算 foo 函数。

      给定两个一维张量 `x` 和 `y`，这个操作会计算 foo 函数。

      示例：

      ```
      # x 的形状是 [1, 1]
      # y 的形状是 [2, 2]
      tf.foo(x, y) ==> [3, 3]
      ```
      参数：
        x: 一个 `int32` 类型的 `Tensor`。
        y: 一个 `int32` 类型的 `Tensor`。
        name: 操作的名字（可选）。

      返回值：
        一个 `int32` 类型的 `Tensor`，是 `x` 和 `y` 的 foo 函数值

      错误引发:
        ValueError: If `x` or `y` are not of type `int32`.
      """

## 文档字符串章节的描述

这一节会详细介绍文档字符串中的每个元素。

### 描述操作 (op) 是做什么的短句

示例：

```
连接张量。
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

### 对传递参数给这个操作 (op) 后会发生什么的简要描述

示例：

    给定一个数值类型的张量输入，这个操作会返回一个
    相同类型和大小但是值会按照 `seq_dim` 维度逆序的张量。
    向量 `seq_lengths` 确定
    维度 0（通常是 batch 对应的维度）内的每个索引都逆序了哪些元素。


    这一操作会返回一个类型为 `dtype`、维度为 `shape` 的张量，
    并且所有的元素都被置为 0。

### 对操作 (op) 举例说明

好的代码示例很简单，易于理解，通常包含一个简短的
代码段用以阐明示例在说明什么。当操作 (op) 是对一个 Tensor 的形状进行
操作时，在示例中包含
操作前和操作后往往也是有用的。

`squeeze()` 操作有一个非常好的伪代码示例：

    # 't' 是一个形状为 [1, 2, 1, 3, 1, 1] 的张量
    shape(squeeze(t)) ==> [2, 3]

`tile()` 操作在描述性文本方面提供了一个很好的样例：

    例如，用 `[2]` 对 `[a, b, c, d]` 进行 `tile()` 操作会产生 `[a b c d a b c d]`。

最好使用 Python 来展示示例代码。永远不要将它们放到 C++ 的
操作文件中，并且也要避免将它们放到 Python 的操作文档中。我们推荐
尽可能地将代码示例放到
[API 指南](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/docs_src/api_guides)。
否则，将它们添加到调用操作构造函数的模块或者类的文档字符串中。


在 `api_guides/python/math_ops.md` 有一个模块文档字符串的例子：

    ## 分割

    TensorFlow 有多种操作方式供你
    执行数学运算和张量分割。
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
说明：如果对这个张量求值会产生错误。它的值必须
使用 `Session.run()`、
`Tensor.eval()` 或 `Operation.run()` 的可选参数 `feed_dict` 导入 。
```

### 对参数和输出（返回）张量的描述

要保证描述言简意赅。不应该
在参数部分介绍这个操作是如何工作的。

如果这个操作对输入张量或输出张量的维度有很强的限制，这里就应该提及。
请记住，对于 C++ 操作，张量的类型是自动添加的，
为“A ..type .. Tensor”或“A类型在{...list of types...}”。
在这种情况下，如果操作对
张量的维度有约束，则可以添加诸如“必须为 4 维”的文本，或者用
`=`（为了防止添加张量类型）开始描述，例如写上“一个 4 维的 
float 张量”。

例如，这里有两种方式来对一个 C++ 操作中 image 参数进行记录（注意
符号“=”）：

```
image: 必须为 4 维。要调整尺寸的图像。
```

```
image:= 一个 4 维的 `float` 型张量。要调整尺寸的图像。
```

在文档中，这些会被渲染成如下所示的 Markdown 格式

```
image: 一个 `float` 型张量。必须为 4 维。要调整尺寸的图像。
```

```
image: 一个 4 维的 `float` 型张量。要调整尺寸的图像。
```

### 可选参数说明（“attrs”）

文档生成器总会描述每个 attr 的类型及其默认
值（如果有的话）。
由于 C++ 和 Python 生成的文档的描述是非常不同的，因此您不能使用等号来覆盖它。

对任何额外的 attr 描述选择合理的措辞，以便在类型
和默认值之后能够读得流畅。首先展示类型和默认值，后面是附加
说明。因此，完整的句子是最好的。

这里有一个 `image_ops.cc` 中的例子：

    REGISTER_OP("DecodePng")
        .Input("contents: string")
        .Attr("channels: int = 0")
        .Attr("dtype: {uint8, uint16} = DT_UINT8")
        .Output("image: dtype")
        .SetShapeFn(DecodeImageShapeFn)
        .Doc(R"doc(
    Decode a PNG-encoded image to a uint8 or uint16 tensor.

    The attr `channels` indicates the desired number of color channels for the
    decoded image.

    Accepted values are:

    *   0: Use the number of channels in the PNG-encoded image.
    *   1: output a grayscale image.
    *   3: output an RGB image.
    *   4: output an RGBA image.

    If needed, the PNG-encoded image is transformed to match the requested
    number of color channels.

    contents: 0-D.  The PNG-encoded image.
    channels: Number of color channels for the decoded image.
    image: 3-D with shape `[height, width, channels]`.
    )doc");

这将在 `api_docs/python/tf/image/decode_png.md`
中生成下面的 Args 部分：

    #### Args:

    * <b>`contents`</b>: A `Tensor` of type `string`. 0-D.  The PNG-encoded
      image.
    * <b>`channels`</b>: An optional `int`. Defaults to `0`. Number of color
      channels for the decoded image.
    * <b>`dtype`</b>: An optional `tf.DType` from: `tf.uint8,
      tf.uint16`. Defaults to `tf.uint 8`.
    * <b>`name`</b>: A name for the operation (optional).
