# Embeddings

本文档介绍了 embeddings 的概念，给出了如何在 TensorFlow 中对 embedding 进行训练的简单示例，并解释了如何使用 TensorBoard Embedding Projector ([live example](http://projector.tensorflow.org)) 查看 embeddings。前两部分是针对机器学习和 TensorFlow 的新手，而 Embedding Projector 则是针对所有水平的用户。

有关这些概念的另一个教程可以在[机器学习速成课程的 Embeddings 部分](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture)中找到。

[TOC]

**Embedding** 是从离散对象（如单词）到实数向量的映射。例如，英文单词的 300 维 embedding 可以如下所示：

```
blue:  (0.01359, 0.00075997, 0.24608, ..., -0.2524, 1.0048, 0.06259)
blues:  (0.01396, 0.11887, -0.48963, ..., 0.033483, -0.10007, 0.1158)
orange:  (-0.24776, -0.12359, 0.20986, ..., 0.079717, 0.23865, -0.014213)
oranges:  (-0.35609, 0.21854, 0.080944, ..., -0.35413, 0.38511, -0.070976)
```

这些向量中的各个维度通常没有固定的意义。相反，机器学习利用的是向量之间的位置和距离的整体模式。

Embeddings 对于机器学习的输入非常重要。分类器（更普遍的包括神经网络）基于实数向量运行。它们在密集向量（这个向量中的所有值对对象的定义都有意义）上训练模型的最佳方式。然而，对于机器学习中很多重要的输入，比如文本中的文字，并没有自然的向量表示。Embedding 函数是将这些离散的输入对象转换为可用连续向量的业界标准做法。

Embeddings 作为机器学习的输出也同样很有价值。由于 embeddings 是将对象映射到向量上，应用程序可以使用向量空间中的相似性（例如，欧几里得度量或向量夹角）作为对象间相似度健壮而灵活的度量工具。一个常见的用途就是找出最近元素。例如，使用与上面相同的单词 embeddings，这里是每个单词的三个最近元素和相应的向量夹角：

```
blue:  (red, 47.6°), (yellow, 51.9°), (purple, 52.4°)
blues:  (jazz, 53.3°), (folk, 59.1°), (bluegrass, 60.6°)
orange:  (yellow, 53.5°), (colored, 58.0°), (bright, 59.9°)
oranges:  (apples, 45.3°), (lemons, 48.3°), (mangoes, 50.4°)
```

应用程序能从这些数据中推测出在某些方面苹果和橙子（相距 45.3°）比柠檬和橙子（相距 48.3°）更相似。

## 在 TensorFlow 中使用 embeddings

要在 TensorFlow 中创建 embeddings，我们首先将文本拆分成单词，然后为词汇表中的每个单词分配一个整数。让我假设这已经完成了，并且 `word_ids` 是这些整数的一个向量。例如，“I have a cat.” 这句话可以被拆分成 `[“I”, “have”, “a”, “cat”, “.”]`，然后相应的 `word_ids` tensor 会具备维度 `[5]` 并由 5 个整数组成，我们需要创建 embeddings 变量并像下面一样使用 `tf.nn.embedding_lookup` 函数：

```
word_embeddings = tf.get_variable(“word_embeddings”,
    [vocabulary_size, embedding_size])
embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, word_ids)
```

在此之后，在我们的例子中 tensor `embedded_word_ids` 会具备维度 `[5, embedding_size]`，其中就是这 5 个单词的每一个词的 embeddings（密集向量）。在训练结束时，`word_embeddings` 将包含词汇表中所有单词的 embeddings。

Embeddings 可以在多种神经网络中、使用各种损失函数和数据集来训练，例如，可以使用递归神经网络在给定大规模语料库中词句的基础上，根据前一单词来预测下一个单词，或者可以训练两个网络进行多语言翻译。这些方法在[词向量表示](../tutorials/representation/word2vec.md)教程中有详细描述。

## 将 Embeddings 可视化

TensorBoard 包含了 **Embedding Projector**，一个让你能够交互式查看 embeddings 的工具。这个工具可以从你的模型中读取 embeddings 并将它们渲染到二维或三维空间中。

Embedding Projector 有三个面板：

- Data panel 在顶部左侧，这里你可以选择运行，通过点击改变 embedding 变量，和数据列颜色或将他们打上标签。
- Projections panel 在底部左侧，在这里你可以选择展示的类型。
- Inspector panel 在右侧，这里你可以查询特殊的点并查看最近元素列表。

### Projections
Embedding Projector 为数据集降维提供了三种方法。

- **[t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)**：一个非线性的不确定算法，通常是以扭曲全局结构为代价试图保留数据中的局部邻域。你可以选择是否计算为二维或三维的展示。

- **[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)**：一个线性的确定算法（主成分分析），试图在尽可能少的维度上捕获尽可能多的数据可变性。PCA 倾向于从数据中发现大规模结构，但可能会扭曲局部邻域。Embedding Projector 计算前 10 个主要组件，你可以从中选择两个或三个来查看。

- **Custom**：使用数据中的标签，得到一条在你指定的水平和垂直轴上的线性映射。例如，你通过给定的文本模式“ Left ”与“ Right ”指定水平轴，Embedding Projector 找出所有被“ Left ”模式匹配到的点，并计算该组的质心；“ Right ”也是类似。通过这两点的直线定义为水平轴。对于“ UP ”和“ DOWN ”文本模式，垂直轴也是类似的计算出两个点集的质心得出。

获取其他有价值文章可以查看 [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) 和 [Principal Component Analysis Explained Visually](http://setosa.io/ev/principal-component-analysis/)。

### 探索

你可以通过单击和拖动自然地进行缩放、旋转、平移来直观的探索。将鼠标悬停在点上将显示该点的任何[元数据](#元数据)。你也可以检查最近元素的子集。点击一个点会在右窗格列出最近的元素，以及到当前点的距离。展示区域中也突出显示最近邻点。

有时候我们只需要观察点集的一部分，将视图聚焦在这一部分上是非常有帮助的。为此，你可以通过多种方式选择点：

- 点击一个点后，最近的元素也会被选择。
- 搜索后，选中匹配的的点集。
- 启用选择，单击一个点并拖动定义一个选择范围。

然后点击右侧检查器面板顶部的“ Isolate **nnn** points ”按钮，下图显示了 101 个被选中的点，并且" Isolate 101 points "可供用户点击。

![选中最近的相邻元素](https://www.tensorflow.org/images/embedding-nearest-points.png "Selection of nearest neighbors")

**在文字 embedding 数据集中选中" import "的最近相邻元素。**

提示：使用选中功能来自定义展示很强大。下面，我们选择了“ politics ”的最近的 100 个相邻元素并将他们映射到“ worst ”-“ best ”向量作为的 X 轴上。Y 轴是随机的。因此，在右边是“ ideas ”、“ science ”、“ perspective ”、“ journalism ”，而在左边则是” crisis “、“ violence ”和“ conflict ”。

<table width="100%;">
  <tr>
    <td style="width: 30%;">
      <img src="https://www.tensorflow.org/images/embedding-custom-controls.png" alt="Custom controls panel" title="Custom controls panel" />
    </td>
    <td style="width: 70%;">
      <img src="https://www.tensorflow.org/images/embedding-custom-projection.png" alt="Custom projection" title="Custom projection" />
    </td>
  </tr>
  <tr>
    <td style="width: 30%;">
      projection 自定义控制。
    </td>
    <td style="width: 70%;">
      "politics" 在 "best" - "worst" 向量上定制的 projection。
    </td>
  </tr>
</table>

如果你想要分享你的发现，可以使用右下角的书签面板，并将当前状态（包括任何展示栏中的计算坐标）保存为小文件。然后可以将  Projector 指向一组这样一个或多个的文件，制作下面的板块。其他用户就可以查看这一系列书签了解它们。

<img src="https://www.tensorflow.org/images/embedding-bookmark.png" alt="Bookmark panel" style="width:300px;">

### 元数据

如果你正在使用 embedding，则可能需要将坐标/图像附加到数据点上。你可以通过生成包含每个点标签的元数据文件，然后在 Embedding Projector 的数据面板中单击“Load data”来执行操作。

元数据可以是标签或图像，它们储存在单独的文件中。对于标签，格式应该是 [TSV file](https://en.wikipedia.org/wiki/Tab-separated_values)（表示为红色的制表符），其第一行包含列标题（以粗体显示），后续行包含元数据值。例如：

<code>
<b>Word<span style="color:#800;">\t</span>Frequency</b><br/>
  Airplane<span style="color:#800;">\t</span>345<br/>
  Car<span style="color:#800;">\t</span>241<br/>
  ...
</code>

 除标题外，元数据文件中行的顺序与 embedding 变量中的向量的顺序应该是相匹配的。因此，元数据文件中的第（i + 1）行对应于 embedding 变量的第 i 行。如果 TSV 元数据文件只有一个列，并假设每一行都是 embedding 的标签。这是一个特外，但它确实符合常用的“ vocab file ”格式。

要将图片用作元数据，你必须制作一个 [sprite image](https://www.google.com/webhp#q=what+is+a+sprite+image)，其中包含小缩略图，在 embedding 过程中一个该文件可供每个向量使用。它应该按行先存储缩略图：第一个数据点放在左上角，尽管最后一行不需要填充，最后一个数据点放在右下角，如下图所示。

<table style="border: none;">
<tr style="background-color: transparent;">
  <td style="border: 1px solid black">0</td>
  <td style="border: 1px solid black">1</td>
  <td style="border: 1px solid black">2</td>
</tr>
<tr style="background-color: transparent;">
  <td style="border: 1px solid black">3</td>
  <td style="border: 1px solid black">4</td>
  <td style="border: 1px solid black">5</td>
</tr>
<tr style="background-color: transparent;">
  <td style="border: 1px solid black">6</td>
  <td style="border: 1px solid black">7</td>
  <td style="border: 1px solid black"></td>
</tr>
</table>

打开[这个链接](https://www.tensorflow.org/images/embedding-mnist.mp4)查看一个有趣的 Embedding Projector 中图像缩略图案例。

## 小型问答会

**“embedding” 是一个操作还是一个对象？**
都是，人们讨论在一个矢量空间中 embedding 词语（操作）和生成单词 embedding（对象）。两者共同点在于 embedding 的概念都是将离散对象映射到矢量的过程。创建或者应用映射是一个操作，但映射本身是一个对象。

**embedding 是高维还是低维？**
这不一定。例如，与包括数百万个单词和短语的矢量空间相比，只包含 300 维的矢量空间通常只能被称为低维的（并且密集）。但在数学意义上，它是高维的，显示出许多与我们人类直觉了解的二维和三维空间截然不同的特性。

**embedding 和 embedding 层是一样的吗？**
不是，**embedding 层**是神经网络的一部分，而 **embedding** 则是一个普通的概念。
