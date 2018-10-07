# 单词的向量表示

本文中，我们来通过 [Mikolov 等人的论文](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) 来看看 word2vec 这个模型。此模型被用来训练单词的向量表示，此方法又称为『词嵌入』。

## 要点

本文主要目是要摘要出 TensorFlow 中构建一个 word2vec 模型时的那些有趣而关键性的内容。

* 首先，我们需要知道为什么要用向量来代表单词。
* 其次，我们需要看看模型背后的逻辑，以及它是如何训练的（使用一些数学技巧来获得好的训练效果）。
* 同时，我们要用 TensorFlow 实现一个简单的模型。
* 最后，我们会介绍能让原生版本更好地拓展的一些方法。

本文只会粗略地过一遍代码，如果你想深入了解的话，可以看看示例 [tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py) 中的一个极简的实现。这个简单示例包含下载数据的代码，只需稍微训练一下这些数据，就可以把结果可视化出来了。一旦你可以轻松的阅读并运行这个版本之后，你就可以去看一个更严谨的实现代码了： [models/tutorials/embedding/word2vec.py](https://github.com/tensorflow/models/tree/master/tutorials/embedding/word2vec.py) 。它展示了一些更高级的 TensorFlow 原则，你可从中学到如何高效的使用线程来将数据移动到一个文本模型中，如何在训练中设置检查点，等等。

首先，让我们来看一看为什么我们想要学习词嵌入。如果你已经是一个词嵌入专家了，可以跳过本节，直接深入细节。

## 动机：为什么要学习词嵌入？

图像和音频处理系统在处理丰富，高维度的数据集时会将图像的单个像素信号量或音频的功率谱密度系数编码成向量。对于图像识别或者语音识别这样的任务来说，我们知道要成功的识别出正确结果的所有信息都在数据里（因为人类可以从这些原始数据中得到答案）。然而，传统的自然语言处理系统将单词看做确定的原子符号，因此 `Id537` 会代表 'cat'，`Id143` 会代表 'dog'。这些编码都是任意的，并且它们对于单个字符之间的关系系统毫无帮助。这意味着当模型处理关于 'dogs' 数据的时候，基本上用不到它从处理 'cats' 上学到的知识（然而它们有很多共同点，都是动物，四条腿，宠物等）。将单词表示成唯一离散的 id，这样会使数据变的稀疏，通常情况下这意味着为了成功的训练统计模型，我们需要更多的数据。而使用向量代表单词就可以克服这些困难。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/audio-image-text.png" alt>
</div>

[Vector space models](https://en.wikipedia.org/wiki/Vector_space_model)（VSMs）是在一个连续的向量空间中代表单词，并且把语义上相似的单词映射到它附近的点（就是这些单词彼此都靠的很近）。VSMs 在 NLP 领域中历史悠久，但是它依赖的所有方法在某种程度上都是分享语义上的意思，或者声明相同上下文中出现的单词 [Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis)。而另一个不同的利用这个原则的方法是深入下面两个分类：*计数方法*（例如：[Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis))，和 *预测方法*（例如：
[neural probabilistic language models](http://www.scholarpedia.org/article/Neural_net_language_models))。[Baroni et al.](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf) 更详细的阐述了这种区别，但是一言以蔽之就是：计数方法会计算在一个大文本集中一些单词以及它们类似单词一起出现的频率，然后将每一个单词的这些统计数据降维到一个小而密的向量。预测模型则是根据学习到的小而密的*词向量*（模型深思熟虑的参数）来直接从它的邻近单词中预测出单词。对于从源数据中学习词向量来说，Word2vec 是一个计算效率很高的预测模型。这其中包含两层含义，Continuous Bag-of-Words（CBOW）和 Skip-Gram 模型（[Mikolov et al.](https://arxiv.org/pdf/1301.3781.pdf) 中的 3.1 和 3.2 章节）。从算法角度来说，这些模型是类似的，CBOW 是从源上下文数据（'the cat sits on the'）中预测出目标单词（比如 'mat'），skip-gram 则相反，它是从目标单词中预测出源上下文数据。这种反向操作可能看起来很随意，但是从统计的角度讲，它是有效的，因为 CBOW 可以消除很多分散的信息（通过将整个上下文看做一个观察点）。在大多数情况下，这种操作对于小一点的数据集也是适用的。然而，skip-gram 将每一个上下文目标对都看做一个新的观察点，这样对于更大的数据集是更有利的。在下面的篇幅中，我们将主要来看 skip-gram 这个模型。

## 放大对比噪音训练

神经概率语言模型传统的使用 [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood) (ML) 定理来最大化下一个单词的的概率，根据 [*softmax* function](https://en.wikipedia.org/wiki/Softmax_function) 这个函数使用之前的单词 \\(h\\) (代表 "history") 来推出下一个单词 \\(w_t\\) (代表 "target")。

$$
\begin{align}
P(w_t | h) &= \text{softmax}(\text{score}(w_t, h)) \\
           &= \frac{\exp \{ \text{score}(w_t, h) \} }
             {\sum_\text{Word w' in Vocab} \exp \{ \text{score}(w', h) \} }
\end{align}
$$

在这里 \\(\text{score}(w_t, h)\\) 计算出了在上下文为 \\(h\\)（通常使用点积）的情况下单词 \\(w_t\\) 的兼容性。我们通过在训练集上最大化它的 [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function) 来训练这个模型，比如通过最大化

$$
\begin{align}
 J_\text{ML} &= \log P(w_t | h) \\
  &= \text{score}(w_t, h) -
     \log \left( \sum_\text{Word w' in Vocab} \exp \{ \text{score}(w', h) \} \right).
\end{align}
$$

这样会为语言建模产生一个正确标准化的概率模型。然而这样做的代价也是昂贵的，因为我们需要使用当前上下文 \\(h\\) 中所有其他的 \\(V\\) 单词 \\(w'\\) 的得分来对*每一步的训练*都计算和标准化每一个概率。

<div style="width:60%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/softmax-nplm.png" alt>
</div>

另一方面，对于 word2vec 的特征学习来说，我们不需要一个完整的概率模型。在同样的上下文中，CBOW 和 skip-gram 模型采用了二元分类 ([logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)) 来从 \\(k\\) 以及虚假的（噪音）单词 \\(\tilde w\\) 中区别出真正的目标单词 \\(w_t\\)。我们会在下面使用 CBOW 模型来示范。对于 skip-gram 模型来说，仅仅改变成相反的方向就可以了。

<div style="width:60%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/nce-nplm.png" alt>
</div>

从数学的角度讲，每一个例子的目标都是去最大化

$$
J_\text{NEG} = \log Q_\theta(D=1 |w_t, h) +
  k \mathop{\mathbb{E}}_{\tilde w \sim P_\text{noise}}
     \left[ \log Q_\theta(D = 0 |\tilde w, h) \right]
$$

这里，\\(Q_\theta(D=1 | w, h)\\) 表示在数据集 \\(D\\) 的上下文 \\(h\\) 中有单词 \\(w\\) 的模型的二元逻辑回归概率，它是根据学习之后的词向量 \\(\theta\\) 来计算的。实际上我们会通过在噪音分布中画出对比的单词来近似的得到期望（比如我们可以计算一个[Monte Carlo average](https://en.wikipedia.org/wiki/Monte_Carlo_integration))。

这样做的目的就是最大化这个期望，模型会给真正的单词分配一个高的概率，给噪音单词分配一个低的概率。从学术角度说，它被称作 [Negative Sampling](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)，并且这里对使用这个损失函数还有良好的数学激励：因为这个操作的更新在极限情况下会近似于 softmax 函数的更新。但是从计算角度讲，这是非常诱人的，因为现在计算损失函数的规模相当于我们选择的 (\\(k\\)) 的*噪音单词*的数量，并且不是所有的在词汇表 (\\(V\\)) 中的单词。这样就会使得训练更快速。实际上我们会使用 TensorFlow 的一个辅助函数 `tf.nn.nce_loss()` 来获得类似的损失 [noise-contrastive estimation (NCE)](https://papers.nips.cc/paper/5165-learning-word-embeddings-efficiently-with-noise-contrastive-estimation.pdf)。

让我们从直观上来感受一下在实践中是如何工作的！

## Skip-gram 模型

举个例子，让我们考虑一下下面的数据集

`the quick brown fox jumped over the lazy dog`

首先让我们构建一个单词数据集以及这些单词所在的上下文环境。我们可以以任何合理的方式定义这个 'context'，而且实际上人们查看语法的上下文（比如依赖当前目标单词的语法，可以查看 [Levy et al.](https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf)），以及目标左边的单词，目标右边的单词等。现在我们坚持使用单纯的定义并把目标左边的单词和目标右边的单词作为窗口定义为 'context'。使用一个大小为 1 的窗口，我们就会得到下面的数据集：

`([the, brown], quick), ([quick, fox], brown), ([brown, jumped], fox), ...`

即 `(context, target)` 这样的数据对。回想一下，skip-gram 会反转上下文和目标，并且从目标单词中试着预测每一个上下文的单词，所以我们的任务就变成了从 'quick' 预测 'the' 和 'brown'，从 'brown' 预测 'quick' 和 'fox' 等。因此我们的数据集就变成了下面这样：

`(quick, the), (quick, brown), (brown, quick), (brown, fox), ...`

即 `(input, output)` 这样的数据对。这个目标函数是基于整个数据集定义的，但是我们可以使用 [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)(SGD) 来优化这个目标函数，每次使用一个样本（或者在 `16 <= batch_size <= 512` 这个条件下，使用 `batch_size` 中的一个 'minibatch'）。现在就让我们来看看这个过程的一个步骤吧。

让我们注意观察上面第一个训练的情况中的训练步骤 \\(t\\)，就是目标是从 `quick` 预测出 `the` 的训练。我们选择了 `num_noise` 个噪音（对比的）样本，这些样本是通过均匀分布 \\(P(w)\\) 的噪音分布中拉取出来的。简单来说就是我们设置 `num_noise=1`，并且选择 `sheep` 作为噪音样本。下一步我们就可以计算这个观察样本和噪音样本对的损失函数了。例如，此时我们的目标就是：

$$
J^{(t)}_\text{NEG} = \log Q_\theta(D=1 | \text{the, quick}) +
  \log(Q_\theta(D=0 | \text{sheep, quick}))
$$

这个目标就是通过更新嵌入的参数 \\(\theta\\) 来提升（这里就是最大化）这个结果。为此，我们需要在考虑到嵌入参数 \\(\theta\\) 的情况下得到梯度的损失值，比如 \\(\frac{\partial}{\partial \theta} J_\text{NEG}\\)（幸运的是，TensorFlow 提供了非常简单的辅助函数来帮助我们完成计算！）。然后我们再执行一次更新，向梯度方向移动一小步。当整个训练集都完成这个过程之后，直到模型可以成功的从噪音单词中分辨出真实单词时，这个模型才会对每一个单词周围『绕来绕去』的词向量有影响。

我们可以先使用一些像 [t-SNE dimensionality reduction technique](https://lvdmaaten.github.io/tsne/) 的实例来将学习之后的向量降到 2 维，然后我们就可以对它们进行可视化了。当我观察这些可视化的数据时，这些向量所捕获的通用的，并且非常有效的关于单词和与单词之间关系的语义信息就变的显而易见了。当我们第一次发现在诱发的向量中某个特定方向会指向一个特定的语义关系时，简直是太有趣了，比如  *male-female*， *verb tense*， 甚至 *country-capital* 这些单词之间的关系，就像下图展示的那样（也可以去查看 [Mikolov et al., 2013](https://www.aclweb.org/anthology/N13-1090) 上的例子）。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/linear-relationships.png" alt>
</div>

这就解释了为什么对于很多典型的 NLP 预测问题来说这些作为特征的向量都是非常有用的，比如词性标记，指定物体的识别（可以查看 [Collobert et al., 2011](https://arxiv.org/abs/1103.0398)
([pdf](https://arxiv.org/pdf/1103.0398.pdf)) 的原始工作,或者 [Turian et al., 2010](https://www.aclweb.org/anthology/P10-1040) 的追踪任务）。

现在让我们通过画漂亮的图片来使用它们吧！

## 创建 Graph

这就是嵌入的全部内容，接下来让我们来定义嵌入矩阵吧。开始这就是一个大型的随机矩阵。然后我们会给每个单元格初始化一个均匀分布的值。


```python
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
```

对比噪音的估计损失是根据逻辑回归模型来定义的。这里我们需要为词汇表中的每一个单词定义权重和 biases（也叫作 `input embeddings` 的对立 `output weights`）。好，让我们来定义它吧。

```python
nce_weights = tf.Variable(
  tf.truncated_normal([vocabulary_size, embedding_size],
                      stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
```

现在我们有了参数，然后就可以定义 skip-gram 模型的 graph 了。简便起见，假设我们已经用一个词汇表对我们的文本库整数化了，这样的话每一个单词都是用一个整数来代替的（更多详细信息请查看 [tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)）。skip-gram 需要两个输入。一个是代表原上下文单词的全部整数，另一个是目标单词。让我们为这些输入创建一些占位符节点，这样我们就可以在之后供给数据了。

```python
# Placeholders for inputs
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
```

现在我们需要做的就是分批查询每一个源单词的向量。使用 TensorFlow 的辅助函数非常简单。

```python
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
```

好的，现在我们已经拥有了每一个的单词的词向量，让我们使用噪音对比的训练目标来试着预测一下目标单词。

```python
# Compute the NCE loss, using a sample of the negative labels each time.
loss = tf.reduce_mean(
  tf.nn.nce_loss(weights=nce_weights,
                 biases=nce_biases,
                 labels=train_labels,
                 inputs=embed,
                 num_sampled=num_sampled,
                 num_classes=vocabulary_size))
```

现在我们得到了一个损失节点，我们需要将必要的节点加入到计算梯度和更新参数中去。因此我们使用随机提督下降法，同样，使用 TensorFlow 的辅助函数就好了。

```python
# We use the SGD optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)
```

## 训练模型

训练模型现在就非常简单了，只需要使用一个 `feed_dict` 将数据推送到 placeholders 中，并且在一个循环中使用这个新的数据来调用 `tf.Session.run` 就可以了。

```python
for inputs, labels in generate_batch(...):
  feed_dict = {train_inputs: inputs, train_labels: labels}
  _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)
```

完整实例代码请查看 [tensorflow/examples/tutorials/word2vec/word2vec_basic.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)。

## 可视化训练之后的词向量

训练完成之后，我们可以使用 t-SNE 来
可视化训练之后的词向量。

<div style="width:100%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/tsne.png" alt>
</div>

完美！和我们想的一样，同样结尾的单词互相之间离的很近。还有一个更加重量级的 word2vec 的实现，它会展现更多 TensorFlow 的高级特性，请查看 [models/tutorials/embedding/word2vec.py](https://github.com/tensorflow/models/tree/master/tutorials/embedding/word2vec.py)。

## 评估词向量：Analogical Reasoning

词向量在各种各样的 NLP 预测任务中都非常有用。除了训练一个完善的词性标记模型或者实体命名模型，一个简单的评估词向量的方法是直接使用它们来预测语法和语义关系，比如这个句子 `king is to queen as father is to ?`。这就是 *analogical reasoning*，这个任务的介绍在 [Mikolov and colleagues](https://www.aclweb.org/anthology/N13-1090)。你可以从 [download.tensorflow.org](http://download.tensorflow.org/data/questions-words.txt) 下载到这个任务的数据集。

想看如何实现这个评估过程，请在 [models/tutorials/embedding/word2vec.py](https://github.com/tensorflow/models/tree/master/tutorials/embedding/word2vec.py) 查看 `build_eval_graph()` 和 `eval()` 这两个函数。

超参数的选择会显著的影响到任务的准确性。为了达到这个任务当前水平的的表现，我们需要使用一个很大的数据集来训练，并且精细的调优这些超参数以及利用像二次抽样这样的小技巧，不过二次抽样不在我们的教程范围内。

## 优化实现

我们朴实无华的实现展现了 TensorFlow 的灵活性。例如，如果我们想改变训练的目标，仅仅通过调用 `tf.nn.nce_loss()` 就可以使用一个现成的像 `tf.nn.sampled_softmax_loss()` 这样的函数。如果你对损失函数有一些新的想法，那么你可以在 TensorFlow 中手动写一个新的目标函数，并且让优化器计算它的导数。在机器学习模型开发的探索阶段灵活性是非常重要的，因为这样我们才可以快速迭代以及实验不同的想法。

一旦你有了一个满意的模型结构，优化你的实现方法来让让模型更有效率（以及在更少的时间内囊括更多的数据）就会变得很有必要。举个例子，在这篇教程中使用的本地代码会非常拖后腿，因为我们使用 Python 来读取和供给数据条目 -- 在 TensorFlow 这里每一条数据都只需要很少的工作量。如果你发现你的模型在输入数据方面有很大的性能瓶颈，那么你就需要为你的问题实现一个自定义的数据读取器，就像[新的数据格式](../../extend/new_data_formats.md)中说的那样。对于 Skip-Gram 来说，我们已经为了做了这些，请查看 [models/tutorials/embedding/word2vec.py](https://github.com/tensorflow/models/tree/master/tutorials/embedding/word2vec.py)。

如果你的模型已经没有 I/O 问题，但是你仍然想得到更好的性能，那么你可以编写你自己的 TensorFlow 操作，就像[添加一个新 Op](../../extend/adding_an_op.md) 中描述的那样。重申一遍，我们已经提供了 Skip-Gram 模型的完整实例 [models/tutorials/embedding/word2vec_optimized.py](https://github.com/tensorflow/models/tree/master/tutorials/embedding/word2vec_optimized.py)。可以随意的和其他模型比较来测量每一阶段性能的提升。

## 总结

本篇教程中，我们讲解了 word2vec 模型，一个学习词向量计算高效的模型。我们解释了为什么词向量是有用的，讨论了高效训练的技术，同时也展示了如何在 TensorFlow 中实现这些功能。总的来说，我们希望已经展示了 TensorFlow 是如何给你们提供早期的实验阶段你们需要的灵活性，以及后续你们对定制化优化实现的管理需求。
