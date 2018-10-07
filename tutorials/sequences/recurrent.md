# 循环神经网络

## 介绍

[理解 LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/){:.external} 详细介绍了循环神经网络(RNN)以及 LSTM 的介绍。

## 语言模型

此教程将展示如何在高难度的语言模型中训练循环神经网络。该问题的目标是获得一个能确定语句概率的概率模型。它是通过之前已经给出的词语来预测后面的词语。我们将使用 [PTB(Penn Tree Bank)](https://catalog.ldc.upenn.edu/ldc99t42) 数据集；同时因为它数据量比较小，所以训练起来也相对快速。

语言模型是很多有趣难题的关键所在，比如语音识别，机器翻译，图像字幕等。更多的相关资料可以查看[这里](https://karpathy.github.io/2015/05/21/rnn-effectiveness)。

本教程的目的是重现 [Zaremba et al., 2014](https://arxiv.org/abs/1409.2329) ([pdf](https://arxiv.org/pdf/1409.2329.pdf)) 的结果，它在 PTB 数据集上得到了很棒的质量。

## 教程文件

这篇教程使用的文件在[我们的仓库](https://github.com/tensorflow/models)中可以找到，路径是 `models/tutorials/rnn/ptb`。

文件 | 作用
--- | ---
`ptb_word_lm.py` | 用于在 PTB 数据集上训练语言模型的代码。
`reader.py` | 用于读取数据集的代码。

## 下载和准备数据

本教程需要的数据在 data/ 路径下，来源于 [Tomas Mikolov 网站上的 PTB 数据集](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)

该数据集已经经过预处理，总共包含了 10000 个不同的词语，其中包括语句结束标记符，以及标记稀有词语的特殊符号 (<unk>) 。我们在 `reader.py` 中转换所有的词语，让他们各自有唯一的整型标识符，便于神经网络处理。

## 模型

### LSTM

模型的核心由一个 LSTM 单元组成。它每次处理一个词语，并计算当前语句中下一个可能词出现的概率。网络的储存状态在刚初始化时是一个全零向量初始化，随后每读取一个词语后会进行更新。而且，出于计算的考虑，我们将以 `batch_size` 为最小批量来处理数据。在这个例子中，有一个很重要的点是 `current_batch_of_words` 和一个「句子」的词语不是对应关系。每一个在 batch 里面的词语只与时间 t 对应。TensorFlow 会自动地帮你累加每个 batch 的梯度值。

示例：
```
 t=0  t=1    t=2  t=3     t=4
[The, brown, fox, is,     quick]
[The, red,   fox, jumped, high]

words_in_dataset[0] = [The, The]
words_in_dataset[1] = [brown, red]
words_in_dataset[2] = [fox, fox]
words_in_dataset[3] = [is, jumped]
words_in_dataset[4] = [quick, high]
batch_size = 2, time_steps = 5
```

基本的伪代码如下：

```python
words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# 初始化 LSTM 存储状态。
state = lstm.zero_state(batch_size, dtype=tf.float32)
probabilities = []
loss = 0.0
for current_batch_of_words in words_in_dataset:
    # 每次处理一批词语后更新状态值。
    output, state = lstm(current_batch_of_words, state)

    # LSTM 输出可用于产生下一个词语的预测。
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)
```

### 截断反向传播

RNN 的输出结果依赖于不定长度的输入，这是它网络的特点所决定的。不幸的是，这让反向传播的计算变得很困难。为了能够学习流程易于处理，通过的做法是创建一个「unrolled」版本的网络，这个网络包含了固定数量（`num_steps`）的 LSTM 输入和输出。这样模型就可以使用近似 RNN 的方式来训练。这可以每次通过提供长度为 `num_steps` 的输入，和每次迭代完成之后进行反向传导来实现它。

以下是用于创建截断反向传播图的简单代码：

```python
# 一次给定的迭代中的输入占位符。
words = tf.placeholder(tf.int32, [batch_size, num_steps])

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
# 初始化 LSTM 存储状态。
initial_state = state = lstm.zero_state(batch_size, dtype=tf.float32)

for i in range(num_steps):
    # 每处理一批词语后更新状态值。
    output, state = lstm(words[:, i], state)

    # 其余的代码。
    # ...

final_state = state
```

下面展现了如何在整个数据集实现迭代：

```python
# 一个 numpy 数组，保存每一批词语之后的 LSTM 状态。
numpy_state = initial_state.eval()
total_loss = 0.0
for current_batch_of_words in words_in_dataset:
    numpy_state, current_loss = session.run([final_state, loss],
        # 根据上一次迭代结果初始化 LSTM 状态。
        feed_dict={initial_state: numpy_state, words: current_batch_of_words})
    total_loss += current_loss
```

### 输入

在输入 LSTM 前，词语 ID 被嵌入到了一个密集的表示中（查看[特征表示教程](../../tutorials/representation/word2vec.md)）。这种方式允许模型高效地表示词语，也便于写代码：

```python
# embedding_matrix 是一个形状张量 [vocabulary_size, embedding size]
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)
```

嵌入的矩阵会被随机的初始化，模型会通过数据学会分辨不同词语的意思。

### 损失函数

我们想使目标词语的平均负对数概率最小

$$ \text{loss} = -\frac{1}{N}\sum_{i=1}^{N} \ln p_{\text{target}_i} $$

我们可以直接使用现成的 `sequence_loss_by_example` 函数，虽然这实现起来并不难。论文中的典型衡量标准是每个词语的平均困惑度（perplexity），计算式为

$$e^{-\frac{1}{N}\sum_{i=1}^{N} \ln p_{\text{target}_i}} = e^{\text{loss}} $$

同时我们会观察训练过程中的困惑度值（perplexity）。

### 多个 LSTM 层堆叠

要想给模型更强的表达能力，可以添加多层 LSTM 来处理数据。第一层的输出作为第二层的输入，以此类推。类 MultiRNNCell 可以无缝地将其实现：

```python
def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(lstm_size)
stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell() for _ in range(number_of_layers)])

initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
for i in range(num_steps):
    # 每次处理一批词语后更新状态值。
    output, state = stacked_lstm(words[:, i], state)

    # 其余的代码。
    # ...

final_state = state
```

## 运行代码

在运行代码之前，如教程刚开始所述那样下载 PTB 数据集。然后，在 home 目录下解压 PTB 数据集。

```bsh
tar xvfz simple-examples.tgz -C $HOME
```
**(注意：在 windows 下，你可能需要其他的[工具](https://wiki.haskell.org/How_to_unpack_a_tar_file_in_Windows).)**

现在，从 [TensorFlow 模型仓库](https://github.com/tensorflow/models)中克隆一份代码后，执行下面命令：

```bsh
cd models/tutorials/rnn/ptb
python ptb_word_lm.py --data_path=$HOME/simple-examples/data/ --model=small
```

这里有 3 个支持的模型配置文件在我们教程的代码里：「small」，「medium」和「large」。它们的区别在于 LSTM 的大小，以及用于训练的超参数集。模型越大，得到的结果应该更好。在测试集中 `small` 模型应该可以达到低于 120 的困惑度（perplexity），`large` 模型则是低于 80，但它可能花费数小时来训练。

## 除此之外

还有几个优化模型的技巧没有提到，包括：

- 降低学习率计划表。
- 在 LSTM 层间使用 dropout。

继续学习和更改代码以进一步改善模型吧。
