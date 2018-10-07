# 简易语音识别

这个教程会教你建立一个能够识别出十个不同词语基础语音的神经网络。但这个模型远不如实际场景中的语音识别系统那么复杂，这就好比 MNIST 数据集无法用于训练实际场景中的图像识别。所以这个教程只会带你了解其中涉及的技术方法，当你完成这个教程时，你将得到一个语音识别模型。它能够识别一个语音片段，将其归类为静默，或者下列词语中的一个："yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go" 。你也安卓应用中运行这个模型。

## 准备

首先，你要安装好 TensorFlow。我们会用一个脚本下载大小超过 1 GB 的训练数据，所以你的设备要有良好的网络连接和充足的储存空间。由于这个训练过程会花费数个小时，所以确保你的机器在这段时间内能保持正常运行。

## 训练

进入 TensorFlow 目录，运行如下命令来开始训练过程：

```bash
python tensorflow/examples/speech_commands/train.py
```

这个代码会先下载 [Speech Commands dataset](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)，这个数据集包含了 105,000 个 WAVE 声音文件，内容是人们说三十个不同词语的录音。这个数据集由谷歌收集并遵循知识共享协议发布，你能够通过[贡献你的五分钟语音项目](https://aiyprojects.withgoogle.com/open_speech_recording)来改进这个数据集。这个文件大小超过 2 GB，所以下载过程会花费一些时间，但为了防止出错，你也要不时查看一下运行日志。一旦下载完成，之后就不需要再做这一步了。你可以在 [Speech Commands paper](https://arxiv.org/abs/1804.03209) 中找到有关此数据集的更多信息。

下载完成后，你会看到如下的日志信息：

```
I0730 16:53:44.766740   55030 train.py:176] Training from step: 1
I0730 16:53:47.289078   55030 train.py:217] Step #1: rate 0.001000, accuracy 7.0%, cross entropy 2.611571
```

这表明初始化过程已经完成，训练的循环过程已经开始。你能看到每一步它都会输出日志信息。下面详细讲解日志中数据的含义：

`Step #1` 告诉我们这是在训练循环第一轮。在这个例子中共有 18,000 轮，所以你能够通过查看步骤数知道还有多久训练过程会结束。

`rate 0.001000` 是控制网络权重变更速度的学习速率。在初始阶段这个值会是个相当高的数字（0.001），但在后面的训练中它会减小十倍到 0.0001。

`accuracy 7.0%` 代表当前训练过程预测对了多少。这个值会经常的上下波动，但是它的平均值会随着训练不断增长。这个模型会输出一个数组，每一个代表了一个类别，每个数代表了输入和那个类别之间相似度的预测值。最后得到的预测标签是获得最高分数的类别。这个分数通常在 0 和 1 之间，并且数值越大代表模型对这个结果越有信心。

`croos entropy 2.611571` 是损失函数返回的结果，损失函数用来指导训练过程。这个分数通过比较当前训练得出的分数向量与正确标签得出的，它会随着训练的进行而逐渐减小。

一百轮之后，你会看到这样一行输出信息：

```
I0730 16:54:41.813438 55030 train.py:252] Saving to
"/tmp/speech_commands_train/conv.ckpt-100"
```

这表示现在把当前训练权重保存到了一个记录文件，如果你的训练脚本运行中断了，你能够找到最近的一个保存点并把它作为一个参数重启训练脚本。 `--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-100`  这样脚本就会从保存点开始训练过程。

## 混淆矩阵

在四百轮之后，这日志信息会被输出如下信息：

```
I0730 16:57:38.073667   55030 train.py:243] Confusion Matrix:
 [[258   0   0   0   0   0   0   0   0   0   0   0]
 [  7   6  26  94   7  49   1  15  40   2   0  11]
 [ 10   1 107  80  13  22   0  13  10   1   0   4]
 [  1   3  16 163   6  48   0   5  10   1   0  17]
 [ 15   1  17 114  55  13   0   9  22   5   0   9]
 [  1   1   6  97   3  87   1  12  46   0   0  10]
 [  8   6  86  84  13  24   1   9   9   1   0   6]
 [  9   3  32 112   9  26   1  36  19   0   0   9]
 [  8   2  12  94   9  52   0   6  72   0   0   2]
 [ 16   1  39  74  29  42   0   6  37   9   0   3]
 [ 15   6  17  71  50  37   0   6  32   2   1   9]
 [ 11   1   6 151   5  42   0   8  16   0   0  20]]
```

第一部分是一个[混淆矩阵](https://www.tensorflow.org/api_docs/python/tf/confusion_matrix)。要搞懂它，你先要知道当前训练中使用了哪些标签，这个教程中是 "_silence_"， "_unknown_"， "yes"， "no"， "up"， "down"， "left", "right"， "on"， "off"，"stop"，和 "go"。 每一列代表了一组被预测为这个标签的片段。所以第一列表示了所有被预测为静默的片段，第二列就是那些被预测为“未知词语”的片段，第三列是"yes"，以此类推。

每一行代表了这些标签正确的标签，第一行是所有静默的片段，第二段是“未知词语”的片段，第三行是 "yes"，依次类推。

这个矩阵比准确率这一个数字要有用的多，因为它能够很好展示了这个网络在预测中犯下的错误的，在这个例子中你能够看到第一行中除了第一个其他所有的条目都是 0。因为第一行是所有真正静默的片段，这就表示它们中没有被错误标签为单词的，所以静默中没有错误的负项。这表明这个网络已经能够很好的区分静默和词语。

如果从第一列向下看，我们能看到许多的非零值。由于这一列代表了所有被预测为静默，所以除了第一个单元格，其他行中的数值都是错误。这表示有些包含词语的片段被预测为静默的，所以这里面有不少的错误预测。

一个完美的模型会产出一个除了对角线其他部分的数值都是零的混淆矩阵。找出和这种矩形模式的偏移能帮助你找出这个模型在哪里最容易混淆，并且一旦你能辨别出问题后，就能通过增加更多的数据或者清除一些类型来找到问题所在。

## 验证模型

在混淆矩阵后，你能够看到这样一行：

`I0730 16:57:38.073777 55030 train.py:245] Step 400: Validation accuracy = 26.3%
(N=3093)`

将数据集分为三进行是一种很好的训练模式。最多的一块（在这个例子中大约是 80% 的数据）被用来训练网络，一个小块（这里是 10%，通常被称做“验证数据”)被用来在训练过程中评估模型的准确性，还有一个块（最后的 10%，“测试数据”）被用来在训练完成后测试模型的准确性。

这样分块的原因是在训练过程中网络总会有模型过度适配输入信息的威胁。通过分离出验证数据集，能够确保模型处理着它没见过的数据。测试数据集则是又一道额外的保障，它确保了模型不是只能准确预测训练数据集和验证数据集中的数据，而是有很大数值范围的不同数据。

这个训练脚本会自动的将数据集分离为这三块，上面这个输出日志就是模型运行在验证数据集中的准确率。理论上，这个值应当相当接近训练的准确率。如果训练数据集的准确率增加而验证数据集的没有，那么这表明模型可能过拟合了，它只学习了训练数据中的数据，而没有普适性。

## Tensorboard

使用 Tensorboard 能够很好的显示训练过程到底是怎样进行的。脚本会默认的把事件信息存储在/tmp/retrain_logs,你能通过以下命令加载：

`tensorboard --logdir /tmp/retrain_logs`

然后在浏览器中访问 [http://localhost:6006](http://localhost:6006) ，你就可以看到展现你模型运行过程的表格和图形。

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/speech_commands_tensorflow.png"/>
</div>

## 训练完成

经过几个小时（由你机器的运行速度决定）训练，脚本能够全部完成。它会打印出一个对测试数据集预测产生的的混淆矩阵以及准确率。默认设置下，你应该得到一个在 85% 到 90% 之间的准确率。

因为语音识别在移动设备上有很大用处，所以下一步我们会把它输出为一个能够在移动平台上流畅运行的简洁版。为了做到这点，要运行如下命令：

```
python tensorflow/examples/speech_commands/freeze.py \
--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-18000 \
--output_file=/tmp/my_frozen_graph.pb
```

当这个“凝固”的模型被创建后，你能通过`label_wav.py` 脚本测试，如下：

```
python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav
```

这会打印出三个标签：

```
left (score = 0.81477)
right (score = 0.14139)
_unknown_ (score = 0.03808)
```

我们希望"left" 得到最高的分值，因为这是正确的标签，但是由于训练是随机的，模型可能不能准确预测你尝试的第一个文件。你可以继续测试同一目录下的其他 .wav 文件来看看它的准确率。

得出的分数在 0 和 1 之间，越高的分数代表模型对它的预测结果越有自信。

## 在安卓应用中运行模型

要查看这个模型在一个真实的应用中怎样运行，最便捷的方法是下载[预构建的安卓应用演示](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#prebuilt-components)，并把它们装在你的手机上。在你的应用列表中你会看到 'TF Speech' ，打开它，你能看到和刚才训练模型时一样的词语列表，它以 “Yes" 和 "No"开始，给这个应用使用麦克风的权限，尝试说这些词语，模型就能在识别出它们时，在界面上高亮显示识别出的词语。

你也能自己构建这个应用，因为它是开源的，请查看 [github 上 TensorFlow 仓库中的可用部分](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#building-in-android-studio-using-the-tensorflow-aar-from-jcenter)。它默认会下载[ 一个来自 tensorflow.org 预训练模型 ](http://download.tensorflow.org/models/speech_commands_v0.01.zip)，它能够很简单的[替换成你自己训练好的模型](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install-model-files-optional)。如果你要这样做，你要确保在主要的[语音激活率 JAVA 源文件](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android/src/org/tensorflow/demo/SpeechActivity.java)中的常量和你在训练中改变的默认值保持一致，例如 `SAMPLE_RATE` 和 `SAMPLE_DURATION`。你也会看到有个[ JAVA 版本的识别命令行](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android/src/org/tensorflow/demo/RecognizeCommands.java)。它和 C++ 版本教程里的很像，如果你能很好的给那个版本调参，你也能在语音激活率中更换这些参数，并在服务测试中得到一样的结果

这个演示应用会根据你复制进在你的冻结图形旁的资源表的标签文本文件更新它界面中的结果列表，这代表你能不必要改变代码就能很轻松的尝试不同的模型。如果你改变了文件路径，你会需要更新 `LABEL_FILENAME` 和 `MODEL_FILENAME` 来指向你添加的文件。

## 这个模型是怎么工作的？

这个教程中用到的架构是基于一些来自 [Convolutional Neural Networks for Small-footprint Keyword Spotting](http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf)。它被选中不是因为最先进，而是因为它相当的简单，容易被训练和易于理解。有很多不同的建立处理音频的神经网络的方法，包括 [recurrent networks](https://svds.com/tensorflow-rnn-tutorial/) 或 [dilated (atrous) convolutions](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)。这个教程基于卷积神经网络，做图像识别的工程师会很熟悉它。一开始。这可能会让人感到惊讶，毕竟音频本质是一种时间上连续的一维信号，而不是二维的空间问题。

我们通过来定义一个时间段方法来解决这个问题，我们相信我们的语音能够分离为多个时间段，而且这个在时间段的音频信号可以转换为一个图像。这是通过将输入的音频样本分割成几毫秒的短片段，并计算一组频段的频率强度来完成的。来自同一段的每一组频率强度组成一个数组向量，这些向量按照时间顺序排列形成一个二维数组。这个数组的数值就能被当做一个单道影像，也被叫做[声谱图](https://en.wikipedia.org/wiki/Spectrogram)。如果你想查看一个声音样本生成了一个怎样的图像，你可以运行 `wav_to_spectrogram` 工具

```
bazel run tensorflow/examples/wav_to_spectrogram:wav_to_spectrogram -- \
--input_wav=/tmp/speech_dataset/happy/ab00c4b2_nohash_0.wav \
--output_image=/tmp/spectrogram.png
```

如果你打开 `/tmp/spectrogram.png` 你能够看到这样的图像：

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/spectrogram.png"/>
</div>

因为 TensorFlow 的存储次序，这个图像的时间方向是从上往下的，频率方向是从左往右的，而不像通常的声谱图中，时间方向是从左往右的。你应该能够看出图像中一些不同的部分，比如第一个音节 "Ha" 与 "ppy" 明显不同。

因为人类对一些频率特别敏感，在语音识别中，传统的做法是对这种表示进行进一步的处理以将其转换为一组[梅尔频率倒谱系数](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum),或者简称为 MFCCS。这也是一个二维的单通道表示法，所以也能被当做一个图像。如果你基本上将声音当做目标而不是语音，你会发现你能够跳过这一步直接操作声谱图。

这个由这些进行中步骤生成的图像然后会被送入一个多层卷积神经网络，这个网络有一个随后会经过 softmax 函数归一化。你能够在[tensorflow/examples/speech_commands/models.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands/models.py)查看这一部分。

## 数据流的正确率

很多音频识别应用需要处理一个连续的语音流，而不是单个剪辑。在这种环境下使用此模型的传统方法是在不同偏移时间里重复运行它，然后在一个很短的时间里产生一个平滑的预测结果。如果你把输入当做是一个图像，它沿着时间轴连续滚动。我们想要识别出来的词语可能会在任何时间开始，所以我们需要提取一系列的快照，来尝试得到一个队列，这个队列能够捕捉到我们输入数据中大部分的话语表达。一个足够高的取样率能产生更多时间段，使我们更易于捕捉到词语，所以平均结果能够提升预测的整体预测准确率。

给你一个怎样在流数据上运行语音识别模型的例子，查看 [test_streaming_accuracy.cc](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands/)。这个可以使用 [识别命令行](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands/recognize_commands.h)类处理一个长形态的语音输入，尝试识别词语，并将预测结果与真实的时间标签列表进行比较。这是一个制作处理音频信号流语音识别模型的好例子。

你需要一个很长的音频文件来测试它，它需要标记出说出词语的时间位置信息。如果你不想自己记录一个，你能用 `generate_streaming_test_wav` 模块生成一些合成的测试数据。默认的它会生成一个十分钟大约平均三秒一个词语的 .wav 文件和一个记录了每个词语说出时间的文本文件。这些词语是从你现有数据集的测试部分中提取后混入背景噪音。要得到它，请使用：

```
bazel run tensorflow/examples/speech_commands:generate_streaming_test_wav
```

这个命令将一个 .wav 文件存储到 `/tmp/speech_commands_train/streaming_test.wav`，和一个罗列所有标签的文本文件存到 `/tmp/speech_commands_train/streaming_test_labels.txt`。然后你能够使用以下命令进行准确率测试：

```
bazel run tensorflow/examples/speech_commands:test_streaming_accuracy -- \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_commands_train/streaming_test.wav \
--ground_truth=/tmp/speech_commands_train/streaming_test_labels.txt \
--verbose
```

这会输出正确匹配的词语的数量，被错误标识的数量，和没有词语时模型被触发的次数。有各种参数来控制分割信号的作业，包括设置用来均分片段时间长度的 `--average_window_ms`，设置模型中应用间的时间间隔的 `--clip_stride_ms`，设置当一个词语被发现后触发停止后续词语检测时间的 `--suppression_ms`，和一个可以认为是可信结果的最低平均分值 `--detection_threshold`。

流据数准确率测试会输出三个数字，而不像在前面模型训练中只有一个值。这是因为在不同的应用中会有不一样的要求，有些能容忍频繁的错误，只求找到正确结果（高查全），而另外一些在即使有些没有被检测到也要保证预测标签大概率为正确的（高精度）。这个工具产生的数据可以让你了解模型在应用中运行的性能，你可以尝试调整参数来使模型达到你所要求的性能。要得到能够准确适配你应用的参数，你可以看看怎样生成一个 [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) 来帮助你理解其中的平衡。

## 识别命令行

数据流正确率工具使用了一个简单的解码器，它包含了一个简短的 C++ 类，叫做[识别命令行](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands/recognize_commands.h)。这个类提供了输入 TensorFlow 模型的数据，它平均分割音频数据，然后当它有充足的证据认为一个词语被识别出来时返回一个标签信息。这个类的实现相当的小，只是追踪最近的几个预测值然后均分它们，所以当需要接入其他平台和语言时很简单。例如，在安卓上用 Java 或者在树莓派上用 Python 实现相似的东西很方便。只要这些实现使用相同的逻辑，你能使用数据流测试工具控制各个参数来均分数据流，然后把它们传递到你的应用中得到相似的结果。

## 进阶训练

这个训练脚本设计的初衷是处理一个相对小的文件生成个良好的点对点式的结果，但是有你能更改的许多选项用来根据你自己的需求来定制结果。

### 定制训练数据集

这个脚本默认会下载 [Speech Commands dataset](https://download.tensorflow.org/data/speech_commands_v0.02.tgz)，但是你也能提供你自己的训练数据。要训练你自己的数据，你要确保你至少有几百份语音记录，而且每一份记录你都要做识别并把他们按文件夹分类。例如，如果你要识别狗叫的“汪汪叫”和猫叫的“喵喵喵”，你需要先创建一个叫`动物叫声`的根目录，然后在其中建立两个子文件夹`汪汪叫`和`喵喵喵`，然后把你的音频文件放入相应的文件夹。

要让脚本文件找到你的新语音文件，你需要设置 `--data_url=` 来停用下载,并用 `--data_dir=/your/data/folder/` 来寻找到你刚才创建的文件。

你要确保这些文件是 16 位、小端模式、PCM 编码的 WAVE 格式文件。采样率默认为 16,000，但是只要保证你所有的文件采样率一样（脚本不支持重新采样），你就能通过改变 `--sample_rate` 参数来使用不同采样率的文件。这些音频片段也应当是大致相同的时间长度，默认预期时间长度是一秒钟，你能通过设置  `--clip_duration_ms` 来设置。如果你在一开始有很多的静默片段，你能看下词语对齐工具来将它们标准化（[这是个快速使用此工具的方法](https://petewarden.com/2017/07/17/a-quick-hack-to-align-single-word-audio-recordings/)）。

需要注意的一个问题就是，你的数据集中可能有许多相同的声音重复，如果它们分布在你的训练集、验证集和测试集中，可能会产生误导性的指标。例如， Speech Commands 数据集中人们多次重复同一单词。这些重复中的每一个都可能和其他的非常相似，所以如果模型训练过拟合或适应了训练数据集，当它在测试集中看到一个非常相似的副本时可能表现出不切实际的准确率。要避免这种危险，Speech Commands 试图确保所有由同一个人说的同一单词被放入了一样的部分中。根据于片段文件名的哈希值来分配文件到训练集、测试集或者验证集总，来确保即使有新片段添加进来或者训练样例分配到其他的数据集中也能使分配保持这种规则。来保证所有一个人讲的的词语都在一个相同的数据集里，[the hashing function](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands/input_data.py)会在计算时忽略文件名后有 `_nohash_` 的文件。这也代表着文件名像 `pete_nohash_0.wav` 和 `pete_nohash_1.wav`,它们一定会在相同的数据集中。

### 未知类

你的应用可能会听到一些不在你训练数据集的声音，并且你希望模型在这种情况下不将其识别为噪音。为了帮助网络学习什么样的声音可以被忽略，你需要提供一些不在任何类型中的音频片段。做到这点，你需要创建 `quack`,`oink`，和 `moo` 子文件夹并放入用户会遇到的其他动物发出的噪音。`--wanted_words` 参数告诉脚本哪些类是你关心的，所有子文件夹名中提及的会在训练中加入一个 `_unknown_` 类。这个 Speech Commands 数据集在未知类中有二十个词语，包括 0 到 9 的数字和随机的名字像 “Sheila”。

默认的，有 10% 的训练样例是从未知类中选出的，但是你可以通过 `--unknown_percentage` 参数来控制它。增加这个数字会减少模型将未知词语识别为期望词语的可能性，但是将它提升过大会适得其反，因为模型可能会认为所有单词都分类为未知是最安全的！

### 背景噪音

即使在环境中还有其他不想关的声音，应用也必须要识别出音频。为了建立一个对抗强烈干扰的模型，我们需要对具有相似属性的录制音频进行训练。在 Speech Commands 数据集中的文件是用户在不同环境用不同设备录制的，而不是在一个录音室中，所以这样有助于增加训练的真实性。要做得更好，可以将随机环境音频混合到训练输入中。在 Speech Commands 数据集中有一个特别的文件夹叫做 `_background_noise_`，它里面装有长达一分钟包含白噪音、机器声或者日常家庭生活声音的 WAVE 文件。

这些文件的小片段是被随机选中并以一个以低音量在训练中被混入片段中。它的响度也是随机的，并以 `--background_volume` 属性按比例来控制，这里 0 是静默，1 是全音量。而且也不是所有的片段都加入背景音，`--background_frequency` 属性控制哪些部分会混入背景音。

你自己的应用可能会要在它的运行场景中处理与默认不同的背景噪音，所以你能在 `_background_noise_` 文件夹中添加你自己的音频片段。这些文档应当和主数据集采用一样的采样率，但是它的持续时间更长，这样才能在其中取到一个良好的随机片段。

### 静默

在大多数情况下，你寻找的声音总是间断出现的，所以知道什么时候没有匹配到语音是很重要的。要做到这一点，有一个特别的标签 `_silence_` ，它能表明这模型没有侦测到任何感兴趣的东西。因为在真实的环境中，永远没有真正的静默，所以我们需要提供静默和不相关的音频样例。为此，我们重用了 `_background_noise_` 文件夹，它也会混入真实片段中，放入短片的声音数据并和真实类型为 `_silence_` 的片段一起混和输入。有 10% 的训练数据默认的会这样输入，但也能通过 `--silence_percentage` 属性能够用来控制它。和未知词语一样，将这个值设置过高会增加模型识别静默的可能性更大，但是会以词语的错误识别为代价，但是太大也会导致陷入总是猜测是否为静默的圈套中。

### 时间偏移

加入背景噪音是一种扭曲训练集的方法，它能有效提升数据集大小，从而提升整体的准确性，而时移是另一种方法。这涉及训练集样本数据的随机偏移，开始或者结束的一小部分被切除然后填充进零。这能在训练集中的开始时间模拟自然的变化，并用 `--time_shift_ms` 参数控制，默认为 100 ms。增加这个值会增加更多的变化，但是这有切断音频中重要部分的风险。使用扭曲增加数据的另一个相关方法是使用[时间拉伸和音高缩放](https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling)，但这些已经超出了这个教程的范围。

## 定制模型

这个脚本使用的默认模型相当的大，每轮推算要进行八十亿次浮点运算每秒而且使用 940,000 权重参数。这个数值在桌面环境或者移动手机上都是可以接受的，但是它在只拥有有限资源的设备上以交互模式运行时，对它来说就是过大的运算压力了。为了应对这种使用场景，有以下几种替代方案可供选择：

**low_latency_conv** 

基于在 [Convolutional Neural Networks for Small-footprint Keyword Spotting paper](http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf) 中的 'cnn-one-fstride4'技术。它的精确度略低于 'conv'，但权重参数的数值大致相同，而且它只需要一千一百万次浮点运算每秒即可进行一次预测，这样就使预测速度更快了。

使用这种模型，你要在命令行中特别指定 `--model_architecture=low_latency_conv` 。你也需要修正学习速率和训练轮次，所以完整的命令会是这样的：

```
python tensorflow/examples/speech_commands/train \
--model_architecture=low_latency_conv \
--how_many_training_steps=20000,6000 \
--learning_rate=0.01,0.001
```

这会使训练脚本使用 0.01 的学习速率进行 20,000 轮的训练，然后以小10倍的速度微调 6000 轮。

**low_latency_svdf**

基于[Compressing Deep Neural Networks using a Rank-Constrained Topology paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf)中提供的技术。它的准确性也略低于 'conv' 但是它只使用了大概七百五十万个参数，并且最有成效的是，在测试时允许它有个最优的执行（即当你真的在你的应用程序中使用它），得出结果需要七百五十万次浮点运算每秒。

要使用这种模型，你要在命令行中特别设定 `--model_architecture=low_latency_svdf`，并修正学习速率和训练轮次，所以完整的命令会是这样的：

```
python tensorflow/examples/speech_commands/train \
--model_architecture=low_latency_svdf \
--how_many_training_steps=100000,35000 \
--learning_rate=0.01,0.005
```

请注意，尽管它比前两个技术需要更多的步骤数，计算轮次的减少意味着训练应该占用同样的时间，最终达到 85% 左右的精确度。你还能够通过更改 SVDF layer 中的以下这些参数来进一步的调整这种技术，使它更易于被计算并更加精确：

* rank - 近似的等级（通常越高越好，但是需要更多计算才能得出结果）。
* num_units - 与其他层类型相似，指定层中的节点数（越多的节点质量越高，越多的计算量）。

关于运行时，由于该层允许通过缓存一些内部神经网络激活来优化，所以你需要在冻结图和在数据流模式中执行模型（例如 test_streaming_accuracy.cc）时保持一致的进度。

**其他用于定制的参数**

如果你想实验定制模型，调整声谱图的创建参数就是一个不错的开始。这些参数能够改变输入模型的图像大小，并且在 [models.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands/models.py) 的创建代码中根据不同的大小自动适配计算量和权重。如果你减小了输入，模型会只需要较少的计算，这样用一定的准确性为代价交换潜在的性能提升就是一个好方法。 `--window_stride_ms` 参数控制每个频率分析样本与之前一个的距离。如果增加此值，则在给定的持续时间内采样数量会减少，输入的时间长度也会缩短。 `--dct_coefficient_count` 参数控制控制着多少数据段被用于计数，所以减少这个值会缩小另一个维度的输入。`--window_size_ms` 参数不会影响大小，但是可以控制用于计算每个样本的频率的区域的范围。减少控制训连样本持续时间的 `--clip_duration_ms` 也会在你寻找短时间音频有所帮助，因为你减少了输入的时间长度。你需要确保所有的训练数据在片段的开头部分包含了正确的音频。

如果你为你的问题构想了个完全不同的模型，你你可以将它插入 [models.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands/models.py) 并让脚本的其余部分处理所有的预处理和训练工作。你可以添加一个新模型到 `create_model` 中，找到你新模型的名称并调用模型创建函数。这个函数输出了输入的声谱图的尺度，还有其他一些模型的信息，并且期望创建 TensorFlow 来读取并产生一个输出预测向量，以及一个控制丢失率的控制器。脚本的其他部分将这个模型整合到一个更大的图中，进行输入计算并用 softmax 函数和损失函数来训练它。

当你调整模型和训练超参数时，一个常见的问题是由于数值精度的问题，非数值可能会混入。一般来说，你可以通过减小学习速率和权重初始化来解决这些问题，但是如果它们总是出现，你可以启用 `--check_nans` 来追踪错误源头。这将在 TensorFlow 中的大部分常规操作间插入检查操作，并在遇到问题时重视培训过程并显示有用的错误信息。
