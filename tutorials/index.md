# 教程


本章包括了演示在 TensorFlow 中进行各种特定任务的教程。如果你刚接触 TensorFlow，建议在学习以下教程前先阅读 @{$get_started$Get Started} 一章。

## 图像

以下教程涵盖了图像识别的各个方面：

  * @{$layers}，介绍了卷积神经网络（CNN），并演示了如何在 TensorFlow 中构建 CNN。
  * @{$image_recognition}，本教程简单介绍了图像识别领域以及用于识别图像的 Inception 预训练模型。
  * @{$image_retraining}，内容如题所示。
  * @{$deep_cnn}，演示了如何构建一个小型的 CNN 来进行图像识别。本教程面向 TensorFlow 的进阶用户。


## 序列

以下教程主要讲解了如何解决序列数据的机器学习问题。

  * @{$recurrent}，演示了如何使用一个循环神经网络对句中的后一个单词进行预测。
  * @{$seq2seq}，演示了如何使用一个序列到序列（seq2seq）模型将英语文本翻译为法语。
  * @{$recurrent_quickdraw} 构建一个直接通过笔划顺序对图片进行分类的模型。
  * @{$audio_recognition}，展示了如何构建一个基本的语音识别网络。

## 数据的表示

以下教程讲解了可在 TensorFlow 中使用的各类数据表示方法。

  * @{$wide}，使用
    @{tf.feature_column$feature columns} 将各种类型的数据传入线性模型，解决一个分类问题。
  * @{$wide_and_deep}，在上文线性模型教程的基础之上，加入一个深度前馈神经网络以及兼容 DNN 的数据表示方法。
  * @{$word2vec}，演示了如何构建词嵌入。
  * @{$kernel_methods}，展示了如何使用显式的核映射来改进线性模型的质量。

## 非机器学习类

TensorFlow 的核心是一个强大的数值计算系统，因此虽然 TensorFlow 专注于机器学习领域，但是您也可以使用 TensorFlow 解决一些其它类型的数学问题。例如：

  * @{$mandelbrot}
  * @{$pdes}

