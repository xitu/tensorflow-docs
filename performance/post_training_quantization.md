## Post-training 量化

Post-training 量化是一种可以减小模型大小，同时降低 3 倍延迟并且仍然能够保持模型精度的一般方法。Post-training 量化将权重从浮点量化为 8 位精度。此技术在 [TensorFlow Lite model converter](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/toco) 中作为一个功能选项被使用。

```
import tensorflow as tf
converter = tf.contrib.lite.TocoConverter.from_saved_model(saved_model_dir)
converter.post_training_quantize = True
tflite_quantized_model = converter.convert()
open("quantized_model.tflite", "wb").write(tflite_quantized_model)
```

在推理时，权重值从 8 位精度转换为浮点数，并使用浮点内核进行计算。此转换只执行一次并进行缓存以减少延迟。

为了进一步改善延迟，混合运算符动态地将激活量化为 8 位，并使用 8 位权重和激活函数执行计算。此种优化方式可以提供接近完全定点推断时的延迟。但是，输出仍然使用浮点存储，因此混合运算的加速效果仍然小于完全定点计算。混合操作可用于大多数计算密集型网络：

- [tf.contrib.layers.fully_connected](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected)
- [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
- [tf.nn.embedding_lookup](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup)
- [BasicRNN](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicRNNCell)
- [tf.nn.bidirectional_dynamic_rnn for BasicRNNCell type](https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn)
- [tf.nn.dynamic_rnn for LSTM and BasicRNN Cell types](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)

由于权重在训练后被量化，因此可能存在精度损失，特别是对于较小的网络。[TensorFlow Lite model repository](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md#image-classification-quantized-models)提供了为特定网络提供预训练的完全量化模型。检查量化模型的准确性以验证任何精度上的降低是否在可接受的限度内是很重要的。这里有一个工具可以评估 [TensorFlow Lite 模型精确度](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/tools/accuracy/README.md)。

如果精确度下降幅度过大，可以考虑使用[注重量化的训练](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantize/README.md)。

### 量化张量的表示

TensorFlow 将数字浮点数组转换为 8 位表示形式作为压缩问题。由于训练的神经网络模型中的权重和激活张量倾向于分布在相对小范围内的值。（例如，对于权重为 -15 到 +15，对于图像模型激活为 -500 到 1000）。并且由于神经网络在处理噪声数据时具有健壮性，因此通过量化到一小组值引入的误差将整体结果的精度保持在可接受的阈值内。选定的表示必须执行快速计算，尤其是在运行模型时产生的具有大量计算的大型矩阵乘法。 

这用两个浮点表示，它存储对应于最低和最高量化值的总体最小值和最大值。量化阵列中的每个条目表示该范围内的浮点值，在最小值和最大值之间线性分布。例如，当一个 8 位数组的最小值为 -10.0，最大值为 30.0f 时，其量化值表示如下： 

<figure>
  <table>
    <tr><th>量化值</th><th>浮点值</th></tr>
    <tr><td>0</td><td>-10.0</td></tr>
    <tr><td>128</td><td>10.0</td></tr>
    <tr><td>255</td><td>30.0</td></tr>
  </table>
  <figcaption>
    <b>表格 2</b>: 量化值范围示例
  </figcaption>
</figure>

这种表示方式的好处有：

- 它有效地表示任意大小的范围。
- 数值无需对称。
- 有符号数和无符号数均可被表示。
- 线性扩展使乘法变得简单。
