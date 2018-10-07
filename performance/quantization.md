# 定点量化

量化技术计算并存储了更加紧凑的数字格式。[TensorFlow Lite](/mobile/tflite/) 增加了使用 8 位的定点量化表示。

由于现代神经网络的挑战之一是进行高精度的优化，首先要做的是改善训练期的精度和速度。使用浮点数运算是保持精度的简单方法之一，同时 GPU 也被设计为能为这些运算进行加速。

然而，随着越来越多的机器学习模型需要被部署到移动设备上，推理的效率已经成为了一个关键性问题。对于**训练期**的计算需求，随着在不同架构上训练的模型的数量增加而迅速增长；对于**推断**的计算需求，也随着用户数量的增加而成比例的增加。

## 量化的优势

使用 8 位定点量化表示的计算可以加速模型的运行速度，同时也能降低功耗。这点对于无法高效运行浮点计算的移动设备和嵌入式应用是非常有用的。比如物联网（IoT）和机器人设备。此外，在后端扩展这种支持、研究更低精度的神经网络还有很多机遇。

### 更小的文件大小 {: .hide-from-toc}

神经网络的模型需要消耗大量的磁盘空间。举个例子，原始的 AlexNet 需要至少 200 MB 的空间来存储浮点格式的模型文件 —— 几乎全部用于模型数百万的权重。在权重间只有细微差异的表示中，简单的压缩格式效果不佳（如 Zip）。

权重在所有层中都以数值形式出现。对每一层而言，权重倾向于分布在一定范围内。量化技术则可以通过存储每层中的最大和最小的权重，然后压缩每层权重的浮点值转换为表示在 256 个值范围内最接近真实实数的 8 位整数，从而达到压缩文件大小的目的。

### 更快的推断 {: .hide-from-toc}

由于计算完全是在 8 位输入和输出上执行的，量化减少推理计算所需的计算资源。这在训练阶段需要引入更多浮点计算，但在推断期间则会加速很多。

### 内存效率 {: .hide-from-toc}

对比浮点值而言，获取 8 位值只需要 25% 的内存和带宽，更加有效的避免了 RAM 访问的瓶颈。在很多情况下，神经网络的运行性能取决于内存的访问。使用八位定点值的权重与激活值所带来的提升是显著的。

通常情况下，SIMD 操作能使在每个时钟周期内运行更多的操作。某些情况下，DSP 芯片还可以加速八位计算，最终获得大规模的加速。

## 定点量化技术

我们的目标是要在训练和推断期间内，对于权重和激活值使用相同的精度，但是一个相当重要的区别是在前向和后向传播中，推断只使用了前向过程。所以当我们训练模型期间同时加入量化，就要确保前向过程的训练和推理的精度相匹配。

为了尽量减少完全定点模型（权重与激活）的精度损耗，我们在训练中就使用量化。这样就模拟了模型在前向传播中的量化从而权重会倾向于其值在量化推断期间表现更好。后向传播使用量化后的权重、激活值及模型直接给估计器使用（见 [Bengio et al, 2013](https://arxiv.org/abs/1308.3432) ）。

此外，还需要在训练期间确定激活值的最小值和最大值。这是为了训练中使用量化时不费吹灰之力的将其转换为定点推断模型。从而消除了一个需要单独校准的步骤。

## 使用 TensorFlow 进行量化训练

TensorFlow 可以在训练模型的同时完成量化。由于训练时需要对梯度进行少量调整，所以仍然使用浮点值。为了保证在增加量化误差时候模型仍然是浮点值，@{$array_ops#Fake_quantization$fake quantization} 节点模拟了前向与后向量化的效果。

由于很难将这些伪量化操作添加到所有模型所需的位置，有一个函数可以帮助我们重写整个训练图。从而创建一个伪量化的训练图，我们有：

```python
# 构建模型的前向传播
loss = tf.losses.get_total_loss()

# 调用训练重写，将模型内部的 FakeQuantization 节点以及 fold batchnorm 
# 为训练期进行重写。这通常需要使用训练时的量化工具对浮点模型进行调优。
# 当重头开始训练时，quant_delay 可以用来激活量化后的训练过程收敛到浮点图，
# 从而有效的对模型进行调优
tf.contrib.quantize.create_training_graph(quant_delay=2000000)

# 如同往常一样调用后向传播
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer.minimize(loss)
```

重写后的 **eval 图**与**训练图**并非平凡地等价，这是因为量化操作会影响 batchnorm 这一步骤。因此，我们为 **eval 图**增加了单独的重写步骤：

```Python
# 构建 eval 图
logits = tf.nn.softmax_cross_entropy_with_logits_v2(...)

# 调用 eval 重写，在 FakeQuantization 节点重写图并使用 fold batchnorm 技术
tf.contrib.quantize.create_eval_graph()

# 将检查点和 eval 图原型保存并冻结到磁盘后提供给 TFLite
with open(eval_graph_file, ‘w’) as f:
  f.write(str(g.as_graph_def()))
saver = tf.train.Saver()
saver.save(sess, checkpoint_name)
```

重写训练和评估图的方法是一个活跃的研究和实验领域。尽管重写和量化训练可能无法奏效或者不能提高所有模型的性能，但我们正在努力推广这些技术。

## 生成全量化模型

前面演示的重写后的、重新计算求出参数后的 eval 图仅仅只是**模拟**了量化这一过程。为了从训练的量化模型生成实际的定点运算，还需要将其转换为定点内核。TensorFlow Lite 支持从 `create_eval_graph` 生成图形并进行此转换。

首先，通过 TensorFlow Lite 工具链创建一个冻结的图：

```Shell
bazel build tensorflow/python/tools:freeze_graph && \
  bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=eval_graph_def.pb \
  --input_checkpoint=checkpoint \
  --output_graph=frozen_eval_graph.pb --output_node_names=outputs
```

然后将输出结果提供给 TensorFlow Lite 优化转换器（TOCO）以获得全量化的 TensorFLow Lite 模型：

```Shell
bazel build tensorflow/contrib/lite/toco:toco && \
  ./bazel-bin/third_party/tensorflow/contrib/lite/toco/toco \
  --input_file=frozen_eval_graph.pb \
  --output_file=tflite_model.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape="1,224, 224,3" \
  --input_array=input \
  --output_array=outputs \
  --std_value=127.5 --mean_value=127.5
```

请查看 `tf.contrib.quantize` 和 [TensorFlow Lite](/mobile/tflite/) 文档。

## 量化模型的精度

定点形式的 [MobileNet](https://arxiv.org/abs/1704.0486) 模型由八位权重与激活方式发布。通过使用重写器，模型实现了表 1 中列出的 Top-1 精度。作为比较，这里针对同样的模型列出了浮点精度。用于生成这些模型的代码可以连同所有预训练的 [mobilenet_v1 模型](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)一起使用。

<figure>
  <table>
    <tr>
      <th>图片大小</th>
      <th>深度</th>
      <th>Top-1 精度:<br>浮点值</th>
      <th>Top-1 精度:<br>定点：8 位权重及激活值</th>
    </tr>
    <tr><td>128</td><td>0.25</td><td>0.415</td><td>0.399</td></tr>
    <tr><td>128</td><td>0.5</td><td>0.563</td><td>0.549</td></tr>
    <tr><td>128</td><td>0.75</td><td>0.621</td><td>0.598</td></tr>
    <tr><td>128</td><td>1</td><td>0.652</td><td>0.64</td></tr>
    <tr><td>160</td><td>0.25</td><td>0.455</td><td>0.435</td></tr>
    <tr><td>160</td><td>0.5</td><td>0.591</td><td>0.577</td></tr>
    <tr><td>160</td><td>0.75</td><td>0.653</td><td>0.639</td></tr>
    <tr><td>160</td><td>1</td><td>0.68</td><td>0.673</td></tr>
    <tr><td>192</td><td>0.25</td><td>0.477</td><td>0.458</td></tr>
    <tr><td>192</td><td>0.5</td><td>0.617</td><td>0.604</td></tr>
    <tr><td>192</td><td>0.75</td><td>0.672</td><td>0.662</td></tr>
    <tr><td>192</td><td>1</td><td>0.7</td><td>0.69</td></tr>
    <tr><td>224</td><td>0.25</td><td>0.498</td><td>0.482</td></tr>
    <tr><td>224</td><td>0.5</td><td>0.633</td><td>0.622</td></tr>
    <tr><td>224</td><td>0.75</td><td>0.684</td><td>0.679</td></tr>
    <tr><td>224</td><td>1</td><td>0.709</td><td>0.697</td></tr>
  </table>
  <figcaption>
    <b>表 1</b>：MobileNet 在 Imagenet 验证集上的 Top-1 精度
  </figcaption>
</figure>

## 量化张量的表示

作为压缩，TensorFlow 会将浮点数数组转换为 8 位定点表示。由于训练好的神经网络中，权重和激活张量的值更倾向于分布在相对范围较小之内（例如，权重为 -15 到 +15 或者 -500 到 +100 用于图像模型的激活函数）。由于神经网络倾向于鲁棒地处理噪声，量化所引入的误差在整体结果的精度总是保持在可接受的阈值之内。选择的表示形式必须具备快速执行计算的能力，尤其是在运行模型时组成大量计算的大型矩阵乘法。

这种具有两个浮点数的表示形式存储了整体值中最小和最大值所对应的最低和最高的量化值。每个量化值数组的实例表示了一个浮点数的范围，线性的分布在最小值和最大值之间。举个例子，最小值为 -10.0 和 最大值为 30.0f 及一个八位数组，其量化值的表示如下：

<figure>
  <table>
    <tr><th>量化的</th><th>浮点的</th></tr>
    <tr><td>0</td><td>-10.0</td></tr>
    <tr><td>128</td><td>10.0</td></tr>
    <tr><td>255</td><td>30.0</td></tr>
  </table>
  <figcaption>
    <b>表 2</b>：量化值的平均的例子
  </figcaption>
</figure>

这种表示形式的优势在于：

* 有效的表示了任意范围的大小
* 值无需对称
* 表示形式同时表达了有符号数和无符号数
* 其线性展开使得乘法相对简单。

其他备用的技术通过使用整个表示中非线性地分配浮点值作为低位的深度，然而这使得计算时间方面相对更加昂贵（见 Han et al.[2016](https://arxiv.org/abs/1510.00149) ）。

对量化格式有清晰定义的好处在于，对于未准备好量化的操作，或者为了调试而检查张量，始终可以从定点到浮点之间来回转换。
