# 模型优化

在将机器学习模型部署到移动设备时，推理效率是一个关键问题。当训练的计算需求随着在不同体系结构上训练的模型的数量而增长时，推理的计算需求与用户的数量正比增长。Tensorflow模型优化工具包最大限度地降低了推理的复杂性 — 模型大小，延迟和功耗。

## 用例

模型优化适用于：

- 将模型部署到算力，内存和功耗受限的边缘设备上。例如，移动设备和物联网（IoT）设备。
- 减少无线模型更新的有效负载大小。
- 在由定点操作约束的硬件上执行。
- 优化专用硬件加速器的模型。

## 优化方法

模型优化使用的多种技术：

- 减少参数的数量，例如剪枝和结构化剪枝。
- 降低表示精度，例如量化。
- 通过减少参数或快速执行将原始模型拓扑更新为更有效的拓扑，例如，张量分解方法和蒸馏。

## 模型量化

使用允许降低权重的精确表示并且可选地，存储和计算的激活方式来量化深度神经网络。量化有以下几个好处：

- 支持在已有的 CPU 平台上运行。
- 量化激活降低了读取和存储中间激活器的存储器访问成本。
- 许多 CPU 和硬件加速器实现提供 SIMD 指令功能，这对量化特别有帮助。

[TensorFlow Lite](../lite) 为量化提供了几个级别的支持。

[Post-training 量化](post_training_quantization.md) 将权重和激活量化为训练后，且使用简单。[Quantization-aware training](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantize/README.md){:.external} 考虑到训练网络可以以最小的准确率下降来量化，而且只是对卷积神经网络结构的一个子集有用。

### **延迟和准确率结果**

下边是在几个模型上进行了 post-training quantization 和 quantization-aware training 的延迟和准确率的结果。所有的延迟在单核处理器的 Pixel 2 上测得。随着不同的组合配置，数据结果展示如下：

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Top-1 Accuracy (Original) </th>
      <th>Top-1 Accuracy (Post Training Quantized) </th>
      <th>Top-1 Accuracy (Quantization Aware Training) </th>
      <th>Latency (Original) (ms) </th>
      <th>Latency (Post Training Quantized) (ms) </th>
      <th>Latency (Quantization Aware Training) (ms) </th>
      <th> Size (Original) (MB)</th>
      <th> Size (Optimized) (MB)</th>
    </tr>
    <tr><td>Mobilenet-v1-1-224</td><td>0.709</td><td>0.657</td><td>0.70</td>
      <td>180</td><td>145</td><td>80.2</td><td>16.9</td><td>4.3</td></tr>
    <tr><td>Mobilenet-v2-1-224</td><td>0.719</td><td>0.637</td><td>0.709</td>
      <td>117</td><td>121</td><td>80.3</td><td>14</td><td>3.6</td></tr>
   <tr><td>Inception_v3</td><td>0.78</td><td>0.772</td><td>0.775</td>
      <td>1585</td><td>1187</td><td>637</td><td>95.7</td><td>23.9</td></tr>
   <tr><td>Resnet_v2_101</td><td>0.770</td><td>0.768</td><td>N/A</td>
      <td>3973</td><td>2868</td><td>N/A</td><td>178.3</td><td>44.9</td></tr>
 </table>
  <figcaption>
    <b>表格 1</b> 选择 CNN 模型的模型量化的好处
  </figcaption>
</figure>

## 量化工具的选择

当你开始时，请检查 tensorflow lite 模型存储库中的模型是否适用于你的应用。如果不适用，我们建议用户从 post-training 量化工具开始，因为它适用性强且无需训练数据。对于那些不满足准确率和延迟要求的，或硬件加速器支持很重要的情况下，quantization-aware training 是一个更好的选择。

注意：Quantization-aware training 支持卷积神经网络架构的子集。
