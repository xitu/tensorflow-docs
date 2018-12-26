# 性能

此文档罗列了在 Android 和 iOS 设备上运行一些经典的模型时 TensorFlow Lite 性能基准。

这些性能基准值由 [Android TFLite benchmark binary](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/tools/benchmark) 和 [iOS benchmark app](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/tools/benchmark/ios) 生成。


# Android 性能基准

在测试 Android 基准时，为了减少差异 CPU 关联都设定到使用设备上最多的核心（[详细](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/tools/benchmark#reducing-variance-between-runs-on-android)请查看）

它假定模型都是下载并解压到 `/data/local/tmp/tflite_models` 目录下。基准程序按照[这些说明](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/tools/benchmark#on-android)构建并假定位于 `/data/local/tmp` 目录下。

使用以下命令运行基准程序：

```
adb shell taskset ${CPU_MASK} /data/local/tmp/benchmark_model \
  --num_threads=1 \
  --graph=/data/local/tmp/tflite_models/${GRAPH} \
  --warmup_runs=1 \
  --num_runs=50 \
  --use_nnapi=false
```

这里，`${GRAPH}` 是模型的名字，`${CPU_MASK}` 是按照下表选择的 CPU 关联：

设备 | CPU_MASK |
-------| ----------
Pixel 2 | f0 |
Pixel xl | 0c |


<table>
  <thead>
    <tr>
      <th>模型名</th>
      <th>设备 </th>
      <th>推理所用平均时间（std dev）</th>
    </tr>
  </thead>
  <tr>
    <td rowspan = 2>
      <a href="http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
    </td>
    <td>Pixel 2 </td>
    <td>166.5 ms (2.6 ms)</td>
  </tr>
   <tr>
     <td>Pixel xl </td>
     <td>122.9 ms (1.8 ms)  </td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224_quant.tgz">Mobilenet_1.0_224 (quant)</a>
    </td>
    <td>Pixel 2 </td>
    <td>69.5 ms (0.9 ms)</td>
  </tr>
   <tr>
     <td>Pixel xl </td>
     <td>78.9 ms (2.2 ms)  </td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>
    </td>
    <td>Pixel 2 </td>
    <td>273.8 ms (3.5 ms)</td>
  </tr>
   <tr>
     <td>Pixel xl </td>
     <td>210.8 ms (4.2 ms)</td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>
    </td>
    <td>Pixel 2 </td>
    <td>234.0 ms (2.1 ms)</td>
  </tr>
   <tr>
     <td>Pixel xl </td>
     <td>158.0 ms (2.1 ms)</td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>
    </td>
    <td>Pixel 2 </td>
    <td>2846.0 ms (15.0 ms)</td>
  </tr>
   <tr>
     <td>Pixel xl </td>
     <td>1973.0 ms (15.0 ms)  </td>
  </tr>
  <tr>
    <td rowspan = 2>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>
    </td>
    <td>Pixel 2 </td>
    <td>3180.0 ms (11.7 ms)</td>
  </tr>
   <tr>
     <td>Pixel xl </td>
     <td>2262.0 ms (21.0 ms)  </td>
  </tr>

 </table>

# iOS 基准

为了测试 iOS 基准，修改了 [benchmark app](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/tools/benchmark/ios) 以包含合适的模型并且`benchmark_params.json` 中 `num_threads` 设定为 1。

<table>
  <thead>
    <tr>
      <th>模型名</th>
      <th>设备 </th>
      <th>推理所用平均时间（std dev）</th>
    </tr>
  </thead>
  <tr>
    <td>
      <a href="http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz">Mobilenet_1.0_224(float)</a>
    </td>
    <td>iPhone 8 </td>
    <td>32.2 ms (0.8 ms)</td>
  </tr>
  <tr>
    <td>
      <a href="http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224_quant.tgz)">Mobilenet_1.0_224 (quant)</a>
    </td>
    <td>iPhone 8 </td>
    <td>24.4 ms (0.8 ms)</td>
  </tr>
  <tr>
    <td>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz">NASNet mobile</a>
    </td>
    <td>iPhone 8 </td>
    <td>60.3 ms (0.6 ms)</td>
  </tr>
  <tr>
    <td>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz">SqueezeNet</a>
    </td>
    <td>iPhone 8 </td>
    <td>44.3 (0.7 ms)</td>
  </tr>
  <tr>
    <td>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz">Inception_ResNet_V2</a>
    </td>
    <td>iPhone 8</td>
    <td>562.4 ms (18.2 ms)</td>
  </tr>
  <tr>
    <td>
      <a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz">Inception_V4</a>
    </td>
    <td>iPhone 8 </td>
    <td>661.0 ms (29.2 ms)</td>
  </tr>
 </table>
