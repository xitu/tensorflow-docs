# Developer Guide
# 开发者指南

Using a TensorFlow Lite model in your mobile app requires multiple considerations: you must choose a pre-trained or custom model, convert the model to a TensorFLow Lite format, and finally, integrate the model in your app.

在你的移动 app 中使用 TensorFlow Lite 模型需有如下注意事项：你必须选择一个预训练或者自定义模型，把这个模型转化为 TensorFlow Lite 格式，最后把模型整合进你的 app 中。

## 1. Choose a model
## 1. 模型选择
Depending on the use case, you can choose one of the popular open-sourced models, such as *InceptionV3* or *MobileNets*, and re-train these models with a custom data set or even build your own custom model.

取决于使用实例，你可以选择任何一个流行的开源模型，例如 *InceptionV3* 或者 *MobileNets*，然后使用你自定义的数据集对（你选择的）这些模型进行重新训练，或者甚至构建你自定义的模型。

### Use a pre-trained model
### 使用预训练模型
[MobileNets](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html) is a family of mobile-first computer vision models for TensorFlow designed to effectively maximize accuracy, while taking into consideration the restricted resources for on-device or embedded applications. MobileNets are small, low-latency, low-power models parameterized to meet the resource constraints for a variety of uses. They can be used for classification, detection, embeddings, and segmentation—similar to other popular large scale models, such as [Inception](https://arxiv.org/pdf/1602.07261.pdf). Google provides 16 pre-trained [ImageNet](http://www.image-net.org/challenges/LSVRC/) classification checkpoints for MobileNets that can be used in mobile projects of all sizes.

[MobileNets](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html) 是一系列专为 TensorFlow 设计的移动（场景）优先的计算机视觉模型，这类模型用于有效地最大化（提升）精确度，同时，模型还考虑到设备内置应用或嵌入式应用的资源限制问题。 MobileNets 是小型化，低延迟，低能耗的模型 ，能够参数化地满足各种各样使用案例中资源限制的要求。这些模型可以被用于分类、检测、嵌入和分割 —— 原理和其他一些流行的大规模模型相似，例如 [Inception](https://arxiv.org/pdf/1602.07261.pdf). 谷歌提供了

[Inception-v3](https://arxiv.org/abs/1512.00567) is an image recognition model that achieves fairly high accuracy recognizing general objects with 1000 classes, for example, "Zebra", "Dalmatian", and "Dishwasher". The model extracts general features from input images using a convolutional neural network and classifies them based on those features with fully-connected and softmax layers.

[Inception-v3](https://arxiv.org/abs/1512.00567) 是一个能对 1000 种常见事物，例如，“斑马”、“斑点狗”、“洗碗机”，进行非常高精度识别的图像识别模型。这个模型使用了一个卷积神经网络来提取输入图片的一般特征，然后在这些特征的基础上使用全连接和 softmax 层来对这些图片进行分类。

[On Device Smart Reply](https://research.googleblog.com/2017/02/on-device-machine-intelligence.html) is an on-device model that provides one-touch replies for incoming text messages by suggesting contextually relevant messages. The model is built specifically for memory constrained devices, such as watches and phones, and has been successfully used in Smart Replies on Android Wear. Currently, this model is Android-specific.

[On Device Smart Reply](https://research.googleblog.com/2017/02/on-device-machine-intelligence.html) 是一个

These pre-trained models are [available for download](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md)

### Re-train Inception-V3 or MobileNet for a custom data set
### 
These pre-trained models were trained on the *ImageNet* data set which contains 1000 predefined classes. If these classes are not sufficient for your use case, the model will need to be re-trained. This technique is called *transfer learning* and starts with a model that has been already trained on a problem, then retrains the model on a similar problem. Deep learning from scratch can take days, but transfer learning is fairly quick. In order to do this, you need to generate a custom data set labeled with the relevant classes.

The [TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/) codelab walks through the re-training process step-by-step. The code supports both floating point and quantized inference.

### Train a custom model
### 自定义模型训练
A developer may choose to train a custom model using Tensorflow (see the @{$tutorials} for examples of building and training models). If you have already written a model, the first step is to export this to a @{tf.GraphDef} file. This is required because some formats do not store the model structure outside the code, and we must communicate with other parts of the framework. See [Exporting the Inference Graph](https://github.com/tensorflow/models/blob/master/research/slim/README.md) to create .pb file for the custom model.

TensorFlow Lite currently supports a subset of TensorFlow operators. Refer to the [TensorFlow Lite & TensorFlow Compatibility Guide](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/g3doc/tf_ops_compatibility.md) for supported operators and their usage. This set of operators will continue to grow in future Tensorflow Lite releases.

## 2. Convert the model format
## 2. 模型格式转换
The model generated (or downloaded) in the previous step is a *standard* Tensorflow model and you should now have a .pb or .pbtxt @{tf.GraphDef} file. Models generated with transfer learning (re-training) or custom models must be converted—but, we must first freeze the graph to convert the model to the Tensorflow Lite format. This process uses several model formats:

* @{tf.GraphDef} (.pb) —A protobuf that represents the TensorFlow training or computation graph. It contains operators, tensors, and variables definitions.
* *CheckPoint* (.ckpt) —Serialized variables from a TensorFlow graph. Since this does not contain a graph structure, it cannot be interpreted by itself.
* `FrozenGraphDef` —A subclass of `GraphDef` that does not contain variables. A `GraphDef` can be converted to a `FrozenGraphDef` by taking a CheckPoint and a `GraphDef`, and converting each variable into a constant
  using the value retrieved from the CheckPoint.
* `SavedModel` —A `GraphDef` and CheckPoint with a signature that labels input and output arguments to a model. A `GraphDef` and CheckPoint can be extracted from a `SavedModel`.
* *TensorFlow Lite model* (.tflite) —A serialized [FlatBuffer](https://google.github.io/flatbuffers/) that contains TensorFlow Lite operators and tensors for the TensorFlow Lite interpreter, similar to a `FrozenGraphDef`.

### Freeze Graph
### 固化图
To use the `GraphDef` .pb file with TensorFlow Lite, you must have checkpoints that contain trained weight parameters. The .pb file only contains the structure of the graph. The process of merging the checkpoint values with the graph structure is called *freezing the graph*.

You should have a checkpoints folder or download them for a pre-trained model (for example, [MobileNets](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)).

To freeze the graph, use the following command (changing the arguments):

使用如下一些命令来对图进行固化（使用时请修改参数）：
```
freeze_graph --input_graph=/tmp/mobilenet_v1_224.pb \
  --input_checkpoint=/tmp/checkpoints/mobilenet-10202.ckpt \
  --input_binary=true \
  --output_graph=/tmp/frozen_mobilenet_v1_224.pb \
  --output_node_names=MobileNetV1/Predictions/Reshape_1
```

The `input_binary` flag must be enabled so the protobuf is read and written in a binary format. Set the `input_graph` and `input_checkpoint` files.

The `output_node_names` may not be obvious outside of the code that built the model. The easiest way to find them is to visualize the graph, either with [TensorBoard](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2/#3) or `graphviz`.

The frozen `GraphDef` is now ready for conversion to the `FlatBuffer` format (.tflite) for use on Android or iOS devices. For Android, the Tensorflow Optimizing Converter tool supports both float and quantized models. To convert the frozen `GraphDef` to the .tflite format:

```
toco --input_file=$(pwd)/mobilenet_v1_1.0_224/frozen_graph.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --output_file=/tmp/mobilenet_v1_1.0_224.tflite \
  --inference_type=FLOAT \
  --input_type=FLOAT \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1 \
  --input_shapes=1,224,224,3
```

The `input_file` argument should reference the frozen `GraphDef` file containing the model architecture. The [frozen_graph.pb](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz) file used here is available for download. `output_file` is where the TensorFlow Lite model will get generated. The `input_type` and `inference_type` arguments should be set to `FLOAT`, unless converting a @{$performance/quantization$quantized model}. Setting the `input_array`, `output_array`, and `input_shape` arguments are not as straightforward. The easiest way to find these values is to explore the graph using Tensorboard. Reuse the arguments for specifying the output nodes for inference in the `freeze_graph` step.

It is also possible to use the Tensorflow Optimizing Converter with protobufs from either Python or from the command line (see the  [toco_from_protos.py](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/toco/python/toco_from_protos.py) example). This allows you to integrate the conversion step into the model design workflow, ensuring the model is easily convertible to a mobile inference graph. For example:

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
out = tf.identity(val, name="out")

with tf.Session() as sess:
  tflite_model = tf.contrib.lite.toco_convert(sess.graph_def, [img], [out])
  open("converteds_model.tflite", "wb").write(tflite_model)
```

For usage, see the Tensorflow Optimizing Converter [command-line examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/toco/g3doc/cmdline_examples.md).

Refer to the [Ops compatibility guide](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/g3doc/tf_ops_compatibility.md) for troubleshooting help, and if that doesn't help, please [file an issue](https://github.com/tensorflow/tensorflow/issues).

The [development repo](https://github.com/tensorflow/tensorflow) contains a tool to visualize TensorFlow Lite models after conversion. To build the [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/tools/visualize.py) tool:

```sh
bazel run tensorflow/contrib/lite/tools:visualize -- model.tflite model_viz.html
```

This generates an interactive HTML page listing subgraphs, operations, and a graph visualization.


## 3. Use the TensorFlow Lite model for inference in a mobile app
## 3. 在移动 app 中引用 TensorFlow Lite 模型
After completing the prior steps, you should now have a `.tflite` model file.

在完成了上述的步骤之后，你现在应该已经获得了一个 `.tflite` 模型文件了。
### Android
### 安卓
Since Android apps are written in Java and the core TensorFlow library is in C++, a JNI library is provided as an interface. This is only meant for inference—it provides the ability to load a graph, set up inputs, and run the model to calculate outputs.

The open source Android demo app uses the JNI interface and is available [on GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/java/demo/app). You can also download a [prebuilt APK](http://download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk). See the @{$tflite/demo_android} guide for details.

The @{$mobile/android_build} guide has instructions for installing TensorFlow on Android and setting up `bazel` and Android Studio.

### iOS
### iOS
To integrate a TensorFlow model in an iOS app, see the [TensorFlow Lite for iOS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/g3doc/ios.md) guide and @{$tflite/demo_ios} guide.

查看 [TensorFlow Lite for iOS](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/g3doc/ios.md) 指南 和 @{$tflite/demo_ios} 指南，来把 TensorFlow 模型整合进一个 iOS app中。

#### Core ML support
### Core ML 支持
Core ML is a machine learning framework used in Apple products. In addition to using Tensorflow Lite models directly in your applications, you can convert trained Tensorflow models to the [CoreML](https://developer.apple.com/machine-learning/) format for use on Apple devices. To use the converter, refer to the [Tensorflow-CoreML converter documentation](https://github.com/tf-coreml/tf-coreml).

Core ML 是一个用于苹果产品的机器学习框架。除了直接在你的应用中应用 Tensorflow Lite 模型，
### Raspberry Pi
### 树莓派
Compile Tensorflow Lite for a Raspberry Pi by following the [RPi build instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/rpi.md) This compiles a static library file (`.a`) used to build your app. There are plans for Python bindings and a demo app.

依据下述的 [RPi 构建指导](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/rpi.md) 为树莓派编译 Tensorflow Lite 模型。这个操作编译了一个用于构建你 app 的静态库文件(`.a`)。里面包含了一些用于 Python 绑定的计划和一个 demo app.
