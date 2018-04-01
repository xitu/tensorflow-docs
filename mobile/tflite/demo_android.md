# Android 的 TensorFlow Lite Demo

TensorFlow Lite 的 demo 是一个摄像头 app，这个 app 持续分类，
它从后面摄像头拍摄到的任何东西。使用了量化移动网络模型。

你需要一个安卓设备，并且这个设备的安卓版本号要高于5.0，才能运行这个 demo。

为了让你在安卓设备上使用 TensorFlow Lite，我们将带领你
在 Android studio 上构建和发布我们的 TensorFlow demo app。

注意:更多的细节指南，请看 
[TFLite Codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/index.html#0)

当然，这也可以使用 Bazel 去构建 demo app，但是，我们仅仅推荐
进阶的十分熟悉 Bazel 构建环境的用户这样做。
关于使用 Bazel 更多的信息， 请看我们在 [Github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite#building-tensorflow-lite-and-the-demo-app-from-source) 上的页面。

## 在 Android Studio 上构建和发布

1. 从 Github 上克隆 TensorFlow 的储存库，如果你以前没克隆过：

        git clone https://github.com/tensorflow/tensorflow

2. 从[这里](https://developer.android.com/studio/index.html)下载安装最新的 Android Studio。

3. 从 **Android Studio 欢迎** 页面, 选择 **Import Project
   (Gradle, Eclipse ADT, etc)** 选项去作为一个
   已存在的 Android Studio 项目，导入
   `tensorflow/contrib/lite/java/demo`。

    Android Studio可能提示你安装 Gradle的升级和其他版本工具
    升级；你需要接受这些升级。

4. 从[这里]((https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip))下载 TensorFlow Lite 移动网络模型。

    解压复制 `mobilenet_quant_v1_224.tflite` 文件到资源
    目录： `tensorflow/contrib/lite/java/demo/app/src/main/assets/`

5. 在 Android Studio 中构建和测试 app。

你将准许 app 使用设备摄像头的权限。 
将相机指向各种物体，并欣赏模型如何对物体进行分类！
