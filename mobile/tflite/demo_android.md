# Android 示例应用

该 TensorFLow Lite 示例可以在 [GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/java/demo) 上被找到。这是一个使用量化的 MobileNet 模型或是浮点 Inception-v3 模型对图片进行持续分类的相机应用。示例的最低运行要求是 Android 5.0（API 21）。

在示例中，应用会使用 TensorFlow Lite Java API 来预测。应用会为每一帧都进行实时分类，并将可能性最高的类别和检测对象的时间一同显示出来。

有三种方式获取示例应用：

* 下载[预编译 APK](http://download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk)。
* 使用 Android Studio 编译应用。
* 下载 TensorFlow Lite 和这个示例应用的源码，然后用 bazel 编译。

## 下载预编译版本

尝试这个示例最简单的方法是下载[预编译 APK](https://storage.googleapis.com/download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk)。

安装完 APK 后，双击应用图标启动程序。当程序第一次运行时，会请求运行时获取设备摄像头的权限。程序会打开设备的后摄像头，并识别视野内的物体。在图像的底部（如果是全景模式则是图像的左边）会展示可能性最高的三个物体和其可能的分类。

## 在 Android Studio 中用 JCenter源的 TensorFlow Lite AAR 编译

使用 Android Studio 来尝试修改代码并编译：

* 安装最新版本的 [Android Studio](https://developer.android.com/studio/index.html)。
* 确保你的 Android SDK 版本高于 26 且 NDK 版本高于 14（在 Android Studio 设置里面）。
* 将 `tensorflow/contrib/lite/java/demo` 目录作为一个新的 Android Studio 项目导入。
* 安装需要的 Gradle 插件。

现在你可以构建并运行演示程序。

构建过程会下载已量化的 [Mobilenet TensorFlow Lite 模型](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip)，并将其解压到 assets 目录下：`tensorflow/contrib/lite/java/demo/app/src/main/assets/`。

更多的详细细节可在 [TF Lite Android App 页面](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/java/demo/README.md)中查看。

### 使用其它模型

要使用其它模型：

* 下载浮点模型 [Inception-v3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_slim_2016_android_2017_11_10.zip)。
* 解压并拷贝 `inceptionv3_non_slim_2015.tflite` 到 assets 目录。 
* 变更 [Camera2BasicFragment.java](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/Camera2BasicFragment.java)<br> 中特定的分类器
  由：`classifier = new ImageClassifierQuantizedMobileNet(getActivity());`<br>
  改为：`classifier = new ImageClassifierFloatInception(getActivity());`。

## 使用源码编译 TensorFlow Lite 和示例应用

### 克隆 TensorFlow 仓库

```sh
git clone https://github.com/tensorflow/tensorflow
```

### 安装 Bazel

如果你的电脑上没有安装 `bazel`，查看 [安装 Bazel](https://bazel.build/versions/master/docs/install.html)。

注意：Bazel 现在并不支持在 Windows 上进行 Android 编译。Windows 用户可以下载 [预编译版本](https://storage.googleapis.com/download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk)。

### 安装 Android NDK 和 SDK

编译 TensorFlow Lite 的原生（C/C++）代码需要用到 Android NDK。当前推荐版本为 *14b*，可以在 [NDK 存档](https://developer.android.com/ndk/downloads/older_releases.html#ndk-14b-downloads)
上找到。

Android SDK 和编译工具可以[单独下载](https://developer.android.com/tools/revisions/build-tools.html) 或者配合 [Android Studio](https://developer.android.com/studio/index.html) 使用。编译 TensorFlow Lite Android 示例应用推荐使用 API >= 23 的编译工具（但是在 API >= 21 的版本上均可运行）。

在 TensorFlow 仓库的根目录下，更新 `WORKSPACE` 文件中的 `api_level` 以及 SDK 和 NDK 的位置。如果你是通过 Android Studio 安装的，可以在 SDK 管理器中查看 SDK 路径。默认的 NDK 路径为：`{SDK path}/ndk-bundle`。如下所示：

```
android_sdk_repository (
    name = "androidsdk",
    api_level = 23,
    build_tools_version = "23.0.2",
    path = "/home/xxxx/android-sdk-linux/",
)

android_ndk_repository(
    name = "androidndk",
    path = "/home/xxxx/android-ndk-r10e/",
    api_level = 19,
)
```

在 [TF Lite Android 应用页面](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/java/demo/README.md)中可以找到更多信息。

### 编译源码

运行 `bazel` 来编译源码：

```
bazel build --cxxopt=--std=c++11 //tensorflow/contrib/lite/java/demo/app/src/main:TfLiteCameraDemo
```

警告：因为一个已知的 bazel 的 bug，我们只能在 Python 2 环境下编译 Android 示例应用。


## 关于示例

示例应用会缩放每一帧相机获取的图像（224 宽 * 224 高）来匹配量化的 MobileNets 模型（Inception-v3 是 299 * 299）。缩放后的图像被逐行放进[缓冲区](https://developer.android.com/reference/java/nio/ByteBuffer.html)。它的大小是 1 * 224 * 224 * 3 字节，其中 1 代表该批次中图像的数量。224 * 224（299 * 299）是图像的宽和高。3 字节代表一个像素有 3 种颜色。

示例中使用了单进单出的 TensorFlow Lite Java inference API。它输出一个二维数组，第一个维度表示类别索引，第二个维度表示分类的置信度。 两种模型都有 1001 种不同的类别，应用将所有目录的可能性排序，并显示可能性最高的三种。模型文件必须被下载下来并打包到应用的资源目录。
