# 在安卓上构建 TensorFlow

为了让你开始在安卓中使用 TensorFlow， 我们将浏览两种方法来构建我们的 TensorFlow 移动端的 demo，并且在安卓上部署这些 demo，第一种方法就是使用 Android Studio，使用 Android Studio 可以让你在 IDE 中。构建和部署应用。第二种方式是使用 Bazel 和 ADB 在命令行中发布和构建应用。

如何决定用哪一种方法呢？

在 Android 上使用 TensorFlow 最简单的方式是通过 Android Studio 来构建。如果你不准备定制你的 TensorFlow， 或者，如果你想使用 Android Studio 的编辑器或者其他功能去构建一个 app ，并且，仅仅添加 TensorFlow 到应用里面，我们推荐使用 Android Studio 。如果你想使用自定义操作，或者基于其他理由去重新构建 TensorFlow，那么你向下滑参考[使用 Bazel 构建 demo](#使用-bazel-构建-demo) 一节。

## 使用 Android Studio 构建构建 demo 

**准备环境**

如果你没有准备好，你需要做下面两件事情：

- 遵循 Android Studio 网站的说明，安装[Android Studio](https://developer.android.com/studio/index.html)。
  

- 从 GitHub 上克隆 TensorFlow 的仓库：

        git clone https://github.com/tensorflow/tensorflow

**构建**

1.  打开 Android Studio，在欢迎界面中选择 **Open an existing Android Studio project**。

2. 从 **Open File or Project** 窗口中，切换文件目录并选择`tensorflow/examples/android` 目录，这个目录在你克隆的 TensorFlow 的 GitHub repo 中, 点击 OK。

    如果，IDE 需要去同步 Gradle，点击 OK。

    如果你得到了类似于“Failed to find target with hash string 'android-23‘”这样的错误，你也可能需要安装多个平台和工具。

3. 打开 `build.gradle` 文件 （你可以到侧边面板的 **1:Project** 下，并在 Android 下的 **Gradle Script** 中找到它）。 找到`nativeBuildSystem` 变量，如果尚未置为 `none`，就把它置为`none`：

        // set to 'bazel', 'cmake', 'makefile', 'none'
        def nativeBuildSystem = 'none'

4. 点击 *Run* 按钮（绿色的箭头）或从顶部菜单选择 *Run > Run 'android'* 。你可能需要使用 *Build > Rebuild Project* 来重建项目。

如果它请求你使用 Instant Run，点击**Proceed Without Instant Run**。

  你还需要插入一个已经打开了开发者选项的 Android 设备。看[这里](https://developer.android.com/studio/run/device.html)你可以了解更多关于设置开发者设备的更多细节。

这将会安装三个 app 在你的手机中，这些 app 都是 TensorFlow 的 demo。看 [Android 示例程序](#android-示例-app) ，你将得到关于它们更多的信息。

## 使用 Android Studio 添加 TensorFlow 到你的 app 中

添加 TensorFlow 到你自己的 Android app 中，最简单的方法是添加下面几行到你的 Gradle 构建文件中：

    allprojects {
        repositories {
            jcenter()
        }
	}

    dependencies {
        compile 'org.tensorflow:tensorflow-android:+'
    }

这将自动下载最新的稳定版本的 TensorFlow ARR 包，并安装到你的项目中。

##  使用 Bazel 构建 demo

另一种在 Android 中使用 TensorFlow 的方法是使用 [Bazel](https://bazel.build/) 来构建一个 APK，并且通过 [ADB](https://developer.android.com/studio/command-line/adb.html) 来加载它。这需要一些关于构建系统和 Android 开发者工具的知识，但我们会在这里引导你完成基础步骤。

- 首先参考[通过源码安装 TensorFlow](./install_sources.md) 的文档。它也会带领你安装 Bazel 和克隆TensorFlow 的代码。

- 下载 Android 的 [SDK](https://developer.android.com/studio/index.html)和 [NDK](https://developer.android.com/ndk/download/index.html) 如果你以前没下载它们的话。你需要下载最新的 12b 版本的 NDK，和版本为 23 及以上的 SDK。

- 在你复制的 TensorFlow 源代码副本中，更新 [WORKSPACE](https://github.com/tensorflow/tensorflow/blob/master/WORKSPACE) 文件中的 &lt;PATH_TO_NDK&gt;和 &lt;PATH_TO_SDK&gt;。分别为 SDK 和 NDK 的位置。

- 运行 Bazel 去构建 demo APK:

        bazel build -c opt //tensorflow/examples/android:tensorflow_demo

- 使用 [ADB](https://developer.android.com/studio/command-line/adb.html#move) 安装 APK 文件到你的安卓设备:

        adb install -r bazel-bin/tensorflow/examples/android/tensorflow_demo.apk

注意：当使用 Bazel 编译 Android 你需要在命令行中指定`--config=android`。在当前场景中，这个例子是专门为 Android 打造的，所以，在这里你不需要指定。

这将安装三个 app 到你的手机，这些 app 都是 TensorFlow 的部分示例。看 [Android 示例 Apps](#android-示例-app) 来获取更多关于示例程序的信息。

## Android 示例 app


[Android 示例代码](https://www.tensorflow.org/code/tensorflow/examples/android/) 是一个单独的项目，这个项目构建和安装三个示例 app，这些 app 都使用相同的底层代码。这些示例程序都是用手机的摄像头作为输入：

- **TF Classify** 标记指向的对象用的是 Inception v3 模型，并用 Imagenet 来分类。在 Imagenet 中有 1000 种分类，它会遗漏大多数日常物品，并且，也包括了你在日常生活中不经常遇到东西，所以结果会是十分有趣的。 举个例子，这里没有『人』类别，因此，让它猜测照片中的人，它会尽可能去猜测人附近的东西， 例如，安全带或者氧气面罩。如果你想定制这个例子去识别你感兴趣的东西，你可以使用 [TensorFlow for Poets codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0) 作为例子，基于自己的数据训练模型。

- **TF Detect** 使用多盒模型去尝试画出在人在相机中位置的边界框。这些框对每个侦测结果注释了置信度。结果还不是完美的， 这类的物体侦测仍然是一个活跃的研究话题。这个 demo 也包括了可视追踪，当对象在帧之间移动，这比 TensorFlow 推断的速度要快很多。因为帧率明显加快，可以显著提高用户体验，而且这也能够计算出不同帧中指向相同对象的框，这对在一段时间内统计对象个数起着非常重要的作用。

- **TF Stylize** 实现了基于摄像头返回的数据的实时风格转化算法。你可以选择你想用的风格，可以通过屏幕下面的调色板混合它们，也可以将处理的分辨率转换为更高或更低。

当你构建和安装这些示例，你将会看到三个 app 的 icon 在你的手机上，每一个对应一个 demo。点击这些 icon，将会打开对应的 app 并且让你去探索它们是干什么的。当这些 app 运行的时候，你可以点击音量增加按钮来在你的屏幕上启动分析统计。

### Android 接口库

因为安卓 app 使用 java 编写，但是，TensorFlow 的核心使用的是 c++，TensorFlow 有一个 JNI 库来向两种语言之间提供接口。这个接口的只是针对推理的，因此它提供了加载 graph，设置输入，和运行模型来计算特殊的输出。你可以在 [TensorFlowInferenceInterface.java](https://www.tensorflow.org/code/tensorflow/contrib/android/java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java) 中查看少量的一组方法的完整文档。

示例程序都是使用这个接口，因此，它们都是学习使用方法的好地方。你可以在 [ci.tensorflow.org](https://ci.tensorflow.org/view/Nightly/job/nightly-android/) 下载预先构建好的二进制 jar 包。
