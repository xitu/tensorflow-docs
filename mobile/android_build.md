# 在安卓上构建 TensorFlow

为了让你开始在安卓中使用 TensorFlow， 我们将浏览两种方法来
构建我们的 TensorFlow 移动端的 demo，并且在安卓上部署这些 demo。
第一种方法就是使用 Android Studio，使用 Android Studio 可以让你在 IDE 中。
构建和部署应用。第二种方式是使用 Bazel 和 ADB 在命令行中发布
和构建应用。

我们为什么选择这一种或者另外一种方法去构建呢？

使用 Android Studio 是在 Android 上使用 tensorFlow 最简单的方法。如果你
不准备定制你的 TensorFlow， 或者，如果你想使用 Android Studio 的编辑器
或者其他功能去构建一个 app ，并且，仅仅添加 TensorFlow 
到应用里面，我们推荐使用 Android Studio 。

如果你想使用自定义操作，或者基于其他理由去构建 TensorFlow，
那么你向下滑，并且你将会看到我们关于
[使用 Bazel 构建 demo 的描述文档](#build_the_demo_using_bazel).

## 使用 Android Studio 构建构建 demo 

**准备环境**

如果你没有准备好，你需要做下面两件事情：

- 遵循 Android Studio 网站的说明，安装[Android Studio](https://developer.android.com/studio/index.html)。
  

- 从 Github 上克隆 TensorFlow 的仓库：

        git clone https://github.com/tensorflow/tensorflow

**构建**

1.  打开 Android Studio，在欢迎界面中选择 **Open an existing
   Android Studio project**.

2. 从 **Open File or Project** 窗口中，导航并选择
     `tensorflow/examples/android` 目录，这个目录在你克隆的
    TensorFlow 的 Github repo 中, 点击 OK.

    如果，IDE 请求你去同步 Gradle，点击 OK。

    如果你得到了类似于“Failed to find target with hash string 'android-23”这样的错误，
    你也可能需要安装多个平台和工具。

3. 打开 `build.gradle` 文件 （你可以到侧边面板的 **1.Project** 下，
    并在 Android 下的 **Gradle Script** 中找到它）。 找到
    `nativeBuildSystem` 变量，如果尚未置为 `none`，就把它置为`none`：

        // set to 'bazel', 'cmake', 'makefile', 'none'
        def nativeBuildSystem = 'none'

4. 点击启动按钮（绿色的箭头）或者从顶部菜单使用 **Run -> Run 'android'**。

    如果它请求你使用 Instant Run，点击**Proceed Without Instant Run**。

  你还需要插入一个已经打开了
  开发者选项的  Android 
  设备。 看[这里](https://developer.android.com/studio/run/device.html)你可以
  了解更多关于设置开发者设备的更多细节。

这将会安装三个 app 在你的手机中，这些 app 都是 TensorFlow 的 demo。
看 [Android 示例程序](#android_sample_apps) ，你将得到关于
它们更多的信息。

## 使用 Android Studio 添加 TensorFlow 到你的 app 中

添加 TensorFlow 到你自己的 Android app中，最简单的方法是添加
下面几行到你的 Gradle 构建文件中：

    allprojects {
        repositories {
            jcenter()
        }
	}

    dependencies {
        compile 'org.tensorflow:tensorflow-android:+'
    }

这将自动下载最新的稳定版本的 TensorFlow ARR包，
并且，安装到你的项目中。

##  使用 Bazel 构建 demo

另一种在 Android 中使用 TensorFlow 的方法是
使用 [Bazel](https://bazel.build/) 来构建一个 APK，并且通过
using [ADB](https://developer.android.com/studio/command-line/adb.html) 中加载它。这
需要一些关于构建系统和 Android 开发者工具的知识，但我们会
在这里引导您完成基础步骤。

- 首先, 跟随我们关于 @{$install/install_sources$installing from sources} 的文档。
  它也会带领你安装 Bazel 和克隆
  TensorFlow 的代码。

- 下载 Android 的 [SDK](https://developer.android.com/studio/index.html)
  和 [NDK](https://developer.android.com/ndk/downloads/index.html) 如果你以前
  没下载它们的话。你需要下载最新的 12b 版本的 NDK，和版本为 23 及以上
  的 SDK。

- 在你复制的 TensorFlow 源代码中，更新 
  [WORKSPACE](https://github.com/tensorflow/tensorflow/blob/master/WORKSPACE)
  文件中的 &lt;PATH_TO_NDK&gt;和 &lt;PATH_TO_SDK&gt;，
  为SDK 和 NDK 的位置。

- 运行 Bazel 去构建 demo APK:

        bazel build -c opt //tensorflow/examples/android:tensorflow_demo

- 使用 [ADB](https://developer.android.com/studio/command-line/adb.html#move) to
  安装 APK 文件到你的安卓设备:

        adb install -r bazel-bin/tensorflow/examples/android/tensorflow_demo.apk

注意：当使用 Bazel 编译 Android 你需要在命令行中指定 
`--config=android`。在当前场景中，
p这个例子是专门为 Android 打造的，所以，在这里你不需要指定。

这将安装三个 app 到你的手机，这些 app 都是 TensorFlow 的部分
示例。看 [Android 示例 Apps](#android_sample_apps) 来获取更多关于
示例程序的信息。

## Android 示例 app


[Android 示例代码](https://www.tensorflow.org/code/tensorflow/examples/android/) 是
一个单独的项目，这个项目构建和安装三个示例 app，这些 app 
都使用相同的底层代码。这些示例程序都是用手机的摄像头作为
输入：

- **TF Classify** 使用 Inception v3 模型去标记对象。
  它用 Imagenet 来分类。在 Imagenet 中有 1000 种分类，
  这几乎包含了日常生活中的物品，并且，
  也包括了你在日常生活中不经常遇到东西，所以结果会是十分有趣的。 举个
  例子，这里没有『人』类别，因此，让它猜测照片中的人
 ，它会尽可能去猜测人附近的东西， 例如，安全带
  或者氧气面罩。如果你想定制这个例子去识别你感兴趣的东西，
  你可以使用
 
  [TensorFlow for Poets codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0)
  作为例子， 来了解如何培养模型的基础上你自己的数据。

- **TF Detect** 使用多盒模型去尝试画出在
  相机中人的位置绘制边界框。 这些框对每个侦测结
  果注释了信心表。 结果将不会是完美的， 这类
  的物体侦测仍然是一个活跃的研究话题。这个 demo 也
  包括了可视追踪，当对象在帧之间移动，这比
   TensorFlow 推断的速度要快很多。当明显的帧率加快，
  极大的改善了用户体验，但它还能够估计那些框
  指向帧之间画面的同一对象，这对于随着时间变化
  对对象计数起着非常重要的作用。

- **TF Stylize** 实现了基于摄像头返回的数据的实时风格转化算法。
  你可以选择你想用的风格，
  可以通过屏幕下面的调色板混合它们，
  也可以将处理的分辨率转换为更高或更低。

当你构建和安装这些示例，你将会看到三个 app 的 icon 在你的手机上，
每一个对应一个 demo。点击这些 icon，将会打开对应的 app 并且让
你去探索它们是干什么的。当这些 app 运行的时候，
你可以点击音量增加按钮来在你的屏幕上启动分析统计。

### Android 接口库

因为安卓 app 使用 java 编写，但是，TensorFlow 的核心使用的是 c++，
TensorFlow 有一个 JNI 库来向两种语言之间提供接口。这个接口的只是针对推理的，
因此它提供了加载 graph，设置输入，
和运行模型来计算特殊的输出。您可以在
[TensorFlowInferenceInterface.java](https://www.tensorflow.org/code/tensorflow/contrib/android/java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java)
中查看少量的一组方法的完整文档。

示例程序都是使用这个接口，因此，它们都是学习使用方法的好地方。
您可以在
[ci.tensorflow.org](https://ci.tensorflow.org/view/Nightly/job/nightly-android/)
下载预先构建好的二进制 jar 包。
