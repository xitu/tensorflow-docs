# 在 iOS 中构建 TensorFlow

## 使用 CocoaPods

在 iOS 上开始使用 TensorFlow 最简单的方法是使用 CocoaPods 包管理器。你可以将 `TensroFlow-experimental` 这个 pod 添加到你的 Podfile 中，进而安装通用的二进制框架。它上手简单，但却有一些难以定制的缺点。定制对缩小二进制大小非常重要。如果你确实需要具备自定义库的能力，请查看后面的小节来了解相关方法。

## 创建应用

你若你想在自己的应用中增加 TensorFlow 的能力，那么：

- 在 Xcode 中创建或加载你的应用。

- 将下面的内容添加一个命名为 Podfile 的文件到根目录下：

        target 'YourProjectName'
        pod 'TensorFlow-experimental'

- 运行 `pod install` 来下载并安装 `TensorFlow-experimental` pod。

- 打开 `YourProjectName.xcworkspace` 并添加你自己的代码。

- 在应用的 **Build Settings** 选项中，确保在 **Other Linker Flags** 和 **Header Search Paths**中添加 `$(inherited)` 。

## 运行示例

你需要使用 Xcode 7.3 或更新的版本来运行我们的 iOS 示例程序。

目前有三个例子，分别叫做：simple、benchmark 和 camera。现在你可以通过克隆 TensorFlow 仓库来下载示例代码（我们计划在日后将例子在单独的仓库进行提供）从 TensorFlow 文件夹根目录下载 [Inception v1](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip)，然后将标签和图形文件提取到 data 文件夹中（simple 和 camera 示例步骤一致）：

    mkdir -p ~/graphs
    curl -o ~/graphs/inception5h.zip \
     https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip \
     && unzip ~/graphs/inception5h.zip -d ~/graphs/inception5h
    cp ~/graphs/inception5h/* tensorflow/examples/ios/benchmark/data/
    cp ~/graphs/inception5h/* tensorflow/examples/ios/camera/data/
    cp ~/graphs/inception5h/* tensorflow/examples/ios/simple/data/

切换到其中一个示例的目录，下载 [Tensorflow-experimental](https://cocoapods.org/pods/TensorFlow-experimental) pod，并打开 Xcode workspace。注意，安装 pod 的时间可能会很长（大约 450MB）。如果你想运行 Simple 示例，那么：

    cd tensorflow/examples/ios/simple
    pod install
    open tf_simple_example.xcworkspace   # note .xcworkspace, not .xcodeproj
                                         # this is created by pod install

在 Xcode 模拟器中运行 Simple 示例程序。你可以看到一个有着 **Run Model** 按钮的单屏应用。点击它，你会在下方的日志框中看到一些调试信息，这表明已经分析了目录数据中的 Grace Hopper 图像，并识别出了军装。使用相同的过程可以运行其他的样本。相机示例则需要连接一个真实的 iOS 设备。一旦构建并运行它，就能得到一个实时的相机视图，你便能够将相机对准任何对象从而获得识别的结果。

### iOS 示例细节

有三个 iOS 示例程，均在 Xcode 项目中定义：
[tensorflow/examples/ios](https://www.tensorflow.org/code/tensorflow/examples/ios/).

- **Simple**：这是一个展示了如何在尽可能少的代码下加载和运行 TensorFlow 模型的简单例子。它值包含一个单一视图，并包含一个按钮，用户点击时执行模型的加载和推断。

- **Camera**：这个例子与 Android TF Classify 演示程序非常像。它加载了 Inception v3 并输出了估计出的最佳的标签，以显示实时相机图像中的内容。与 Android 版本一样，你可以使用 TensorFlow for Poets 来训练自定义的模型，并以最小的代码修改将其放入此示例中。

- **Benchmark**：它与 Simple 很接近，但它会重复运行计算图并将类似的统计信息输出到 Android 上的基准测试工具中。


### 排错

- 确保你使用的是 TensorFlow-experimental pod（而不是 TensorFlow）。

- TensorFlow-experimental pod 大约有 450MB。原因在于我们绑定了多个平台，而 pod 包含了所有平台的 TensorFlow 功能（例如：运算）。最终应用的大小很小（约为 25 MB）。使用完整的 pod 在自己的开发过程中非常方便，但请阅读下一节内容来交接如何通过构建自己定制的 TensorFlow 库来缩减大小。

## 从源码构建 TensorFlow iOS 库

尽管 Cocoapods 是最简单快捷的入门方式，但你有时候需要更加灵活的确定你的应用程序需要附带那些 TensorFlow 组件。对于这种情况，你可以从源码构建 iOS 库，请参考[这篇教程](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/ios#building-the-tensorflow-ios-libraries-from-source)来了解相关操作的详细说明。
