# iOS 演示 APP

TensorFlow Lite 示例程序是一款相机应用，通过量子化的 MobileNet 模型来对摄像头后置摄像头所看到的内容进行分类。下面的说明想你展示了如何在 iOS 上构建和运行这个程序。

## 准备

* 你必须安装 [Xcode](https://developer.apple.com/xcode/) 并且具有一个有效的 Apple 开发者 ID，同时还需要一个链接了你开发者账号的 iOS 设备及全部正确的证书。对于这些步骤，我们假设你已经能够载你当前开发环境中编译并将应用部署在一个 iOS 设备上。

* 演示程序需要使用相机，因此必须在一台真实的 iOS 设备上运行。你当然可以构建并使用 iPhone 模拟器运行此程序，但它不会为分类问题提供任何相机画面。

* 你不需要构建整个 TensorFlow 库来运行示例程序，但你需要克隆整个 TensorFlow 仓库：

        git clone https://github.com/tensorflow/tensorflow

* 你还需要安装 Xcode 提供的命令行工具：

        xcode-select --install

    如果你是第一次安装，那么需要先运行一次 Xcode 进行许可，才能继续。

## 构建 iOS 示例应用

1. 如果你没有 CocoaPods 则可以使用下面的命令进行安装:

   ```bash
    sudo gem install cocoapods
   ```

2. 下载示例应用的模型文件（可以通过克隆的目录来完成）：

        sh tensorflow/contrib/lite/examples/ios/download_models.sh

3. 下载 pod 生成的 workspace 文件：

        cd tensorflow/contrib/lite/examples/ios/camera
        pod install

   如果你已经安装了这个 pod 并且上线的命令无效，请尝试

        pod update

   在这个步骤之后，你会具有一个名叫 `tflite_camera_example.xcworkspace ` 的文件。

4. 使用下面的命令在 Xcode 中打开项目：

        open tflite_camera_example.xcworkspace

    如果 `tflite_camera_example` 项目尚未打开则会启动 Xcode。

5. 在 Xcode 中编译运行程序。

    注意，如前文所述，你必须将你的设备链接到一个开发者账户上才能完成设备的部署。

你还需要授权应用具有使用相机的权限。然后你就可以将设备对准各种物体，来欣赏模型如何对物体进行分类了！
