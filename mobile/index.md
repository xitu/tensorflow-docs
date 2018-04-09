# 概述

TensorFlow 被设计用来给移动平台，提供好的解决方案。当前我们在移动平台和嵌入式平台上发布机器学习应用有两个解决方案：@{$mobile/mobile_intro$TensorFlow for Mobile} 和 @{$mobile/tflite$TensorFlow Lite}。

## TensorFlow Lite 与 TensorFlow mobile 

这里有一些两者的不同点：

- TensorFlow Lite 是 TensorFlow mobile 的进化版。在大多数情况下，使用 TensorFlow Lite版本开发 app 将会有更小的二进制文件，更少的依赖，和更优秀的性能。

- TensorFlow Lite 现在还是开发者预览版本，所以，并未涵盖所有用例。我们希望你能在生产用例中使用 TensorFlow mobile。

- TensorFlow Lite 默认仅仅支持有限的操作集合。并不是所有的模型都适用。 TensorFlow 的移动版本，提供所有支持的功能。

TensorFlow Lite 在移动平台，提供更好的性能和更小的二进制文件，并且可以在支持硬件加速的平台上使用硬件加速。另外，它还有更少的依赖，它可以更简单的构建和托管，能够支持更多受限设备的应用场景。 TensorFlow Lite 也允许通过 [Neural NetworksAPI](https://developer.android.com/ndk/guides/neuralnetworks/index.html)来实现目标加速。

TensorFlow Lite 目前涵盖了有限的操作，虽然，TensorFlow for mobile 默认支持受限的操作集合，但原则上，如果你要使用任意的 TensorFlow 操作符，可以通过定制内核来构建。这种用例 TensorFlow Lite 当前不支持，需要继续使用 TensorFlow mobile。随着 TensorFlow Lite 的发展，它将添加更多的操作，而且这个决定很容易去做。
