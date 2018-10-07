# 集成 TensorFlow 库

一旦你在解决你所需问题的模型上取得了进展，那么在立即在应用中进行测试就变得非常重要了。通常情况下，你的训练数据与实际世界中面临的数据是存在意想不到的差异，尽快清晰的了解这些差距，才能更快的改进你的产品体验。这个页面讨论了如何在你的应用中集成 TensorFlow 库，只要你能够将 TensorFlow 移动端的演示程序成功部署，就能成功的构建你自己的应用。

## 库链接

在你尝试构建这些例子之前，你将需要从一个现有的应用程序中调用 TensorFlow。最简单的方法就是使用 @{$mobile/ios_build#using_cocoapods$here} 描述的使用 Pod 的安装步骤。但是，如果你想要使用源码来安装 TensorFlow（例如包含一些自定义的运算符），那么你需要将 Tensorflow 以框架的形式引入，包含正确的头文件，链接到构建所需的库文件和依赖项。

### Android

对 Android 而言，你只需要链接一个叫做 `libandroid_tensorflow_inference_java.jar` 的 JAR 文件即可。有三种方式：

1. 在 jcenter AAR 中引入，例如 [这个应用](https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/android/tfmobile/build.gradle#L59-L65)

2. 从 [ci.tensorflow.org](http://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/) 中下载编译好的开发版本。

3. 根据我们 [Android GitHub 仓库](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android)的指示自行构建 JAR 文件。

### iOS

在 iOS 上集成 TensorFlow 库稍微复杂一些。这是一份你需要在 iOS 应用上执行的步骤清单：

- 链接 `tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a`：通常情况下，将 `-L/your/path/tensorflow/contrib/makefile/gen/lib/` 和 `-ltensorflow-core` 添加到你的链接器标志中。
- 链接并生成 protobuf 库：将 `-L/your/path/tensorflow/contrib/makefile/gen/protobuf_ios/lib`、`-lprotobuf` 和 `-lprotobuf-lite` 添加到你的编译命令中。
- 包含路径：你需要将 `tensorflow/contrib/makefile/downloads/protobuf/src`、 `tensorflow/contrib/makefile/downloads`、`tensorflow/contrib/makefile/downloads/eigen` 和 `tensorflow/contrib/makefile/gen/proto` 这些 TensorFlow 的源文件夹路径添加作为第一入口。
- 确保针对 TensorFlow 的库二进制文件是通过 `-force_load` 参数编译而成（或视平台而定），从而保证正确链接。关于这个操作必要性的更多细节你可以在下一个小节，[全局构造的黑魔法](#全局构造的黑魔法)中了解到。在类 Linux 平台下，你需要使用诸如 `-Wl,--allow-multiple-definition -Wl,--whole-archive` 等不同链接标志。

此外，你还需要将其链接到 Accelerator 框架中，因为它能够对某些计算操作进行加速。

## 全局构造的黑魔法

当你运行程序尝试调用 TensorFlow 时候，你可能会遇到 `No session factory registered for the given session options` 的错误，同时它也是几个相当微妙的错误之一。要理解为什么会发生这种情况以及如何解决这个问题，你需要了解一下 TensorFlow 的架构。

TensorFlow 整个框架被设计得相当模块化，其中包含了大量的独立的特定对象以及一个轻薄的内核，并且可以根据需要进行混合与匹配。为了实现这一点，C++ 中的编码模式必须让模块在没有一个汇总列表的情况下（且每个列表与实现必须分开更新），能够简单的通知框架它们所提供的服务。同时，它还必须允许单独的库能够在不重新编译内核的情况下添加它们自己的实现。

为了获得这种级别的兼容性，Tensorf 在相当多的地方使用了如下的注册模式：

    class MulKernel : OpKernel {
      Status Compute(OpKernelContext* context) { … }
    };
    REGISTER_KERNEL(MulKernel, “Mul”);

这将作为主要内核集的一部分或是作为单独的自定义库，在一个独立的 `.cc` 文件链接到你的应用中去。黑魔法就在于，`REGISTER_KERNEL()`  这个宏能能够通知 TensorFlow 内核它具有一个关于 Mul 操作的实现，从而在任何需要它的计算图中调用。

从编程的角度来看，这个设置是相当便利的。实现和注册代码位于同一个文件，同时添加新的实现与编译和链接它一样简单。但是，最难的地方就在于 `REGISTER_KERNEL()` 的实现方式。C++ 并没有提供这种良好的注册机制，因此我们必须使用一些具有技巧的代码。在 TensorFlow 内部，这个宏被实现为类似于下面代码的东西：

    class RegisterMul {
     public:
      RegisterMul() {
        global_kernel_registry()->Register(“Mul”, [](){
          return new MulKernel()
        });
      }
    };
    RegisterMul g_register_mul;

这个宏设置了一个具有构造函数的 `RegisterMul` 类，其构造函数会在有人希望全局内核入口创建一个 `Mul` 内核的时候告诉这个入口。从而，内核入口类就具有一个全局对象，并且其构造函数需要在任何程序启动前调用它。

虽然听起来很合理，但可惜的是自定义的全局对象没有被任何其他代码使用，而链接器又被设计成在没有使用时就会将其代码删除的形式，从而导致的结果就是：构造函数从未被调用，并且该类也没有被注册。在 TensorFlow 中，所有模块都使用了这种模式，而在代码运行时，`Session` 的实现中就首次检查了这个构造，这也是为什么这个问题会发生的原因。

解决方法就是强制链接器在即使代码没有使用的情况下，也不忽略库中的任何代码。在 iOS 中，可以中 `-force_load` 标志并制定库的路径，而在 Linux 中，你需要使用 `--whole-archive`。它们指导了链接器不要积极的对编译作出精简，而是保留使用 TensorFlow 时所需的全局变量。

不同形式 `REGISTER_*` 宏的实际实现在实践中相当复杂，但它们都有着相同的底层问题。如果你对它们的工作方式感兴趣，[op_kernel.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/op_kernel.h#L1091) 是一个研究的起点。

## Protobuf 问题

TensorFlow 依赖 [Protocol Buffer](https://developers.google.com/protocol-buffers/)，通常称为 protobuf。这个库利用了数据结构的定义，从而为各种语言生成可访问的代码。比较棘手的问题在于，生成的代码需要链接到与框架完全相同版本共享的库中，才能作为生成器使用。当 `protoc` 来自于标准链接库中不同版本的 protobuf 库路径时候，会触发一些问题。例如，你可能正在使用一个本地编译在 `~/projects/protobuf-3.0.1.a` 中的 `protoc` 的副本，但是你又有在系统中安装在 `/user/local/lib` 和 `/usr/local/include` 下的 3.0.0 版本的 `protoc`。

在使用 protobuf 进行编译或链接时，这个问题就会导致出错。通常，构建工具会照顾到这一点，但是如果你使用的是 makefile，那么请确保你构建时使用的是局部构建的 protobuf 库，可以参考[这个 Makefile](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/makefile/Makefile#L18)。

另一个可能出现问题的情况是在需要生产 protobuf 头文件和源文件时出现的。这个过程使得构建过程更加复杂，这是因为第一阶段必须通过 protobuf 的定义来创建所有需要的代码文件，只有在此之后才能继续编译库的代码。

### 同应用下的不同 protobuf 版本

Protobufs 生成了整个 TensorFlow C++ 接口头文件的一部分。这使得使用这个库作为一个独立框架变得相对复杂。

如果你的应用已经使用了某个版本的 protobuf，那么在集成 TensorFlow 可能会会遇到一些麻烦，因为 TensorFlow 可能要求使用另一个版本的 protobuf。如果你尝试将两个版本链接到一个相同的库时，你将看到大堆的符号错误。为了解决这个特定的问题，我们有一个实验性的脚本 [rename_protobuf.sh](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/makefile/rename_protobuf.sh) 来帮你解决这个问题。

在下载完全部依赖后，你需要将其作为 Makefile 构建的一部分：

    tensorflow/contrib/makefile/download_dependencies.sh
    tensorflow/contrib/makefile/rename_protobuf.sh

## 调用 TensorFlow API

一旦你的框架可用后，你就得调用它了。通常的模式是先加载代表预先设置加载你的模型，它表示了一个数值计算的模型，然后通过输入（例如相机的图像）、运行该模型从而接收输出（例如预测标签）。

在 Android 上，我们提供了专用于 Java 的 Inference 库，而在 iOS 和 Raspberry Pi 上你可以直接调用 C++ 的 API。

### Android

Android 上的一个典型的 Inference 库的用法如下：

```Java
// 从磁盘加载模型
TensorFlowInferenceInterface inferenceInterface =
new TensorFlowInferenceInterface(assetManager, modelFilename);

// 将输入数据复制给 TensorFlow
inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);

// 调用运行 Inference 程序
inferenceInterface.run(outputNames, logStats);

// 将输出的 Tensor 复制回输出 outputs 数组
inferenceInterface.fetch(outputName, outputs);
```

你可以在这个 [Android 示例](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java#L107)中找到相关源码。

### iOS 与 Raspberry Pi

在 iOS 和 Raspberry Pi 中也有类似的代码：

```c++
// 加载模型
PortableReadFileToProto(file_path, &tensorflow_graph);

// 从模型中创建 session 会话
tensorflow::Status s = session->Create(tensorflow_graph);
if (!s.ok()) {
  LOG(FATAL) << "Could not create TensorFlow Graph: " << s;
}

// 运行模型
std::string input_layer = "input";
std::string output_layer = "output";
std::vector<tensorflow::Tensor> outputs;
tensorflow::Status run_status = session->Run({{input_layer, image_tensor}},
                           {output_layer}, {}, &outputs);
if (!run_status.ok()) {
  LOG(FATAL) << "Running model failed: " << run_status;
}

// 访问输出数据
tensorflow::Tensor* output = &outputs[0];
```

上面的代码基于 [iOS 示例代码](https://www.tensorflow.org/code/tensorflow/examples/ios/simple/RunModelViewController.mm)，但是其实与 iOS 并没有关系；相同的代码同样可以在任何支持 C++ 的平台上运行。

你可以在[这里](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/pi_examples/label_image/label_image.cc)找到与 Raspberry Pi 相关的例子。
