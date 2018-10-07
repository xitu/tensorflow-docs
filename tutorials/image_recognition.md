# 图像识别

我们的大脑视觉成像似乎很容易。区分狮子和美洲豹，识别路标或者辨认人脸这些行为对于人来说都是小菜一碟。但是对于计算机来说有些问题真的太难解决了：但是这仅仅是因为我们的大脑在识别图像这方面实力超群。

在过去的几年中，机器学习领域在解决这些困难的问题方面取得了巨大的进步。尤其是，我们已经发现了一种叫做深度 [卷积神经网络](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/) 的模型，它在强图像识别任务上的表现已经非常可观--即在一些领域已经有了相当或超过人类的表现。

计算机视觉的研究人员将他们的成果和 [ImageNet](http://www.image-net.org)（一个计算机视觉的理论基准测试程序）进行校验和对抗，结果表明他们已经取得了稳定的进步。这些有继承关系的模型持续的展示着它们的进步，并且每次都会产生新的成果：[QuocNet]，[AlexNet]，[Inception (GoogLeNet)]，[BN-Inception-v2]。Google 内部和外部的研究人员也都发表了一些论文来描述所有的这些模型，但是成果仍然很难再现。我们接下来要做的就是运行我们最新的图像识别模型--[Inception-v3]。

[QuocNet]：https://static.googleusercontent.com/media/research.google.com/en//archive/unsupervised_icml2012.pdf
[AlexNet]：https://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
[Inception (GoogLeNet)]：https://arxiv.org/abs/1409.4842
[BN-Inception-v2]：https://arxiv.org/abs/1502.03167
[Inception-v3]：https://arxiv.org/abs/1512.00567

Inception-v3 从 2012 年就开始使用数据针对 [ImageNet] 这个大型视觉挑战任务训练了。将所有的图片分成像『斑马』，『达尔马西亚狗』，『洗碗工』等 [1000 个类别] 是计算机视觉领域的一个标准任务，例如下面这些图片就是 [AlexNet] 模型分类的结果：

<div style="width:50%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="https://www.tensorflow.org/images/AlexClassification.png">
</div>

为了和其他模型进行比较，我们通过检查，将前五次测试中模型未能正确预测的频率称作--"前五误差率"。[AlexNet] 在 2012 年验证数据集上的前五误差率是 15.3%；[Inception (GoogLeNet)] 是 6.67%；[BN-Inception-v2] 是 4.9%；[Inception-v3] 则达到了 3.46%。

> 人类在 ImageNet 挑战上的表现如何呢？这里有一篇 Andrej Karpathy 写的 [博客]。他自己的前五误差率是 5.1%。

[ImageNet]: http://image-net.org/
[1000 classes]: http://image-net.org/challenges/LSVRC/2014/browse-synsets
[博客]: https://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/

本文将会教你如何使用 [Inception-v3]。你将学习到如何使用 Python 或 C++ 把图片分成 [1000 种类别]。同时，我们也会讨论如何从这个可以用于其他视觉任务的模型中提取出更高层的特征。

让我们来看一看社区是如何使用这个模型的。


## Python API 的使用

`classify_image.py` 这个程序在第一次运行的时候会从 `tensorflow.org` 上下载训练好的模型。你需要保证你的硬盘有 200M 的可用空间。

从 clone [TensorFlow models repo](https://github.com/tensorflow/models) 这个项目开始。运行下面的命令：

    cd models/tutorials/image/imagenet
    python classify_image.py

上面的命令将会对提供的一张熊猫图片进行分类。

<div style="width:15%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/cropped_panda.jpg">
</div>

如果模型运行正常，则会输出下面的信息：

    giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (score = 0.88493)
    indri, indris, Indri indri, Indri brevicaudatus (score = 0.00878)
    lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens (score = 0.00317)
    custard apple (score = 0.00149)
    earthstar (score = 0.00127)

如果你想使用 JPEG 类型的图片，那么你需要编辑 `--image_file` 这个参数。

> 如果你下载的模型数据在另一个目录，那么你需要通过 `--model_dir` 来指定那个目录。

## C++ API 的使用

你可以在生产环境运行 C++ 版本的 [Inception-v3] 模型。你还可以下载包含 GraphDef 的归档，GraphDef 可以像这样定义模型（在 TensorFlow 的根目录下运行）：

```bash
curl -L "https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz" |
  tar -C tensorflow/examples/label_image/data -xz
```

下一步我们需要编译包含载入和运行 graph 代码的 C++ 库。如果你按照[下载 TensorFlow 源代码安装的说明](../../install/source.md)一步步进行，你应该就可以通过在你的终端中运行下面的的命令来编译示例了：

```bash
bazel build tensorflow/examples/label_image/...
```

它会创建一个二进制的可执行文件，你可以像这样来运行它：

```bash
bazel-bin/tensorflow/examples/label_image/label_image
```

这里使用默认的图像框架所附带的示例，将会输出类似下面的内容：

```
I tensorflow/examples/label_image/main.cc:206] military uniform (653): 0.834306
I tensorflow/examples/label_image/main.cc:206] mortarboard (668): 0.0218692
I tensorflow/examples/label_image/main.cc:206] academic gown (401): 0.0103579
I tensorflow/examples/label_image/main.cc:206] pickelhaube (716): 0.00800814
I tensorflow/examples/label_image/main.cc:206] bulletproof vest (466): 0.00535088
```
在这种情况下，我们使用默认图片 [Admiral Grace Hopper](https://en.wikipedia.org/wiki/Grace_Hopper)，你可以看到网络使用 0.8 的高分正确的标识出了她穿的是军装。

<div style="width:45%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/grace_hopper.jpg">
</div>

下面你可以通过 --image= argument 来检测一张自己的图片试试，例如：

```bash
bazel-bin/tensorflow/examples/label_image/label_image --image=my_image.png
```

如果你仔细浏览 [`tensorflow/examples/label_image/main.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc)这个文件，你可以看到它是如何工作的。我们希望这部分代码会帮助你将 TensorFlow 整合到你自己的应用中，所以我们会通过这些主要的函数一步一步的向你展示：

`input_mean` from each pixel value, then divide it by `input_std`.
命令行参数控制着文件从哪里载入以及输入图片的属性。模型希望得到的是 299x299 的 RGB 图片，所以有 `input_width` 和 `input_height` 参数。同时我们也需要将 0 到 255 的整型像素值缩放成 graph 操作的浮点数值。我们通过控制 `input_mean` 和 `input_std` 这两个参数来控制缩放的比例：首先从每一个像素值中减去 `input_mean` 这个值，然后再除以 `input_std`。

这些数值是不是看起来很神奇，其实它们都只是原始模型的作者基于他或者她训练模型使用的输入图片设定好的值。如果你有一个你自己训练的 graph，那么你就需要在训练过程中调节这些值来匹配你使用过的值。

你可以看到 [`ReadTensorFromImageFile()`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc#L88) 这个函数是如何应用在一张图片上的。

```C++
// 给定一张图片的文件名，读取它的数据，接着按照图片来解码，
// 缩放成我们需要的大小，然后按比例转换成我们想要的值。
Status ReadTensorFromImageFile(string file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors) {
  tensorflow::GraphDefBuilder b;
```
让我们先创建一个 `GraphDefBuilder`，`GraphDefBuilder` 是一个
可以用来指定一个将要运行或加载的模型的对象。

```C++
  string input_name = "file_reader";
  string output_name = "normalized";
  tensorflow::Node* file_reader =
      tensorflow::ops::ReadFile(tensorflow::ops::Const(file_name, b.opts()),
                                b.opts().WithName(input_name));
```
然后创建我们想要运行和加载的小模型的节点，重新调整大小并把像素值缩放成主模型想要的数据。我们创建的第一个节点仅仅是一个 `Const` 操作，它保存着我们想要载入的图片的文件名的 tensor。然后传给 `ReadFile` 这个操作当做第一个输入。或许你会注意到我们把 `b.opts()` 这个参数当做最后一个参数传递给所有的创建函数。这个参数可以确保节点被添加到 `GraphDefBuilder` 定义的模型中。
同时我们也通过 `b.opts()` 调用 `WithName()` 来给 `ReadFile` 这个操作命名。这个操作给了节点一个名字，当然了，这个操作其实并不是强制的，即使如果你不这样做，程序也会自动分配一个名字，但是这样不利于调试。


```C++
  // 现在让我们试着搞清楚它的文件类型，并解码。
  const int wanted_channels = 3;
  tensorflow::Node* image_reader;
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = tensorflow::ops::DecodePng(
        file_reader,
        b.opts().WithAttr("channels", wanted_channels).WithName("png_reader"));
  } else {
    // 如果它不是 PNG，那么就一定是 JPEG 了。
    image_reader = tensorflow::ops::DecodeJpeg(
        file_reader,
        b.opts().WithAttr("channels", wanted_channels).WithName("jpeg_reader"));
  }
  // 现在将图片数据转换成浮点型，这样我们就可以正常的计算它了。
  tensorflow::Node* float_caster = tensorflow::ops::Cast(
      image_reader, tensorflow::DT_FLOAT, b.opts().WithName("float_caster"));
  // 在 TensorFlow 中图片操作的惯例就是所有的图片都是批量操作的，
  // 所以它们是由 [batch, height, width, channel] 组成的 4 维数组。
  // 因为我们只有一张图片，所以我们
  // 必须一个 1 的 batch 维度，这样才能使用 ExpandDims()。
  tensorflow::Node* dims_expander = tensorflow::ops::ExpandDims(
      float_caster, tensorflow::ops::Const(0, b.opts()), b.opts());
  // 双向调整，将我图片变成我们需要的维度。
  tensorflow::Node* resized = tensorflow::ops::ResizeBilinear(
      dims_expander, tensorflow::ops::Const({input_height, input_width},
                                            b.opts().WithName("size")),
      b.opts());
  // 减去平均值并除以缩放的比例
  tensorflow::ops::Div(
      tensorflow::ops::Sub(
          resized, tensorflow::ops::Const({input_mean}, b.opts()), b.opts()),
      tensorflow::ops::Const({input_std}, b.opts()),
      b.opts().WithName(output_name));
```
下面我们继续添加节点，然后把文件数据当做图片来解码，将整型数值转换成浮点型数值，
重新缩放，最后我们在像素值上进行提取和视觉的操作。

```C++
  // 这里执行了我们刚构造的 GraphDef 的网络定义，
  // 然后会在输出的 tensor 中返回一个结果。
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(b.ToGraphDef(&graph));
```
最后，
我们得到一个模型的定义，这个模型存储在变量 b 中，它将
会转化成一个用 `ToGraphDef()` 函数定义的完整的 graph。

```C++
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  TF_RETURN_IF_ERROR(session->Run({}, {output_name}, {}, out_tensors));
  return Status::OK();
```
然后我们创建一个 `tf.Session` 对象，这个对象是真正运行 graph 的接口，并且指明了我们想要从哪个节点得到输出以及把输出的数据推送的哪里。

它给了我们 `Tensor` 对象的一个向量，在我们知道的情况下它仅仅是一个单个的对象。你可以把 `Tensor` 想象成一个在这个上下文中的多维数组，它高 299 像素，宽 299 像素，图片的三个通道都是浮点数值。如果你在你的产品中已经有了自己的图片处理框架，那么你应该能够用它来替代，只要你在给主要的 graph 供给图片之前做同样的转换就可以了。

这是一个用 C++ 创建小型 TensorFlow 动态 graph 的示例，但是对于预训练的 Inception 模型我们想要从文件中载入更清晰的图片。你可以在 `LoadGraph()` 这个函数中看到我们是怎么做的。

```C++
// 从硬盘读取一个模型 graph 的定义，
// 创建一个你可以使用的 session 对象来运行它。
Status LoadGraph(string graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
```
如果你已经看过了加载图片的代码，那么你会发现大部分的术语都很熟悉。我们并没有使用 `GraphDefBuilder` 来生产一个 `GraphDef` 对象，而是直接加载一个包含 `GraphDef` 的 protobuf 文件。

```C++
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}
```
然后我们用 `GraphDef` 创建一个 Session 对象，并把这个对象传递给调用者，这样他们就可以随后再来运行了。

`GetTopLabels()` 这个函数和图片载入的函数很像，在这种情况下，我们想要获得运行主 graph 的结果，并且把它转化成一个有最高分标签的有序列表。就像图片加载器一样，它创建了一个 `GraphDefBuilder`，添加了一些节点，并且运行了一个短的 graph 来获得一对 tensors 的输出。在这种情况下他们代表着有序的得分以及最高分结果的下标位置。

```C++
// 分析 Inception graph 的输出信息，并且在它们相关的分类上获取
// 最高的得分以及在 tensor 中的位置。
Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                    Tensor* indices, Tensor* scores) {
  tensorflow::GraphDefBuilder b;
  string output_name = "top_k";
  tensorflow::ops::TopK(tensorflow::ops::Const(outputs[0], b.opts()),
                        how_many_labels, b.opts().WithName(output_name));
  // 这里执行了我们刚构造的 GraphDef 的网络定义，
  // 然后会在输出的 tensor 中返回一个结果。
  
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(b.ToGraphDef(&graph));
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // TopK 这个节点返回了两个输出，得分和他们的原始下标，
  // 所以我们需要添加 :0 和 :1 来区分它们。
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"},
                                  {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
```

`PrintTopLabels()` 这个函数获取了那些有序的结果，然后把它们友好的打印了出来。`CheckTopLabel()` 这个函数也是老熟人了，但是为了调试，我们还是要确定下最顶部的标签就是我们最想要的那个。

最后，[`main()`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc#L252)函数会把所有这些调用都整理在一起。

```C++
int main(int argc, char* argv[]) {
  // 我们需要调用这个函数来设置 TensorFlow 的全局状态。
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  Status s = tensorflow::ParseCommandLineFlags(&argc, argv);
  if (!s.ok()) {
    LOG(ERROR) << "Error parsing command line flags: " << s.ToString();
    return -1;
  }

  // 首先我们加载并初始化模型。
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(FLAGS_root_dir, FLAGS_graph);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }
```
我们加载主 graph。

```C++
  // 从硬盘中获取图片并转换成浮点数组，
  // 调整大小并标准化成主 graph 要求的格式。
  std::vector<Tensor> resized_tensors;
  string image_path = tensorflow::io::JoinPath(FLAGS_root_dir, FLAGS_image);
  Status read_tensor_status = ReadTensorFromImageFile(
      image_path, FLAGS_input_height, FLAGS_input_width, FLAGS_input_mean,
      FLAGS_input_std, &resized_tensors);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = resized_tensors[0];
```

载入，缩放以及处理输入的图片。

```C++
  // 真正的通过模型执行图片
  std::vector<Tensor> outputs;
  Status run_status = session->Run({{FLAGS_input_layer, resized_tensor}},
                                   {FLAGS_output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
```

这里我们运行使用图片作为输入的已经载入的 graph。

```C++
  // 这里是为了保证我们在默认设置下得到了我们想要结果的自动测试程序。
  // 我们知道标签 866（军装）应该
  // 是 Admiral Hopper 图片的最高层标签。
  if (FLAGS_self_test) {
    bool expected_matches;
    Status check_status = CheckTopLabel(outputs, 866, &expected_matches);
    if (!check_status.ok()) {
      LOG(ERROR) << "Running check failed: " << check_status;
      return -1;
    }
    if (!expected_matches) {
      LOG(ERROR) << "Self-test failed!";
      return -1;
    }
  }
```

为了测试，我们检查一下以确保我们得到了我们想要的结果。

```C++
  // 用我们生成的结果做一些有趣的事情。
  Status print_status = PrintTopLabels(outputs, FLAGS_labels);
```

最终我们把我们找到的标签打印出来。

```C++
  if (!print_status.ok()) {
    LOG(ERROR) << "Running print failed: " << print_status;
    return -1;
  }
```

这里的异常处理是使用 TensorFlow 的 `Status` 对象，`Status` 对象使用起来非常方便，因为它的 `ok()` 检查器可以让你知道是否有任何异常发生，并且还可以以可读的错误信息的形式把它们打印出来。

这里我们只展示了目标识别，但是你应该能够在各种各样的领域中以及任何你发现或者你自己训练的模型中使用这些相似的代码。

> **练习**：迁移学习是这样的一种概念，就是如果你知道如何解决好这个问题，那么你应该能够把一些解决相关问题的理念和方法迁移过去。一种表现迁移学习的方法是移除网络最后一个分类层，并且提取出 [next-to-last layer of the CNN](https://arxiv.org/abs/1310.1531)，在这种情况下，就是一个 2048 维的向量。

## 进一步学习的资源

学习更多的通用神经网络，Michael Nielsen 的[免费在线书籍](http://neuralnetworksanddeeplearning.com/chap1.html)是一个很不错的资源。在卷积神经网络方面，Chris Olah 有一些[很棒的博客](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/)，Michael Nielsen 的书中也有 [一章](http://neuralnetworksanddeeplearning.com/chap6.html) 也包含了这部分内容。

更多的关于实现卷积神经网络的资源，你可以去 TensorFlow [深度卷积网络教程](../../tutorials/images/deep_cnn.md)查看，或者跟随我们的 [Estimator MNIST 教程](../estimators/cnn.md)入门指南来来学习。最后，如果你想快速提升在这个领域的研究，可以阅读本篇指南引用的所有论文以及他们近期的工作。
