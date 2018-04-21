# 移动端优化

在移动或嵌入式设备中存在一些你需要去处理的问题，同时你还需要在开发模型时就思考这些问题。

问题包括：

- 模型及其二进制文件的大小
- 应用的运行速度以及模型的加载速度
- 性能与线程管理

我们会在下面简单讨论这些问题。

## TensorFlow 的最低要求

运行包含基本功能的 TensorFlow 需要牺牲 1MB 的程序以及几兆的内存大小，因此 TensorFlow 不能运行在 DSP 或微控制器上。除此之外，限制 TensorFlow 的最大因素通常是设备的计算速度，以及你能否在足够低的计算延迟内运行你所需的模型。你可以参考[模型的分析](https://github.com/xitu/tensorflow-docs/pull/37#%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%88%86%E6%9E%90)用性能测基准测试工具来了解运行模型需要多少 FLOP，然后根据该值对不同设备的运行速度做经验估算。例如，当今的智能手机可以每秒运行 10 GFLOP，所以对于 5 GFLOP 模型而言，可以得到的最佳效果是每秒两帧，根据设备实际的计算模式，你可能需要降低帧率。

这种模型的要求使得只要你能在有限的内存中将神经网络优化得足够好以适应计算延迟，TensorFlow 就可以运行在非常古老或受到限制的智能手机上。考虑到内存的限制，你需要保证 TensorFlow 创建的中间缓存不能太大，你也可以通过性能基准输出做检查。

## 速度

在大部分模型部署任务中，最高优先级之一的任务是如何快速运行推断程序来提供良好的用户体验。首先我们需要查看执行运算图所需的 FLOP 数。你可以使用 `benchmark_model` 工具来进行粗略估计：

    bazel build -c opt tensorflow/tools/benchmark:benchmark_model && \
    bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph=/tmp/inception_graph.pb --input_layer="Mul:0" \
    --input_layer_shape="1,299,299,3" --input_layer_type="float" \
    --output_layer="softmax:0" --show_run_order=false --show_time=false \
    --show_memory=false --show_summary=true --show_flops=true --logtostderr

上面的命令会输出运行此计算图所需计算所需操作数的估算值。然后你就可以使用这些信息来确定你的模型在你的目标设备上运行的可行性。举个例子，2016 年的高端智能手机能够在每秒处理 200 亿个 FLOP，因此执行需要 100 亿 FLOP 的模型的期望速度约为 500 ms。在类似于 Raspberry Pi 3 这种每秒只能处理 50 亿个 FLOP 的设备上，你可能要两秒才能获得一个推断的计算结果。

当有了计算操作消耗的估计之后，它就对你计划的目标设备上有所帮助。如果模型的计算操作太多，那么你可以有很多方式来优化模型的架构并减少这个数量。

一些比较新的技术有 [SqueezeNet](https://arxiv.org/abs/1602.07360) 和 [MobileNet](https://arxiv.org/abs/1704.04861) 等，这些架构专门为移动设备的定制模型 —— 体积小、运行速度快但代价是精度低。你也可以使用这些模型的一些更小更老的替代模型。比如，与 Inception v3 的 2400 万参数量且消耗 90 亿 FLOP 相比，Inception v1 只有 700 万个参数且仅需 30 亿 FLOP。

## 模型大小

运行在移动设备上的模型需要存储在模型的某个地方，巨大的神经网络可能消耗上百兆的存储空间。大部分用户不愿意从应用商店中下载相当大的程序包，因此你必须尽可能的压缩模型的体积。况且，神经网络越小移动设备加载越快。

为了了解你的模型在设备上消耗的磁盘大小，首先你需要在模型上运行 `freeze_graph` 和 `strip_unused_nodes` 后查看 `GraphDef` 文件的磁盘大小（参考 @{$mobile/prepare_models$Preparing models} 一节中关于这个工具的更多细节），因为这样你才能使计算图程序仅包含推断相关的计算节点。为了验证你的结果是否符合预期，你可以运行 `summarize_graph` 工具中查看内建常量中包含的参数个数：

    bazel build tensorflow/tools/graph_transforms:summarize_graph && \
    bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
    --in_graph=/tmp/tensorflow_inception_graph.pb

上面的命令会输出类似下面的结果：

    No inputs spotted.
    Found 1 possible outputs: (name=softmax, op=Softmax)
    Found 23885411 (23.89M) const parameters, 0 (0) variable parameters,
    and 99 control_edges
    Op types used: 489 Const, 99 CheckNumerics, 99 Identity, 94
    BatchNormWithGlobalNormalization, 94 Conv2D, 94 Relu, 11 Concat, 9 AvgPool,
    5 MaxPool, 1 Sub, 1 Softmax, 1 ResizeBilinear, 1 Reshape, 1 Mul, 1 MatMul,
    1 ExpandDims, 1 DecodeJpeg, 1 Cast, 1 BiasAdd

我们当前目标最重要的部分是确定一共有多少个参数。在大部分模型中，这些数据会被存储为 32 位浮点数，如果将常量参数的数量乘以四，就差不多是磁盘上文件大小的近似值。通常每个参数只有 8 bit，得到的最终结果误差会很小，所以如果你的文件太大，可以尝试用 @{$performance/quantization$quantize_weights} 一节中的方法将这些参数做向下的权重量化处理。

    bazel build tensorflow/tools/graph_transforms:transform_graph && \
    bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=/tmp/tensorflow_inception_optimized.pb \
    --out_graph=/tmp/tensorflow_inception_quantized.pb \
    --inputs='Mul:0' --outputs='softmax:0' --transforms='quantize_weights'

如果你检查一下文件的大小，那么这个文件大约是 23MB，即源文件大小的四分之一。

另一种转换是方式是 `round_weights`，它并不会让整个文件变小，但是当对其进行压缩时可以让整个文件大小跟使用 `quantized_weights` 时相当。由于用户在下载应用时本质上是下载应用的一个压缩版本，所以这对于移动开发来说相当有用。

模型的原始文件不能被标准压缩算法压缩得当，这是由于相同数字的位模式可能会完全不同所导致的。`round_weights` 变换会将权重参数四舍五入的结果保存为浮点数。这就意味着如果存储模型中存在相当多的重复字节，压缩效率就会大大增加。大多数情况下会接近八位二进制数的存储大小。

`round_weights` 相对于 `quantize_weights` 的另一个优点是可以使框架在解压参数时不需要分配一个临时缓冲区。这种操作可以降低一些延迟（由于结果仍然需要缓存，所以这个操作只发生在第一次运行），并且使内存映射成为可能，我们会在下面的内容中讨论。

## 二进制文件大小

移动开发和服务器开发之间的最大区别之一就是二进制文件大小的重要性。在桌面计算机中，磁盘上安装数百兆字节的可执行文件已经不再罕见，但对于移动或嵌入式应用来说，保持二进制文件尽可能小，是使用户更容易下载应用的关键因素之一。如上所述，默认情况下， TensorFlow 仅包含了运算符实现的一个子集，但这仍然会产生 12 MB 大小的最终文件。你可以基于自动分析模型将库设置为实际需要的计算部分以减少包的体积，为此：

- 运行 `tools/print_required_ops/print_selective_registration_header.py` 来生成只包含所需运算符的头文件。
- 将 `ops_to_register.h` 头文件放在编译器能够找到的地方，也可以放在 TensorFlow 的根目录下。
- 通过定义 `SELECTIVE_REGISTRATION` 构建 TensorFlow。例如，将 `--copts="-DSELECTIVE_REGISTRATION"` 传递给你的 Bazel 命令。

在此过程中重新编译了整个库并且仅包含所需的操作和类型，从而显著减少可执行文件的大小。例如，在 Inception v3 中，新模型的大小仅为 1.5 MB。

## 模型的分析

一旦你对设备的性能峰值范围有所了解之后，检查一下它目前的实际性能是很有必要的。不要在应用中直接运行，而是单独使用 TensorFlow 的性能基准工具进行测试，有利于计算出单独 TensorFlow 本身对延迟的影响。[tensorflow/tools/benchmark](https://www.tensorflow.org/code/tensorflow/tools/benchmark/) 中的工具能够帮助你做到这一点。要将其用在桌面系统的 Inception v3 上，可以使用如下命令：

    bazel build -c opt tensorflow/tools/benchmark:benchmark_model && \
    bazel-bin/tensorflow/tools/benchmark/benchmark_model \
    --graph=/tmp/tensorflow_inception_graph.pb --input_layer="Mul" \
    --input_layer_shape="1,299,299,3" --input_layer_type="float" \
    --output_layer="softmax:0" --show_run_order=false --show_time=false \
    --show_memory=false --show_summary=true --show_flops=true --logtostderr

你的输出结果与下方的结果类似：

<pre>
============================== Top by Computation Time ==============================
[node
 type]  [start]  [first] [avg ms]     [%]  [cdf%]  [mem KB]  [Name]
Conv2D   22.859   14.212   13.700  4.972%  4.972%  3871.488  conv_4/Conv2D
Conv2D    8.116    8.964   11.315  4.106%  9.078%  5531.904  conv_2/Conv2D
Conv2D   62.066   16.504    7.274  2.640% 11.717%   443.904  mixed_3/conv/Conv2D
Conv2D    2.530    6.226    4.939  1.792% 13.510%  2765.952  conv_1/Conv2D
Conv2D   55.585    4.605    4.665  1.693% 15.203%   313.600  mixed_2/tower/conv_1/Conv2D
Conv2D  127.114    5.469    4.630  1.680% 16.883%    81.920  mixed_10/conv/Conv2D
Conv2D   47.391    6.994    4.588  1.665% 18.548%   313.600  mixed_1/tower/conv_1/Conv2D
Conv2D   39.463    7.878    4.336  1.574% 20.122%   313.600  mixed/tower/conv_1/Conv2D
Conv2D  127.113    4.192    3.894  1.413% 21.535%   114.688  mixed_10/tower_1/conv/Conv2D
Conv2D   70.188    5.205    3.626  1.316% 22.850%   221.952  mixed_4/conv/Conv2D

============================== Summary by node type ==============================
[Node type]  [count]  [avg ms]    [avg %]    [cdf %]  [mem KB]
Conv2D            94   244.899    88.952%    88.952% 35869.953
BiasAdd           95     9.664     3.510%    92.462% 35873.984
AvgPool            9     7.990     2.902%    95.364%  7493.504
Relu              94     5.727     2.080%    97.444% 35869.953
MaxPool            5     3.485     1.266%    98.710%  3358.848
Const            192     1.727     0.627%    99.337%     0.000
Concat            11     1.081     0.393%    99.730%  9892.096
MatMul             1     0.665     0.242%    99.971%     4.032
Softmax            1     0.040     0.015%    99.986%     4.032
<>                 1     0.032     0.012%    99.997%     0.000
Reshape            1     0.007     0.003%   100.000%     0.000

Timings (microseconds): count=50 first=330849 curr=274803 min=232354 max=415352 avg=275563 std=44193
Memory (bytes): count=50 curr=128366400(all same)
514 nodes defined 504 nodes observed
</pre>

这是用于设置了 `show_summary` 标志而产生的摘要视图。在这个输出中，第一个表格表示了花费时间最多的节点的列表。由左至右，每列分别表示：

- node type：进行操作的节点类型。
- start：运算符的启动时间，展示了其在操作顺序中的位置。
- first: 以毫秒为单位。默认情况下 TensorFlow 会执行 20 次运行结果来获得统计数据，这个字段则表示第一次运行基准测试所需的操作时间。
- avg ms：以毫秒为单位。表示整个运行的平均操作时间。
- %：一次运行占总运行时间的百分比。这对理解密集计算区域非常有用。
- cdf%：整个过程中表格中当前运算符及上方全部运算符的累积计算时间。这对理解神经网络不同层之间的性能分布非常重要，有助于查看是否只有少数节点占用大部分时间。
- mem KB：当前层消耗的内存大小。
- Name：节点名称。

第二个表格类似，但并非以特定命名节点划分时间，而是按照运算符类型对他们进行分组。如果你想要从计算图中优化某个单一的操作符，那么这将非常有用。这个表在开始时以计算成本最昂贵的操作符开始，并且仅显示前十个操作符，并具有其他节点的占位符。从左到右依次为：

- Node type：分析节点的类型
- avg ms：所有此类型节点所耗时间的平均值，以毫秒为单位。
- avg %：此类型操作占总操作的比例。
- cdf%：整个过程中表格中当前运算符及上方全部运算符的累积计算时间。有助于理解神经网络不同运算符之间的性能分布。
- mem KB：当前运算符消耗的内存大小

这两个表格均已设置完毕，因此你可以轻松的将结果复制并粘贴到表格文档中，因为他们与制表符一起作为列与列之间的分隔符而输出。因为这份节点类型统计指出了那些代码最消耗时间，对寻找优化点非常有帮助。在这个例子中，你可以看到 Conv2D 算符消耗了 90% 的执行时间。这表明你的计算图已经优化得相当好了，因为卷积与矩阵的乘积是整个神经网络中最具分量的操作。

就经验而言，如果你看到其他运算符占用的时间更多，那么你就要仔细考虑这个问题了。对于神经网络来说，不需要涉及矩阵乘法的运算符是不值得一提的，因此如果你看到这些运算符消耗了较多的时间，那么说明你的网络结构很可能并非最优结构，或者说是我们实现的运算符代码并没有做到最优。如果你遇到了这种情况，欢迎你给我们提[性能 Bug](https://github.com/tensorflow/tensorflow/issues)，同时也提交能复现该问题的模型以及性能测试工具的命令。

上面的提到的内容是运行在你的桌面系统上的，但是这个工具也适用于 Android，因此也对移动开发相当有用。以下是一个示例代码，可以在一个 64 位 ARM 设备上运行：

    bazel build -c opt --config=android_arm64 \
    tensorflow/tools/benchmark:benchmark_model
    adb push bazel-bin/tensorflow/tools/benchmark/benchmark_model /data/local/tmp
    adb push /tmp/tensorflow_inception_graph.pb /data/local/tmp/
    adb shell '/data/local/tmp/benchmark_model \
    --graph=/data/local/tmp/tensorflow_inception_graph.pb --input_layer="Mul" \
    --input_layer_shape="1,299,299,3" --input_layer_type="float" \
    --output_layer="softmax:0" --show_run_order=false --show_time=false \
    --show_memory=false --show_summary=true'

你可以用与上方完全相同的方式来获得这些结果。如果你在模型确定输入和输出的名称时遇到困难，请到 @{$mobile/prepare_models$Preparing models} 一节中查看详细说明，并尝试 `summarize_graph` 工具，它会给你很多帮助。 

在 iOS 上并没有较好的命令行工具的支持，相反 [tensorflow/examples/ios/benchmark](https://www.tensorflow.org/code/tensorflow/examples/ios/benchmark) 中有一个单独的例子在独立的应用中封装了这些功能。它会将统计信息输出到屏幕设备和调试日志中。对于 Android 的示例应用来说，你可以通过音量按钮来开启屏幕上的统计信息。

## 在移动应用中进行分析

性能基准工具生成的结果其实是 TensorFlow 运行时一部分模块生成的，这也就意味着你可以在自己的应用中访问这些结果。你可以在[这里](https://www.tensorflow.org/code/tensorflow/examples/ios/benchmark/BenchmarkViewController.mm?l=139)找到一个例子。

基本步骤为：

1. 创建一个 StatSummarizer 对象：

        tensorflow::StatSummarizer stat_summarizer(tensorflow_graph);

2. 设置运行选项：

        tensorflow::RunOptions run_options;
        run_options.set_trace_level(tensorflow::RunOptions::FULL_TRACE);
        tensorflow::RunMetadata run_metadata;

3. 运行计算图：

        run_status = session->Run(run_options, inputs, output_layer_names, {},
                                  output_layers, &run_metadata);

4. 计算结果并输出：

        assert(run_metadata.has_step_stats());
        const tensorflow::StepStats& step_stats = run_metadata.step_stats();
        stat_summarizer->ProcessStepStats(step_stats);
        stat_summarizer->PrintStepStats();

## 模型可视化

加快你编码速度的最高效的手段就是修改你的模型，从而减少大量的无用工。为此，你需要了解你的模型正在执行什么样的任务，好的开始就是可视化计算过程。为浏览你计算图的整体工作，请使用 [TensorBoard](https://github.com/tensorflow/tensorboard)。

## 线程

桌面版的 TensorFlow 具有相当复杂的线程模型，在允许的情况下，它会自动尝试以多个线程进行运行。在我们的术语中，这叫做『运算符间的并行性（inter-op parallelism）』，并通过在 session 选项中指定 `inter_op_parallelism_threads` 来开启。

在默认情况下，移动设备会串行执行运算符，即 `inter_op_parallelism_threads` 设为 1。移动处理器通常具有很小的内核和一个小缓存，从而访问多个彼此不重叠的内存区域通常不会对性能提升有所帮助。像做卷积运算时不同线程分配相同大小内存，『运算符内的并行性（intra-op parallelism）』对于这种情况则非常有用。

在移动设备中，操作系统会将默认使用的线程数设为 CPU 的核心数，如果无法确定处理器的核心数，则会设置为 2。你可以在 `intra_op_parallelism_threads` 来覆盖运算符使用的默认线程数。如果你的应用本身就有很多线程在执行复杂任务，那么减少这个默认值的好处在于这能够帮助线程之间不会互相干扰。

关于更多关于 session 选项的细节，请查看 [ConfigProto](https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)。

## 使用移动端数据重新训练

在移动应用上运行模型时，导致准确率不理想最可能的原因就是训练数据无法代表真实情况。例如，大多数 ImageNet 的图片都标注得相当好，并且对象位于图片中心、光线充足并且是由相机镜头产生而成。但移动端的照片可能就比较不理想了，通常它们的构图不一、补光不够，甚至还有图像边缘产生的鱼眼失真问题，尤其是自拍。

解决方法就是：用你的移动应用来捕获实际的数据来作为你的训练数据集。这可能需要引入一些额外的工作量，因为你还需要自己为样本添加标签。即便是你将搜集到的数据作为原始数据集的扩展，它依然可以大大优化你的训练。通过这样的方式来修改训练集，并通过修复其他的样本质量问题（例如重复或错误标记的样本）是提高精确性的最佳实践。这比修改你的模型架构或使用其他不同的技术更加有用。

## 优化模型的加载时间与内存占用

大多数操作系统允许使用内存映射加载文件，而非通过 I/O API 来读取。除了在堆上分配内存区域然后从磁盘中读取二进制文件外，你只需要告诉简单的告诉操作系统直接将整个文件加载到内存中就可以了。这样做有一些好处：

* 加速加载时间
* 减少内存的分页调度（增加性能）
* 不计入你的 APP 内存占用

TensorFlow 支持由内存映射的方式来将你模型文件的大部分内容（如权重）一口气加载到内存中区。由于 `Protobuf` 序列化格式的限制，我们必须对模型的加载和处理代码进行一些修改。内存映射的工作原理是，我们有一个单独的文件，其中第一个部分是将一个普通的 `GraphDef`  序列化为 protobuf 格式，然后所有的权重就可以添加到一个直接映射的内存表后。

为了创建这个文件，请运行 `tensorflow/contrib/util:convert_graphdef_memmapped_format` 工具。这需要一个已经通过 `freeze_graph` 运行的 `Graphdef` 文件，并将其转换为最后附带了权重的格式。由于文件不再是标准的 `GraphDef` protobuf，所以你还需要对模型的加载代码进行一些修改。你可以在 [iOS Camera 示例程序](https://www.tensorflow.org/code/tensorflow/examples/ios/camera/tensorflow_utils.mm?l=147)中看到相关例子，请仔细阅读 `LoadMemoryMappedModel()` 函数的实现。

相同的代码（Objective-C 需要修改文件名）也可以用在其他平台上使用。因为我们需要使用内存映射，所以首要任务是创建一个 TensorFlow 环境对象，并使用我们希望的文件设置：

    std::unique_ptr<tensorflow::MemmappedEnv> memmapped_env;
    memmapped_env->reset(
          new tensorflow::MemmappedEnv(tensorflow::Env::Default()));
    tensorflow::Status mmap_status =
          (memmapped_env->get())->InitializeFromFile(file_path);

然后将这个环境传递给后续调用，与加载计算图类似：

    tensorflow::GraphDef tensorflow_graph;
    tensorflow::Status load_graph_status = ReadBinaryProto(
        memmapped_env->get(),
        tensorflow::MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
        &tensorflow_graph);

另外，你还需要创建一个会话的指针来指向你当前创建的环境：

    tensorflow::SessionOptions options;
    options.config.mutable_graph_options()
        ->mutable_optimizer_options()
        ->set_opt_level(::tensorflow::OptimizerOptions::L0);
    options.env = memmapped_env->get();
    
    tensorflow::Session* session_pointer = nullptr;
    tensorflow::Status session_status =
        tensorflow::NewSession(options, &session_pointer);

值得注意的是，我们还禁用了自动优化，因为在某些情况下，这会折叠常量子树，从而创建我们不希望出现的张量的拷贝导致的额外内存占用。

一旦你完成这些步骤后，就可以想往常一样使用 session 和 graph，同时你还能够看到在模型加载时的加载时间和内存使用量。

## 保护模型文件的安全

默认情况下，你的模型会在磁盘上以序列化后标准的 protobuf 格式存储。理论上来说，任何人都可以复制你的模型，但你可能并不希望别人这么做。然而，实际上大多数模型都是应用特定的，并且已经通过编译优化进行了混淆处理，其难度与反编译并复用你应用的代码一样难。但是，如果你确实希望让用户难以访问你的模型文件，可以采取下面这些基本步骤。

我们的大部分示例程序都使用 [ReadBinaryProto()](https://www.tensorflow.org/code/tensorflow/core/platform/env.cc?q=core/platform/env.cc&l=409) 从磁盘上加载一个 `GraphDef`。这的确需要使用一个未加密的 protobuf 文件。辛运的是调用的实现非常简单，你可以编写一个在内存中进行解密的中间件来达到你的目的。下面的代码展示了如何使用自己的解密函数来读取和解密 protobuf：

    Status ReadEncryptedProto(Env* env, const string& fname,
                              ::tensorflow::protobuf::MessageLite* proto) {
      string data;
      TF_RETURN_IF_ERROR(ReadFileToString(env, fname, &data));
    
      DecryptData(&data);  // 自行编写的解密方法
    
      if (!proto->ParseFromString(&data)) {
        TF_RETURN_IF_ERROR(stream->status());
        return errors::DataLoss("Can't parse ", fname, " as binary proto");
      }
      return Status::OK();
    }

要使用这个函数，你需要自己编写 `DecryptData()` 方法。他可以简单到类似于下面这种形式：

    void DecryptData(string* data) {
      for (int i = 0; i < data.size(); ++i) {
        data[i] = data[i] ^ 0x23;
      }
    }

你也可以做一些更加复杂的加密行为，但这已经超出了本文所讨论的范围。
