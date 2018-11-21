# 源码编译

从源代码构建 TensorFlow <b>pip</b> 安装包并将其安装在 Ubuntu Linux 和 macOS 系统上。虽然这些说明可能适用于其他系统，但它仅针对 Ubuntu 和 macOS 进行测试和支持。

注意：我们已经为 Linux 和 macOS 系统提供了经过充分测试的，预先构建的 [TensorFlow 软件包](./pip.md)。


## Linux 和 macOS 系统设置

安装以下编译工具来配置开发环境。

### 安装 Python 和 TensorFlow 的相关依赖包

<div class="ds-selector-tabs">
<section>
<h3>Ubuntu</h3>
<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">sudo apt install python-dev python-pip  # or python3-dev python3-pip</code>
</pre>
</section>
<section>
<h3>mac OS</h3>
<p>要求 Xcode 8.3 或者更新版本。</p>
<p>使用 <a href="https://brew.sh/" class="external">Homebrew</a> 包管理工具安装：</p>
<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"</code>
<code class="devsite-terminal">export PATH="/usr/local/bin:/usr/local/sbin:$PATH"</code>
<code class="devsite-terminal">brew install python@2  # or python (Python 3)</code>
</pre>
</section>
</div><!--/ds-selector-tabs-->

安装 TensorFlow <b>pip</b> 安装包的依赖模块（如果使用虚拟环境，忽略 `--user` 参数）：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">pip install -U --user pip six numpy wheel mock</code>
<code class="devsite-terminal">pip install -U --user keras_applications==1.0.5 --no-deps</code>
<code class="devsite-terminal">pip install -U --user keras_preprocessing==1.0.3 --no-deps</code>
</pre>

在 `REQUIRED_PACKAGES` 下的 <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py" class="external"><code>setup.py</code></a> 文件中罗列了所有的依赖包。

### 安装 Bazel

[安装 Bazel](https://docs.bazel.build/versions/master/install.html){:.external}，用于编译 TensorFlow 的构建工具。

将 Bazel 可执行文件的位置添加到 `PATH` 环境变量中。

### 安装 GPU 支持（可选项，仅支持 Linux）

GPU 现在还<b>不</b>支持 macOS。

阅读 [GPU 支持](./gpu.md)指南，以安装在 GPU 上运行 TensorFlow 所需的驱动程序和其它软件。

注意：设置一个支持 GPU 的 TensorFlow [Docker 镜像](#docker_linux_builds)更容易。

### 下载 TensorFlow 源码

使用 [Git](https://git-scm.com/){:.external} 克隆 [TensorFlow 仓库](https://github.com/tensorflow/tensorflow){:.external}：

<pre class="devsite-click-to-copy">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
<code class="devsite-terminal">cd tensorflow</code>
</pre>

仓库默认是 `master` 开发分支。你可以检出 [release 分支](https://github.com/tensorflow/tensorflow/releases){:.external} 来编译：

<pre class="devsite-terminal prettyprint lang-bsh">
git checkout <em>branch_name</em>  # r1.9, r1.10, etc.
</pre>

要测试源代码副本，请运行以下测试（这可能需要一段时间）：

<pre class="devsite-terminal prettyprint lang-bsh">
bazel test -c opt -- //tensorflow/... -//tensorflow/compiler/... -//tensorflow/contrib/lite/...
</pre>

关键点：如果在最新的开发分支上遇到构建问题，请尝试已知可用的发布分支。


## 编译配置

在 TensorFlow 源码根目录下运行以下命令来配置系统编译参数：

<pre class="devsite-terminal devsite-click-to-copy">
./configure
</pre>

此脚本会提示输入 TensorFlow 依赖项的位置，并询问其它编译配置选项（例如，编译器标志）。以下显示了 `./configure` 的示例运行（你的会话可能不同）：

<section class="expandable">
<h4 class="showalways">查看示例配置会话</h4>
<pre class="devsite-terminal">
./configure
You have bazel 0.15.0 installed。
请指定 python 的位置。[默认是 /usr/bin/python]：<b>/usr/bin/python2.7</b>

找到可用的 Python 库路径：
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
请输入指定想要使用的 Python 库路径。默认是 [/usr/lib/python2.7/dist-packages]

您是否希望使用 jemalloc 构建 TensorFlow 作为 malloc 支持？[Y/n]：
将为 TensorFlow 启用 jemalloc 作为 malloc 支持。

您是否希望通过 Google Cloud 平台支持构建 TensorFlow？[Y/n]：
将为 TensorFlow 启用 Google Cloud 平台支持。

您是否希望使用 Hadoop 文件系统支持构建 TensorFlow？[Y/n]：
将为 TensorFlow 启用 Hadoop 文件系统支持。

您是否希望通过 Amazon AWS 平台支持构建 TensorFlow？[Y/n]：
将为 TensorFlow 启用 Amazon AWS 平台支持。

您是否希望使用 Apache Kafka 平台支持构建 TensorFlow？[Y/n]：
将为 TensorFlow 启用 Apache Kafka 平台支持。

您是否希望使用 XLA JIT 支持构建 TensorFlow？[y/N]：
不会为 TensorFlow 启用 XLA JIT 支持。

您是否希望使用 GDR 支持构建 TensorFlow？[y/N]：
不会为 TensorFlow 启用 GDR 支持。

您是否希望使用 VERBS 支持构建 TensorFlow？[y/N]：
不会为 TensorFlow 启用 VERBS 支持。

您是否希望使用 OpenCL SYCL 支持构建 TensorFlow？[y/N]：
不会为 TensorFlow 启用 OpenCL SYCL 支持。

您是否希望使用 CUDA 支持构建 TensorFlow？[y/N]：<b>Y</b>
将为 TensorFlow 启用 CUDA 平台支持。

请指定您想使用的 CUDA SDK 版本。[留空则默认使用 CUDA 9.0]：<b>9.0</b>

请指定安装 CUDA 9.0 工具包的位置。有关更多详细信息，请参阅 README.md。[默认 /usr/local/cuda]：

请指定您想使用的 cuDNN 版本。[留空默认使用 cuDNN 7.0]：<b>7.0</b>

请指定安装 cuDNN 7 工具包的位置。有关更多详细信息，请参阅 README.md。[默认 /usr/local/cuda]：

您是否希望使用 TensorRT 支持构建 TensorFlow？[y/N]：
不会为 TensorFlow 启用 TensorRT 支持。

请指定您要使用的 NCCL 版本。如果未安装 NCLL 2.2，则可以使用可以自动获取的版本 1.3，但是使用多个 GPU 可能性能更差。[默认 2.2]：1.3

请指定要构建的以逗号分隔的 CUDA 计算功能的列表。
您可以在以下位置找到设备的计算能力：https://developer.nvidia.com/cuda-gpus。
请注意每一个额外的计算功能都会明显的增加编译时间以及生成的二进制文件大小。[默认是：3.5,7.0] <b>6.1</b>

您是否想要使用 clang 作为 CUDA 编译器？[y/N]：
nvcc 将作为 CUDA 编译器。

请指定 nvcc 应使用哪个 gcc 作为主机编译器。[默认 /usr/bin/gcc]：

您是否希望使用 MPI 支持构建 TensorFlow？[y/N]：
不会为 TensorFlow 启用 MPI 支持。

当指定 bazel 选项 "--config=opt" 时，请指定在编译期间使用的优化标志[默认 -march=native]：

您是否要以交互方式为 Android 版本配置 ./WORKSPACE？[y/N]：
不会为 Android 编译 WORKSPACE。

预先配置 Bazel 构建配置信息。您可以通过在构建命令中添加 "--config=<>" 来使用以下任何一项。有关更多详细信息，请参阅 tools/bazel.rc。
    --config=mkl            # 构建 MKL 支持。
    --config=monolithic     # 配置主要是静态整体构造。
配置结束
</pre>
</section>

### 配置选项

对于 [GPU 支持](./gpu.md)，请指定 CUDA 和 cuDNN 的版本。如果系统安装了多个版本的 CUDA 或 cuDNN，请显式设置版本而不是依赖于默认版本。`./configure` 创建指向系统 CUDA 库的符号链接 — 因此，如果更新 CUDA 库路径，则必须在构建之前再次运行此配置步骤。

对于编译优化标志，默认（`-march=native`）优化机器 CPU 类型的生成代码。但是，如果为不同的 CPU 类型构建 TensorFlow，请考虑更具体的优化标志。有关示例，请参阅 [GCC 手册](https://gcc.gnu.org/onlinedocs/gcc-4.5.3/gcc/i386-and-x86_002d64-Options.html){:.external}。

有一些预先配置的构建配置可以添加到 `bazel build` 命令中，例如：

* `--config=mk1` — 支持 [Intel® MKL-DNN](https://github.com/intel/mkl-dnn){:.external}。
* `--config=monolithic` — 配置为静态统一编译。

注意：从 TensorFlow 1.6 开始，二进制文件使用 AVX 指令，这些指令可能无法在较旧的 CPU 上运行。

## 编译 pip 包

### Bazel 编译

#### 仅支持 CPU

使用 `bazel` 使 TensorFlow 包仅支持 CPU：

<pre class="devsite-terminal devsite-click-to-copy">
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
</pre>

#### 支持 GPU

使 TensorFlow 包支持 GPU：

<pre class="devsite-terminal devsite-click-to-copy">
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
</pre>

#### Bazel 编译选项

从源代码构建 TensorFlow 会使用大量的 RAM。如果系统受内存限制，请将 Bazel 的 RAM 使用限制为：`--local_resources 2048,.5,1.0`。

[官方 TensorFlow 软件包](./pip.md)使用 GCC 4 构建并使用较旧的 ABI。对于 GCC 5 及更高版本，使用以下命令使您的构建与旧 ABI 兼容：`--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"`。ABI 兼容性确保针对官方 TensorFlow 包构建的自定义操作继续与 GCC 5 构建的包一起使用。

### 编译包

`bazel build` 命令创建一个名为 `build_pip_package` 的可执行文件 — 这是构建 `pip` 包的程序。例如，下面在 `/tmp/tensorflow_pkg` 目录中构建一个 `.whl` 包：

<pre class="devsite-terminal devsite-click-to-copy">
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
</pre>

虽然可以在同一源代码树下构建 CUDA 和非 CUDA 配置，但建议在同一源代码树中切换这两种配置时运行 `bazel clean`。

### 安装包

生成的 `.whl` 文件的文件名取决于 TensorFlow 版本和您的平台。使用 `pip install` 来安装软件包，例如：

<pre class="devsite-terminal prettyprint lang-bsh">
pip install /tmp/tensorflow_pkg/tensorflow-<var>version</var>-cp27-cp27mu-linux_x86_64.whl
</pre>

成功：TensorFlow 已安装。

## Docker Linux 编译

TensorFlow 的 Docker 开发镜像是一种设置环境以从源代码构建 Linux 包的简单方法。这些镜像已包含构建 TensorFlow 所需的源代码和依赖项。请参阅 TensorFlow [Docker 指南](./docker.md)进行安装，并查看[可用镜像标签列表](https://hub.docker.com/r/tensorflow/tensorflow/tags/){:.external}。

### 仅支持 CPU

以下示例使用 `:nightly-devel` 镜像从最新的 TensorFlow 源代码编译仅支持 CPU 的 Python 2 的安装包。有关可用的 TensorFlow `-devel` 标签，请参阅 [Docker 指南](./docker.md)。

下载最新的开发镜像并启动我们将用于构建 <b>pip</b> 包的 Docker 容器：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">docker pull tensorflow/tensorflow<var>:nightly-devel</var></code>
<code class="devsite-terminal">docker run -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    tensorflow/tensorflow<var>:nightly-devel</var> bash</code>

<code class="devsite-terminal tfo-terminal-root">git pull  # within the container, download the latest source code</code>
</pre>

上面的 `docker run` 命令将在源代码的根目录 `/tensorflow` 中启动一个 shell — 源代码的根目录。它将主机的当前目录安装在容器的 `/mnt` 中，并通过环境变量将主机用户的信息传递给容器（用于设置权限 — Docker 可以使这一点变得简单）。

或者，要在容器中构建主机上的 TensorFlow 副本，请将主机的源码目录树挂载到容器的 `/tensorflow` 目录：

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-bsh">
docker run -it -w /tensorflow -v <var>/path/to/tensorflow</var>:/tensorflow -v $PWD:/mnt \
    -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/tensorflow:<var>nightly-devel</var> bash
</pre>

设置源树后，在容器的虚拟环境中构建 TensorFlow 包：

1. 配置构建—这会提示用户回答构建配置问题。
2. 构建用于创建 <b>pip</b> 包的工具。
3. 运行该工具以创建 <b>pip</b> 包。
4. 调整容器外部文件的所有权。

<pre class="devsite-disable-click-to-copy prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-root">./configure  # answer prompts or use defaults</code>

<code class="devsite-terminal tfo-terminal-root">bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package</code>

<code class="devsite-terminal tfo-terminal-root">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /mnt  # create package</code>

<code class="devsite-terminal tfo-terminal-root">chown $HOST_PERMS /mnt/tensorflow-<var>version</var>-cp27-cp27mu-linux_x86_64.whl</code>
</pre>

在容器中安装并验证包：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-root">pip uninstall tensorflow  # 卸载当前版本</code>

<code class="devsite-terminal tfo-terminal-root">pip install /mnt/tensorflow-<var>version</var>-cp27-cp27mu-linux_x86_64.whl</code>
<code class="devsite-terminal tfo-terminal-root">cd /tmp  # 不要从源目录导入</code>
<code class="devsite-terminal tfo-terminal-root">python -c "import tensorflow as tf; print(tf.__version__)"</code>
</pre>

成功：TensorFlow 已安装。

在你的主机上，TensorFlow <b>pip</b> 包在当前的目录下（拥有主机使用者权限）：<code>./tensorflow-<var>version</var>-cp27-cp27mu-linux_x86_64.whl</code>

### 支持 GPU 

Docker 是为 TensorFlow 构建 GPU 支持的最简单方法，因为<b>主机</b>机器只需要 [NVIDIA®驱动程序](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-in-install-the-nvidia-driver){:.external} <b>不需要安装 NVIDIA®CUDA®Toolkit</b>。请参阅 [GPU 支持指南](./gpu.md)和 TensorFlow [Docker 指南](./docker.md)以设置 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker){:.external}（仅限 Linux）。

下面的示例下载 TensorFlow `:nightly-devel-gpu-py3` 镜像，并使用 `nvidia-docker` 来运行支持 GPU 的容器。此开发镜像配置为构建具有 GPU 支持的 Python 3 <b>pip</b> 包：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">docker pull tensorflow/tensorflow<var>:nightly-devel-gpu-py3</var></code>
<code class="devsite-terminal">docker run --runtime=nvidia -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    tensorflow/tensorflow<var>:nightly-devel-gpu-py3</var> bash</code>
</pre>

然后，在容器的虚拟环境中，构建具有 GPU 支持的 TensorFlow 包：

<pre class="devsite-disable-click-to-copy prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-root">./configure  # answer prompts or use defaults</code>

<code class="devsite-terminal tfo-terminal-root">bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package</code>

<code class="devsite-terminal tfo-terminal-root">./bazel-bin/tensorflow/tools/pip_package/build_pip_package /mnt  # create package</code>

<code class="devsite-terminal tfo-terminal-root">chown $HOST_PERMS /mnt/tensorflow-<var>version</var>-cp35-cp35m-linux_x86_64.whl</code>
</pre>

安装并验证容器中的包并检查 GPU：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal tfo-terminal-root">pip uninstall tensorflow  # remove current version</code>

<code class="devsite-terminal tfo-terminal-root">pip install /mnt/tensorflow-<var>version</var>-cp27-cp27mu-linux_x86_64.whl</code>
<code class="devsite-terminal tfo-terminal-root">cd /tmp  # don't import from source directory</code>
<code class="devsite-terminal tfo-terminal-root">python -c "import tensorflow as tf; print(tf.contrib.eager.num_gpus())"</code>
</pre>

成功：TensorFlow 已安装。


## 测试编译配置

### Linux

<table>
<tr><th>Version</th><th>CPU/GPU</th><th>Python version</th><th>Compiler</th><th>Build tools</th><th>cuDNN</th><th>CUDA</th></tr>
<tr><td>tensorflow-1.10.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.15.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.10.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.15.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.9.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.11.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.9.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.11.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.8.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.10.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.8.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.9.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.7.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.10.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.7.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.9.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.6.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.9.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.6.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.9.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.5.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.8.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.5.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.8.0</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.4.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.5.4</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.4.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.5.4</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow-1.3.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.3.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow-1.2.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.2.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.5</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.1.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.1.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.0.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.0.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>GCC 4.8</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
</table>

### macOS

<table>
<tr><th>Version</th><th>CPU/GPU</th><th>Python version</th><th>Compiler</th><th>Build tools</th><th>cuDNN</th><th>CUDA</th></tr>
<tr><td>tensorflow-1.10.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.15.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.9.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.11.0</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.8.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.10.1</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.7.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.10.1</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.6.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.8.1</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.5.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.8.1</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.4.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.5.4</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.3.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.2.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.5</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow-1.1.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.1.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.0.0</td><td>CPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.0.0</td><td>GPU</td><td>2.7, 3.3-3.6</td><td>Clang from xcode</td><td>Bazel 0.4.2</td><td>5.1</td><td>8</td></tr>
</table>
