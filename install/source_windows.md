# 在 Windows 上编译源码

利用源码在 Windows 上构建 TensorFlow *pip* 包。

注意：我们已经为 Windows 系统提供了经过完善测试和预编译的 [TensorFlow 包](./pip.md)。

## Windows 设置

安装下列编译工具来配置你的 Windows 开发环境。

### 安装 Python 和 TensorFlow 的依赖包

安装 [Python 3.5.x 或者 Python 3.6.x 64-bit Windows 发行版](https://www.python.org/downloads/windows/){:.external}。勾选 *pip* 安装选项并把它添加到 `%PATH%` 环境变量。

安装 TensorFlow *pip* 依赖包：

<pre class="devsite-click-to-copy">
<code class="devsite-terminal tfo-terminal-windows">pip3 install six numpy wheel</code>
<code class="devsite-terminal tfo-terminal-windows">pip3 install keras_applications==1.0.5 --no-deps</code>
<code class="devsite-terminal tfo-terminal-windows">pip3 install keras_preprocessing==1.0.3 --no-deps</code>
</pre>

依赖项在 `REQUIRED_PACKAGES` 下的 <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py" class="external"><code>setup.py</code></a> 文件中列出。

### 安装 Bazel

[安装 Bazel](https://docs.bazel.build/versions/master/install-windows.html){:.external}，TensorFlow 的编译工具。

将 Bazel 可执行文件的位置添加到 `%PATH%` 环境变量。

### 安装 MSYS2

[安装 MSYS2](https://www.msys2.org/){:.external}，里面包含了编译 TensorFlow 需要用到的工具。如果 MSYS2 安装在 `C:\msys64` 路径，将 `C:\msys64\usr\bin` 添加到 `%PATH%` 环境变量。然后运行 `cmd.exe`：

<pre class="devsite-terminal tfo-terminal-windows devsite-click-to-copy">
pacman -S git patch unzip
</pre>

### 安装 Visual C++ Build Tools 2015

安装 *Visual C++ build tools 2015*。这个安装包属于 *Visual Studio 2015*，但可以被独立安装：

1. 进入 [Visual Studio 下载页面](https://visualstudio.microsoft.com/vs/older-downloads/){:.external}，
2. 选择**重发行包和编译工具**，
3. 下载 *Microsoft Build Tools 2015 Update 3*，
4. 运行安装包。

注意：TensorFlow 在 *Visual Studio 2015 Update 3* 经过测试，但它也可能在一些更新版本的 Visual C++ 编译工具上工作。

### 安装 GPU 支持（可选）

参看 Windows 下的 [GPU 支持](./gpu.md)指南，安装驱动和需要的附加软件来在 GPU 上运行 TensorFlow。

### 下载 TensorFlow 源码

用 [Git](https://git-scm.com/){:.external} 克隆 [TensorFlow 仓库](https://github.com/tensorflow/tensorflow){:.external}（`git` 在 MSYS2 中附带）：

<pre class="devsite-click-to-copy">
<code class="devsite-terminal tfo-terminal-windows">git clone https://github.com/tensorflow/tensorflow.git</code>
<code class="devsite-terminal tfo-terminal-windows">cd tensorflow</code>
</pre>

仓库默认为 `master` 分支。你也可以检出到 [release 分支](https://github.com/tensorflow/tensorflow/releases){:.external} 来构建：

<pre class="devsite-terminal tfo-terminal-windows prettyprint lang-bsh">
git checkout <em>branch_name</em>  # r1.9, r1.10, etc.
</pre>

要测试你的源码版本，运行以下测试（可能需要一小会）：

<pre class="devsite-terminal tfo-terminal-windows devsite-click-to-copy">
bazel test -c opt -- //tensorflow/... -//tensorflow/compiler/... -//tensorflow/contrib/lite/...
</pre>

关键点：如果你在最新的开发分支上遇到了编译问题，试着切换到一个已知可行的发行分支。


## 编译配置

在你的 TensorFlow 源码根目录下运行以下命令来配置系统编译：

<pre class="devsite-terminal tfo-terminal-windows devsite-click-to-copy">
python ./configure.py
</pre>

这个脚本会提示你 TensorFlow 依赖的路径并请求附加的编译配置项（比如编译器标识）。下面展示了运行 `python ./configure.py` 的示例（你的会话可能会有所不同）：

<section class="expandable">
<h4 class="showalways">View sample configuration session</h4>
<pre class="devsite-terminal tfo-terminal-windows">
python ./configure.py
Starting local Bazel server and connecting to it...
................
You have bazel 0.15.0 installed.
Please specify the location of python. [Default is C:\python36\python.exe]: 

Found possible Python library paths:
  C:\python36\lib\site-packages
Please input the desired Python library path to use.  Default is [C:\python36\lib\site-packages]

Do you wish to build TensorFlow with CUDA support? [y/N]: <b>Y</b>
CUDA support will be enabled for TensorFlow.

Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]:

Please specify the location where CUDA 9.0 toolkit is installed. Refer to README.md for more details. [Default is C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0]:

Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: <b>7.0</b>

Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0]: <b>C:\tools\cuda</b>

Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,7.0]: <b>3.7</b>

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is /arch:AVX]: 

Would you like to override eigen strong inline for some C++ compilation to reduce the compilation time? [Y/n]:
Eigen strong inline overridden.

Configuration finished
</pre>
</section>

### 可选配置

如果要开启 [GPU 支持](./gpu.md)，指定 CUDA 和 cuDNN 的版本。如果你的系统安装了多个版本的 CUDA 或者 cuDNN，明确指定版本号，而不要依赖默认配置。`./configure.py` 将会对你的系统的 CUDA 库创建符号连接 —— 所以如果你更新了你的 CUDA 库路径，需要在编译前重新执行这一步。

注意：自 TensorFlow 1.6 起，二进制文件使用 AVX 指令集，这会导致旧的 CPU 不被兼容。

## 编译 pip 包

### Bazel 编译

#### CPU-only

使用 `bazel` 来让 TensorFlow 包编译器仅支持 CPU：

<pre class="devsite-terminal tfo-terminal-windows devsite-click-to-copy">
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
</pre>

#### GPU 支持

如果要让 TensorFlow 包编译器支持 GPU：

<pre class="devsite-terminal tfo-terminal-windows devsite-click-to-copy">
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
</pre>

#### Bazel 编译选项

从源码编译 TensorFlow 很占内存。如果你的系统内存有限，通过 `--local_resources 2048,.5,1.0` 选项来限制 Bazel 的内存占用。

如果开启了 GPU 支持，添加 `--copt=-nvcc_options=disable-warnings` 来忽略 nvcc 警告信息。

### 包编译

`bazel build` 命令创建了一个叫做 `build_pip_package` 的可执行文件 —— 这是用来构建 `pip` 包的程序。举个例子，下面的命令在 `C:/tmp/tensorflow_pkg` 目录下构建了一个 `.whl` 包：

<pre class="devsite-terminal tfo-terminal-windows devsite-click-to-copy">
bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg
</pre>

尽管在同一个源代码根目录下同时构建 CUDA 和非 CUDA 配置是可行的，但我们推荐在两种配置间切换时运行 `bazel clean`。

### 包安装

生成的 `.whl` 文件名取决于 TensorFlow 的版本和你的平台。使用 `pip3 install` 来安装包，示例如下：

<pre class="devsite-terminal tfo-terminal-windows prettyprint lang-bsh">
pip3 install C:/tmp/tensorflow_pkg/tensorflow-<var>version</var>-cp36-cp36m-win_amd64.whl
</pre>

成功：TensorFlow 已被安装。

## 使用 MSYS shell 编译

TensorFlow 可以使用 MSYS shell 编译。只要作下列更改，然后照着上文的步骤使用 Windows 的原生命令行（`cmd.exe`）。

### 关闭 MSYS 路径转换 {:.hide-from-toc}

MSYS 会自动将 Unix 路径转换成 Windows 路径，这会导致 `bazel` 无法正常工作（`//foo/bar:bin` 被视作 Unix 绝对路径因为它以斜杠开始。）

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">export MSYS_NO_PATHCONV=1</code>
<code class="devsite-terminal">export MSYS2_ARG_CONV_EXCL="*"</code>
</pre>

### 设置 PATH {:.hide-from-toc}

将 Bazel 和 Python 安装文件夹添加到 `$PATH` 环境变量。假设 Bazel 安装在 `C:\tools\bazel.exe`，Python 安装在 `C:\Python36\python.exe`，将 `PATH` 设置成：

<pre class="prettyprint lang-bsh">
# Use Unix-style with ':' as separator
<code class="devsite-terminal">export PATH="/c/tools:$PATH"</code>
<code class="devsite-terminal">export PATH="/c/Python36:$PATH"</code>
</pre>

对于 GPU 支持版本，将 CUDA 和 cuDNN 的 bin 目录添加到 `$PATH`：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/bin:$PATH"</code>
<code class="devsite-terminal">export PATH="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0/extras/CUPTI/libx64:$PATH"</code>
<code class="devsite-terminal">export PATH="/c/tools/cuda/bin:$PATH"</code>
</pre>

## 可用编译配置

<table>
<tr><th>Version</th><th>CPU/GPU</th><th>Python version</th><th>Compiler</th><th>Build tools</th><th>cuDNN</th><th>CUDA</th></tr>
<tr><td>tensorflow-1.10.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.10.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.9.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.9.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.8.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.8.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.7.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.7.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.6.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.6.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.5.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.5.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>7</td><td>9</td></tr>
<tr><td>tensorflow-1.4.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.4.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow-1.3.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.3.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>6</td><td>8</td></tr>
<tr><td>tensorflow-1.2.0</td><td>CPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.2.0</td><td>GPU</td><td>3.5-3.6</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.1.0</td><td>CPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.1.0</td><td>GPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
<tr><td>tensorflow-1.0.0</td><td>CPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>N/A</td><td>N/A</td></tr>
<tr><td>tensorflow_gpu-1.0.0</td><td>GPU</td><td>3.5</td><td>MSVC 2015 update 3</td><td>Cmake v3.6.3</td><td>5.1</td><td>8</td></tr>
</table>
