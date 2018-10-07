# 安装 TensorFlow for C 语言

TensorFlow 在 [`c_api.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h) 中定义了一套 C 语言 API，用于[构建其它语言的封装](https://www.tensorflow.org/extend/language_bindings)。这套 API 除了使用方便外，还将保持易用性与一致性。

## 支持的平台

本指南解释了如何安装 TensorFlow C 语言版。尽管本指南也可能适用于其它的安装环境，但我们仅测试（也仅确保）了本指南在以下环境机器中的适用性：

  * Linux, 64-bit, x86
  * macOS X, Version 10.12.6 (Sierra) 或更高版本

## 安装

请按照以下步骤安装 TensorFlow C 语言库并启用 TensorFlow C 语言版：

1. 确定你是只在 CPU(s) 上运行 TensorFlow C 语言版，还是在 GPU(s) 的协助下运行。如果无法确定，请在以下指南中阅读“决定安装哪个 TensorFlow”一节：

  *  [在 Ubuntu 上安装 TensorFlow](./install_linux.md#determine_which_tensorflow_to_install)
  *  [在 macOS 上安装 TensorFlow](./install_mac.md#determine_which_tensorflow_to_install)

2. 通过调用下面的 shell 命令，下载并且解压 TensorFlow 的 C 语言库到 `/usr/local/lib`：
  
         TF_TYPE="cpu" # Change to "gpu" for GPU support
         OS="linux" # Change to "darwin" for macOS
         TARGET_DIRECTORY="/usr/local"
         curl -L \
           "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.10.0.tar.gz" |
           sudo tar -C $TARGET_DIRECTORY -xz

`tar` 命令会将 TensorFlow C 语言库解压到 `TARGET_DIRECTORY` 的子目录 `lib` 中。例如，指定 `/usr/local` 作为 `TARGET_DIRECTORY`，那么 `tar` 就会将 TensorFlow C 语言库解压到 `/usr/local/lib` 中。

如果你希望将库解压到不同的目录中，请调整 `TARGET_DIRECTORY`。

  3. 在上一步中，如果你指定了一个系统目录（比如，`/usr/local`）作为 `TARGET_DIRECTORY`，那么请运行 `ldconfig` 配置链接器。例如：
  
  <pre><b>sudo ldconfig</b></pre>
  
  如果你指定了一个非系统目录作为 `TARGET_DIRECTORY`（比如，`~/mydir`），那么你必须将你的解压目录（比如，`~/mydir/lib`）添加到以下两个环境变量中。例如：
  
  <pre>
  <b>export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib</b> # For both Linux and macOS X
  <b>export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib</b> # For Linux only
  <b>export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/mydir/lib</b> # For macOS X only</pre>

## 验证你的安装

在安装完成之后，新建文件，输入以下代码，文件命名为 `hello_tf.c`:

```c
#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());
  return 0;
}
```

### 编译和运行

调用以下命令来编译 `hello_tf.c`

<pre><b>gcc hello_tf.c</b></pre>

运行生成的可执行文件，应该得到以下输出：

<pre><b>a.out</b>
Hello from TensorFlow C library version <i>number</i></pre>

### 定位问题

如果程序编译失败，最有可能的错误是 `gcc` 找不到 TensorFlow C 语言库。解决这个问题的方法是为 `gcc` 指定 `-I` 和 `-L` 参数。比如，`TARGET_LIBRARY` 为 `/usr/local` 时，可以这样调用 `gcc`：

<pre><b>gcc -I/usr/local/include -L/usr/local/lib hello_tf.c -ltensorflow</b></pre>

如果 `a.out` 执行失败，请思考下列问题：：

  * 这个程序编译是否出错？
  * 是否按本指南第三步“[安装](#安装)”指定了正确的环境变量的目录？
  * 是否正确地 `export` 了环境变量？

如果你仍然在编译或者运行时看到了错误信息，请访问 [StackOverflow](https://stackoverflow.com/questions/tagged/tensorflow) 寻求解决方案。
