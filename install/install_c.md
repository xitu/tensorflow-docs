# Installing TensorFlow for C

TensorFlow 在 [`c_api.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h)中定义了一套 C API，用来提供适合于[建立和其他语言的绑定](https://www.tensorflow.org/extend/language_bindings).
这套 API 倾向于简单性和一致性，而不是方便。

## 支持的平台

This guide explains how to install TensorFlow for C.  Although these instructions might also work on other variants, we have only tested (and we only support) these instructions on machines meeting the following requirements:

 * Linux, 64-bit, x86
 * Mac OS X, Version 10.11 (El Capitan) or higher


## 安装
采取下面几步来安装用于 C 的 TensorFlow 库，然后打开用于 C 的 TensorFlow：
  1.选择你将会仅仅运行用于 C 的 TensoFlow 在 CPU（S）上，还是有 GPU（S）的帮助。为了帮你做出选择，在以下指南中阅读这一节，标题为决定安装哪个TensorFlow”：
       * @{$install_linux#determine_which_tensorflow_to_install$Installing TensorFlow on Linux}
       * @{$install_mac#determine_which_tensorflow_to_install$Installing TensorFlow on macOS}

  2.通过调用下面的 shell 命令，下载并且解压 TensorFlow 的 C 库到 `/usr/local/lib`：
         TF_TYPE="cpu" # Change to "gpu" for GPU support
         OS="linux" # Change to "darwin" for macOS
         TARGET_DIRECTORY="/usr/local"
         curl -L \
           "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-${OS}-x86_64-1.6.0.tar.gz" |
           sudo tar -C $TARGET_DIRECTORY -xz

`tar` 命令会解压 TensorFlow C 库到 `TARGET_DIRECTORY` 的子目录 `lib`中。比如指定 `/usr/local` 作为 `TARGET_DIRECTORY`，那么 `tar` 就会解压TensorFlow C 库到 `/usr/local/lib`。

如果你更希望解压库到不同的目录，那么相应的调整 `TARGET_DIRECTORY`。

  3. 在上一步中，如果你指定了一个系统目录（比如，`/usr/local`)作为 `TARGET_DIRECTORY`，然后运行 `ldconfig` 配置链接器。
  比如：
  <pre><b>sudo ldconfig</b></pre>
  如果你指定了一个 `TARGET_DIRECTORY` 而不是系统目录，（比如，`~/mydir`），那么你必须设定你的解压目录（比如，`~/mydir/lib`）到两个环境变量中。
  比如:
  <pre> <b>export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib</b> # For both Linux and macOS X
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
运行生成的可执行文件应该输出以下消息:
<pre><b>a.out</b>
Hello from TensorFlow C library version <i>number</i></pre>

### 定位问题

如果程序编译失败，最有可能的错误是 `gcc` 找不到 TensorFlow C 库.解决这个问题的方法是为 `gcc` 指定 `-I` 和 `-L` 选项.比如，`TARGET_LIBRARY` 是`/usr/local`，你应该这样调用 `gcc`：

<pre><b>gcc -I/usr/local/include -L/usr/local/lib hello_tf.c -ltensorflow</b></pre>

如果执行 `a.out` 失败,你就要问问自己这几个问题了：
  * 这个程序编译有没有错误？
  * 是否按第三步 [安装](#安装), 指定了正确的环境变量的目录?
  * 是否有正确的 `export` 这些环境变量?

如果你仍然会有编译或者运行的错误信息, 请到 [StackOverflow](www.stackoverflow.com/questions/tagged/tensorflow) 寻找或者请求可能的解决方案.
