# 安装 TensorFlow C API

TensorFlow 提供了用于构建[绑定其他语言](../extend/language_bindings.md)的 C API。这些 API 定义在 <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/c_api.h" class="external"><code>c_api.h</code></a> 中，并且优先考虑其设计得简洁与一致而非便利。

## 支持平台

TensorFlow C API 支持以下系统:

* Linux，64-bit，x86
* macOS X，10.12.6 版本（Sierra）或更高

## 步骤

### 下载

<table>
  <tr><th>TensorFlow C 函数库 </th><th>URL</th></tr>
  <tr class="alt"><td colspan="2">Linux</td></tr>
  <tr>
    <td>Linux 下仅支持 CPU </td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.10.1.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.10.1.tar.gz</a></td>
  </tr>
  <tr>
    <td>Linux 下支持 GPU</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.10.1.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.10.1.tar.gz</a></td>
  </tr>
  <tr class="alt"><td colspan="2">macOS</td></tr>
  <tr>
    <td>macOS 下仅支持 CPU</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.10.1.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.10.1.tar.gz</a></td>
  </tr>
</table>

### 提取

将下载的 TensorFlow C 函数库提取到 `/usr/local/lib` 下（或使用另一目录）：

<pre class="devsite-terminal devsite-click-to-copy">
sudo tar -xz <var>libtensorflow.tar.gz</var> -C /usr/local
</pre>

### 链接器

如果你将 TensorFlow C 函数库提取到一个系统目录下，比如 `/usr/local`，使用 `ldconfig` 设置链接器：

<pre class="devsite-terminal devsite-click-to-copy">
sudo ldconfig
</pre>

或者，如果你将 TensorFlow C 函数库提取到一个非系统目录下，比如 `~/mydir`，那设置链接器环境变量：

<div class="ds-selector-tabs">
<section>
<h3>Linux</h3>
<pre class="prettyprint lang-bsh">
export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib
</pre>
</section>
<section>
<h3>mac OS</h3>
<pre class="prettyprint lang-bsh">
export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib
export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/mydir/lib
</pre>
</section>
</div><!--/ds-selector-tabs-->

## 构建

### 示例程序

安装完成 TensorFlow C 函数库后，按照以下源代码创建一个示例程序（`hello_tf.c`）：

```c
#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main() {
  printf("Hello from TensorFlow C library version %s\n", TF_Version());
  return 0;
}
```

### 编译

将示例程序编译创建出可执行文件，然后运行：

<pre class="prettyprint lang-bsh">
<code class="devsite-terminal">gcc hello_tf.c -o hello_tf</code>

<code class="devsite-terminal">./hello_tf</code>
</pre>

这个命令会输出：<code>Hello from TensorFlow C library version <em>number</em></code>

成功：TensorFlow C 函数库配置完成。

如果程序没能够构建成功，确定 `gcc` 可以访问 TensorFlow C 函数库。如果提取到 `/usr/local`，请将函数的库的目录地址传递给编译器：

<pre class="devsite-terminal devsite-click-to-copy">
gcc -I/usr/local/include -L/usr/local/lib hello_tf.c -ltensorflow -o hello_tf
</pre>

## 由源码构建

TensorFlow 已开源。查看[说明](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/lib_package/README.md){:.external}了解如何由源码构建 TensorFlow C 函数库。
