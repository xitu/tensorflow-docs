 # 安装 Tensorflow for Go

TensorFlow 提供了 Go 程序中可以调用的 API。这些 API 非常适合加载 Python 创建的模型以及在 Go 应用中执行。本文将介绍如何安装和配置 [TensorFlow Go 包](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)。

警告：TensorFlow Go API 不在 TensorFlow [API 稳定性保障](../guide/version_semantics.md)的涵盖范围内。

## 支持的平台

这篇教程主要讲述如何安装 Go 版本 TensorFlow。虽然这些命令可能适用于其他平台，但我们现在仅在以下配置环境下进行过测试：

  * Linux, 64-bit, x86
  * macOS X, 10.12.6 (Sierra) 或更高版本

## 安装

Go 版本 TensorFlow 依赖于 TensorFlow C 语言库。按照下面的步骤安装这个库并启用 TensorFlow：

  1. 决定在运行 TensorFlow 时仅仅启用 CPU 还是和 GPU 一起启用。为了帮助你做这个决定，请阅读以下指南中的“决定安装哪个 TensorFlow ”部分：

    *  [在 Ubuntu 上安装 TensorFlow](./install_linux.md#determine_which_tensorflow_to_install)
    *  [在 macOS 上安装 TensorFlow](./install_mac.md#determine_which_tensorflow_to_install)

  2. 通过执行以下命令下载并解压 TensorFlow C 语言库到 `/usr/local/lib` 目录:

         TF_TYPE="cpu" # Change to "gpu" for GPU support
         TARGET_DIRECTORY='/usr/local'
         curl -L \
           "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.10.0.tar.gz" |
         sudo tar -C $TARGET_DIRECTORY -xz

     `tar` 命令会解压 TensorFlow C 语言库到 `TARGET_DIRECTORY` 的子目录 `lib`。比如，指定 `/usr/local` 作为 `TARGET_DIRECTORY` 使得 `tar` 命令可以将 TensorFlow C 语言库解压到 `/usr/local/lib`。
     如果你想把库文件解压到其他目录，更换 `TARGET_DIRECTORY` 就可以了。

  3. 在第二步中，如果你指定了一个系统目录（比如 `/usr/local`）作为 `TARGET_DIRECTORY`，那么需要运行 `ldconfig` 来配置链接。例如：

     <pre><b>sudo ldconfig</b></pre>

     如果你指定的 `TARGET_DIRECTORY` 不是一个系统目录（比如 `~/mydir`），那么你必须要将这个解压目录（比如 `~/mydir/lib`）添加到下面这两个环境变量中：

     <pre>
     <b>export LIBRARY_PATH=$LIBRARY_PATH:~/mydir/lib</b> # 用于 Linux 和 macOS X
     <b>export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/mydir/lib</b> # 仅用于 Linux
     <b>export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:~/mydir/lib</b> # 仅用于 macOS</pre>

  4. 现在 TensorFlow C 语言库已经安装好了，执行 `go get` 来下载对应的包和相应的依赖：

     <pre><b>go get github.com/tensorflow/tensorflow/tensorflow/go</b></pre>

  5. 执行 `go test` 来验证 Go 版本 TensorFlow 是否安装成功： 

     <pre><b>go test github.com/tensorflow/tensorflow/tensorflow/go</b></pre>

如果 `go get` 或者 `go test` 产生错误信息了，可以在 [StackOverflow](http://www.stackoverflow.com/questions/tagged/tensorflow) 上通过搜索和提问来获取可能的解决方法。


## Hello World

安装完 Go 版本 TensorFlow 之后，在 `hello_tf.go` 文件中输入下面的代码：

```go
package main

import (
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"fmt"
)

func main() {
	// Construct a graph with an operation that produces a string constant.
	s := op.NewScope()
	c := op.Const(s, "Hello from TensorFlow version " + tf.Version())
	graph, err := s.Finalize()
	if err != nil {
		panic(err)
	}

	// Execute the graph in a session.
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}
	output, err := sess.Run(nil, []tf.Output{c}, nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(output[0].Value())
}
```

关于 TensorFlow Go 语言的进阶示例请查看 [API 文档中的示例](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go#ex-package)，这个例子使用了一个通过 TensorFlow 预训练的模型来标记图片的内容。

### 运行

通过调用下面的命令来运行 `hello_tf.go`：

<pre><b>go run hello_tf.go</b>
Hello from TensorFlow version <i>number</i></pre>

程序也可能会生成以下形式的多条警告消息，你可以忽略：

<pre>W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library
wasn't compiled to use *Type* instructions, but these are available on your
machine and could speed up CPU computations.</pre>


## 使用源码编译

TensorFlow 是开源系统。你可以按照[另一份文档](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/README.md)中的说明从 TensorFlow 源代码构建适用于 Go 的 TensorFlow。
