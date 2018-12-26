# 安装 TensorFlow Go API

TensorFlow 提供了 [Go API](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go){:.external}，这对于加载由 Python 创建的模型并在 Go 应用中运行它们特别有用。

注意：TensorFlow GO API **并不**包含在  TensorFlow 中。
[API 稳定性说明](../guide/version_compat.md)。

## 支持平台

TensorFlow for Go 支持以下系统：

* Linux, 64-bit, x86
* macOS X, Version 10.12.6 (Sierra) or higher

## 步骤

### TensorFlow C library

安装 [TensorFlow C 函数库](./lang_c.md)，它是 TensorFlow Go 包所必需的。

### 下载

下载并安装 TensorFlow Go 包和其依赖：

<pre class="devsite-terminal devsite-click-to-copy">
go get github.com/tensorflow/tensorflow/tensorflow/go
</pre>

并验证安装成功：

<pre class="devsite-terminal devsite-click-to-copy">
go test github.com/tensorflow/tensorflow/tensorflow/go
</pre>

## 构建

### 示例程序

在 TensorFlow Go 包安装完成后，按照以下源码创建一个示例程序（`hello_tf.go`）：

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

### 运行

运行示例程序：

<pre class="devsite-terminal devsite-click-to-copy">
go run hello_tf.go
</pre>

此命令会输出：<code>Hello from TensorFlow version <em>number</em></code>

成功：TensorFlow for Go 配置完成。

程序可能会产生如下的警告，可忽视：

<pre>
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use *Type* instructions, but these are available on your machine and could speed up CPU computations.
</pre>

## 由源码构建

TensorFlow 已开源。查看[说明](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/README.md){:.external} 来有源码构建 TensorFlow for Go。
