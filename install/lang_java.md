# 安装 TensorFlow for Java

TensorFlow 为 Java 程序提供了 [API](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary) — 对于加载 Python 构建的模型并在 Java 应用程序中运行加载（这些模型）特别有用。

注意：TensorFlow 的 Java API **不**包含在 [TensorFlow API 稳定性保证](../guide/version_compat.md)中。

## 支持平台

TensorFlow for Java 支持以下操作系统：

* Ubuntu 16.04 或更高版本；64 位、x86 架构
* macOS 10.12.6 (Sierra) 或更高版本
* Windows 7 或更高版本；64 位、x86 架构

如果需要在 Android 上安装 TensorFlow，请看 [Android TensorFlow 支持页面](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android){:.external} 或者 [TensorFlow Android 相机 Demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android){:.external}。

## 使用 Apache Maven 的 TensorFlow

如要将 TensorFlow 与 [Apache Maven](https://maven.apache.org){:.external} 结合使用，请将依赖项添加到项目的 `pom.xml` 文件中：

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow</artifactId>
  <version>1.10.1</version>
</dependency>
```

### GPU 支持

如果你的系统带有 [GPU 支持](./gpu.md)，请将以下 TensorFlow 依赖项添加到项目的 `pom.xml` 文件中：

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow</artifactId>
  <version>1.10.1</version>
</dependency>
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow_jni_gpu</artifactId>
  <version>1.10.1</version>
</dependency>
```

### 示例程序

这个示例演示了如何使用 TensorFlow 构建 Apache Maven 项目。首先，将 TensorFlow 依赖项添加到项目的 `pom.xml` 文件中：

```xml
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.myorg</groupId>
  <artifactId>hellotensorflow</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <exec.mainClass>HelloTensorFlow</exec.mainClass>
	<!-- 实例代码需要使用 JDK 1.7 以上 -->
	<!-- maven 编译器插件默认为较低版本 -->
	<maven.compiler.source>1.7</maven.compiler.source>
	<maven.compiler.target>1.7</maven.compiler.target>
  </properties>
  <dependencies>
    <dependency>
	  <groupId>org.tensorflow</groupId>
	  <artifactId>tensorflow</artifactId>
	  <version>1.10.1</version>
	</dependency>
  </dependencies>
</project>
```

创建源文件 (`src/main/java/HelloTensorFlow.java`)：

```java
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class HelloTensorFlow {
  public static void main(String[] args) throws Exception {
	try (Graph g = new Graph()) {
	  final String value = "Hello from " + TensorFlow.version();

	  // Construct the computation graph with a single operation, a constant
	  // named "MyConst" with a value "value".
	  try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
	    // The Java API doesn't yet include convenience functions for adding operations.
		g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
	  }

	  // Execute the "MyConst" operation in a Session.
	  try (Session s = new Session(g);
	      // Generally, there may be multiple output tensors,
		  // all of them must be closed to prevent resource leaks.
		  Tensor output = s.runner().fetch("MyConst").run().get(0)) {
	    System.out.println(new String(output.bytesValue(), "UTF-8"));
	  }
    }
  }
}
```

编译执行：

<pre class="devsite-terminal prettyprint lang-bsh">
mvn -q compile exec:java  # Use -q to hide logging
</pre>

命令行将输出：<code>Hello from <em>version</em></code>

完成: TensorFlow for Java 已经配置好了。

## 使用 JDK 的 TensorFlow

TensorFlow 可以通过 Java Native Interface (JNI) 与 JDK 一起使用。

### 下载

1. 下载 TensorFlow Jar 包（JAR）：[libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.10.1.jar)
2. 下载并解压缩适用于你的操作系统和处理器支持的 Java Native Interface（JNI）文件：

<table>
  <tr><th>JNI version</th><th>URL</th></tr>
  <tr class="alt"><td colspan="2">Linux</td></tr>
  <tr>
    <td>Linux CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.10.1.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.10.1.tar.gz</a></td>
  </tr>
  <tr>
    <td>Linux GPU support</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.10.1.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.10.1.tar.gz</a></td>
  </tr>
  <tr class="alt"><td colspan="2">macOS</td></tr>
  <tr>
    <td>macOS CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.10.1.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.10.1.tar.gz</a></td>
  </tr>
  <tr class="alt"><td colspan="2">Windows</td></tr>
  <tr>
    <td>Windows CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.10.0.zip">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.10.0.zip</a></td>
  </tr>
  <tr>
    <td>Windows GPU support</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-windows-x86_64-1.10.0.zip">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-windows-x86_64-1.10.0.zip</a></td>
  </tr>
</table>

注意：在 Windows 上，本地库（`tensorflow_jni.dll`）在运行时需要 `msvcp140.dll`。参见 [Windows 源码安装](./source_windows.md)指南，了解如何安装 [Visual C++ 2015 Redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=48145){:.external}。

### 编译

使用[上一个示例](#example)中的 `HelloTensorFlow.java` 文件，编译使用 TensorFlow 的程序。请确保 `classpath` 可以访问 `libtensorflow.jar`：

<pre class="devsite-terminal devsite-click-to-copy">
javac -cp libtensorflow-1.10.1.jar HelloTensorFlow.java
</pre>

### 运行

要执行 TensorFlow Java 程序，JVM 必须访问 `libtensorflow.jar` 和提取的 JNI 库。

<div class="ds-selector-tabs">
<section>
<h3>在 Linux / mac OS 上</h3>
<pre class="devsite-terminal devsite-click-to-copy">java -cp libtensorflow-1.10.1.jar:. -Djava.library.path=./jni HelloTensorFlow</pre>
</section>
<section>
<h3>在 Windows 上</h3>
<pre class="devsite-terminal tfo-terminal-windows devsite-click-to-copy">java -cp libtensorflow-1.10.1.jar;. -Djava.library.path=jni HelloTensorFlow</pre>
</section>
</div><!--/ds-selector-tabs-->

命令行输出：<code>Hello from <em>version</em></code>

完成：TensorFlow for Java 已经配置好了。

## 从源码构建

TensorFlow 是开源的。阅读[说明书](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/README.md){:.external}，了解如何从源代码构建 TensorFlow 的 Java 和本机库。
