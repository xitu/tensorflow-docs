# 安装 TensorFlow for Java

TensorFlow 为 Java 程序提供了 API 。这些 API 是在 Java 应用中专门用来加载和执行 Python 创建的模型的。这个教程解释了如何安装并在应用中使用
[TensorFlow for Java](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)
。

警告：TensorFlow 的 Java API **不**包含在 [TensorFlow API 稳定性保证](https://www.tensorflow.org/programmers_guide/version_semantics)中。

## 支持平台

This guide explains how to install TensorFlow for Java.  Although these instructions might also work on other variants, we have only tested (and we only support) these instructions on machines meeting the following requirements:

 * Ubuntu 16.04 or higher; 64-bit, x86
 * macOS X, 10.12.6 (Sierra) 或更高
 * Windows 7 or higher; 64-bit, x86

Android 上的安装说明在单独的
[Android TensorFlow 支持页](https://www.tensorflow.org/code/tensorflow/contrib/android)
中。在安装完成后，请查看这个 Android 上 TensorFlow 的
[完整示例](https://www.tensorflow.org/code/tensorflow/examples/android)。

## 在 Maven 项目中使用 TensorFlow

如果你的项目使用了 [Apache Maven](https://maven.apache.org)，为了使用 TensorFlow Java API ，
在项目的 `pom.xml` 中加入以下内容即可：

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow</artifactId>
  <version>1.8.0-rc1</version>
</dependency>
```

### 示例

例如，这些步骤将创建一个使用 TensorFlow 的 Maven 项目：

  1. 创建项目的 `pom.xml`：


         <project>
             <modelVersion>4.0.0</modelVersion>
             <groupId>org.myorg</groupId>
             <artifactId>hellotf</artifactId>
             <version>1.0-SNAPSHOT</version>
             <properties>
               <exec.mainClass>HelloTF</exec.mainClass>
               <!-- 这个样例代码至少需要 JDK 1.7 。 -->
               <!-- maven 编译器插件默认为一个更低的版本 -->
               <maven.compiler.source>1.7</maven.compiler.source>
               <maven.compiler.target>1.7</maven.compiler.target>
             </properties>
             <dependencies>
               <dependency>
                 <groupId>org.tensorflow</groupId>
                 <artifactId>tensorflow</artifactId>
                 <version>1.8.0-rc1</version>
               </dependency>
             </dependencies>
         </project>


  2. 创建源文件（`src/main/java/HelloTF.java`）：


        import org.tensorflow.Graph;
        import org.tensorflow.Session;
        import org.tensorflow.Tensor;
        import org.tensorflow.TensorFlow;

        public class HelloTF {
          public static void main(String[] args) throws Exception {
            try (Graph g = new Graph()) {
              final String value = "Hello from " + TensorFlow.version();

              // 使用一个简单操作、一个名为 "MyConst" 的常数和一个值 "value" 来构建计算图。
              try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
                // Java API 目前还不包含足够方便的函数来执行“加”操作。
                g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
              }

              // 在一个 Session 中执行 "MyConst" 操作。
              try (Session s = new Session(g);
                   Tensor output = s.runner().fetch("MyConst").run().get(0)) {
                System.out.println(new String(output.bytesValue(), "UTF-8"));
              }
            }
          }
        }

  3. 编译并执行：

     <pre> # 使用 -q 来隐藏 mvn 工具的日志
     <b>mvn -q compile exec:java</b></pre>


前面的这条命令应该输出 <tt>Hello from <i>version</i></tt> 。 如果成功输出，那么你就已经成功地安装了 TensorFlow for Java 并且可以在Maven 项目中使用它。如果没有成功，请前往
[Stack Overflow](http://stackoverflow.com/questions/tagged/tensorflow)
搜索可能的解决方案。你可以跳过阅读本文档的其余部分。

### GPU support

If your Linux system has an NVIDIA® GPU and your TensorFlow Java program requires GPU acceleration, then add the following to the project's `pom.xml` instead:

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow</artifactId>
  <version>1.8.0-rc1</version>
</dependency>
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow_jni_gpu</artifactId>
  <version>1.8.0-rc1</version>
</dependency>
```

GPU acceleration is available via Maven only for Linux and only if your system meets the @{$install_linux#determine_which_tensorflow_to_install$requirements for GPU}.

## 在 JDK 下使用 TensorFlow

这一节将介绍如何使用 JDK 安装得到的 java 和 javac 命令来使用 TensorFlow。如果你的项目中使用了 Apache Maven，请参考使用上一节更简单的安装方法。


### 在 Linux 或 macOS 上安装

采取以下步骤在 Linux 或 macOS 上安装 TensorFlow for Java：

  1. 下载
     [libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.8.0-rc1.jar)，
     它是 TensorFlow Java Archive (JAR)。

  2. 决定你要只在 CPU 上运行 TensorFlow for Java 还是要在 GPU 的协助下运行。为了帮助您决定，请阅读以下指南中标题为“决定要安装哪个 TensorFlow”的部分：

     * @{$install_linux#determine_which_tensorflow_to_install$Installing TensorFlow on Linux}
     * @{$install_mac#determine_which_tensorflow_to_install$Installing TensorFlow on macOS}

  3. 通过运行以下 shell 命令，下载并提取相应的 Java Native Interface（JNI）文件，来为你的操作系统和处理器提供支持：


         TF_TYPE="cpu" # 默认处理器是 CPU 。如果你想要使用 GPU ，就将它设置成 "gpu" 。
         OS=$(uname -s | tr '[:upper:]' '[:lower:]')
         mkdir -p ./jni
         curl -L \
           "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-${TF_TYPE}-${OS}-x86_64-1.8.0-rc1.tar.gz" |
           tar -xz -C ./jni

### 在 Windows 上安装

用如下几步在 Windows 上安装 TensorFlow for Java ：
  1. 下载
     [libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.8.0-rc1.jar)，
    它是 TensorFlow Java Archive (JAR)。
  2. 下载适合 Windows 上的 TensorFlow for Java 的 [Java Native Interface (JNI) 文件](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.8.0-rc1.zip)。
  3. 解压此 .zip 文件。



### 验证安装

安装 TensorFlow for Java 后，在 `HelloTF.java` 文件中输入以下代码来验证安装：

```Java
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class HelloTF {
  public static void main(String[] args) throws Exception {
    try (Graph g = new Graph()) {
      final String value = "Hello from " + TensorFlow.version();

      // 使用一个简单操作、一个名为 "MyConst" 的常数和一个值 "value" 来构建计算图。
      try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
        // Java API 目前还不包含足够方便的函数来执行“加”操作。
        g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
      }

      // 在一个 Session 中执行 "MyConst" 操作。
      try (Session s = new Session(g);
           Tensor output = s.runner().fetch("MyConst").run().get(0)) {
        System.out.println(new String(output.bytesValue(), "UTF-8"));
      }
    }
  }
}
```

并使用以下命令来编译并运行 `HelloTF.java` 。


### 编译

在编译一个使用 TensorFlow 的 Java 程序时，下载的 `.jar` 文件必须在你的 `classpath` 中。例如，你可以通过使用类似如下的指令，使用编译标志 `-cp` 将下载的 `.jar` 文件包含在你的 `classpath` 中：

<pre><b>javac -cp libtensorflow-1.8.0-rc1.jar HelloTF.java</b></pre>


### 运行

要运行依赖 TensorFlow 的 Java 程序，保证下面的
两个文件对于 JVM 来说可用：

  * 下载好的 `.jar` 文件
  * 提取出的 JNI 库

例如，使用以下命令命令在 Linux 和 macOS X 上运行 `HelloTF` 程序：

<pre><b>java -cp libtensorflow-1.8.0-rc1.jar:. -Djava.library.path=./jni HelloTF</b></pre>

使用以下命令在 Windows 上运行 `HelloTF` 程序：

<pre><b>java -cp libtensorflow-1.8.0-rc1.jar;. -Djava.library.path=jni HelloTF</b></pre>

如果程序打印出 <tt>Hello from <i>version</i></tt>，说明你已经成功地安装了 TensorFlow for Java 并且可以使用 API 了。
如果程序输出了其他内容，请查阅
[Stack Overflow](http://stackoverflow.com/questions/tagged/tensorflow)
以寻找解决方案。


### 高级示例

有关更复杂的示例，请参考
[LabelImage.java](https://www.tensorflow.org/code/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java)，它可以识别图像中的物体。


## 从源代码构建

TensorFlow 是开源的。你可以根据这个
[单独的文档](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/README.md)
中的指示从源代码编译 TensorFlow for Java。
