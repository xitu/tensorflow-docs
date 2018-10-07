# 安装 TensorFlow for Java

TensorFlow 为 Java 程序提供了 API 。这些 API 是在 Java 应用中专门用来加载和执行 Python 创建的模型的。这个教程解释了如何安装并在应用中使用
[TensorFlow for Java](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)
。

警告：TensorFlow 的 Java API 不包含在 [TensorFlow API 稳定性保证](../guide/version_semantics.md)中。

## 支持平台

本指南介绍如何安装适用于 Java 的 TensorFlow。虽然这些说明可能也适用于其他配置，但我们只在满足以下要求的计算机上验证过这些说明（而且我们只支持在此类计算机上按这些说明操作）：

 * Ubuntu 16.04 或更高版本；64 位、x86
 * macOS X 10.11 (El Capitan) 或更高版本
 * Windows 7 或更高版本；64 位、x86

Android 上的安装说明在单独的 [Android TensorFlow 支持页面](https://www.tensorflow.org/code/tensorflow/contrib/android)中。在安装完成后，请查看这个适用于 Android 的[完整 TensorFlow 示例](https://www.tensorflow.org/code/tensorflow/examples/android)。

## 在 Maven 项目中使用 TensorFlow

如果你的项目使用了 [Apache Maven](https://maven.apache.org)，为了使用 TensorFlow Java API ，
在项目的 `pom.xml` 中加入以下内容即可：

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow</artifactId>
  <version>1.10.0</version>
</dependency>
```

就这么简单。

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
                 <version>1.10.0</version>
               </dependency>
             </dependencies>
         </project>

  2. 创建源文件（`src/main/java/HelloTF.java`）：

        ```
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
                   // 通常，可能存在多个输出 tensor，必须关闭所有输出 tensor 以防止资源泄漏。
                   Tensor output = s.runner().fetch("MyConst").run().get(0)) {
                System.out.println(new String(output.bytesValue(), "UTF-8"));
              }
            }
          }
        }
        ```

  3. 编译并执行：

     <pre>
     # 使用 -q 来隐藏 mvn 工具的日志
     <b>mvn -q compile exec:java</b></pre>

前面的这条命令应该输出 <tt>Hello from <i>version</i></tt>。 如果成功输出，那么你就已经成功地安装了 TensorFlow for Java 并且可以在 Maven 项目中使用它。如果没有成功，请前往 [Stack Overflow](http://stackoverflow.com/questions/tagged/tensorflow) 查找可行的解决方案。你可以跳过本文档的其余部分。

### GPU support

如果你的 Linux 系统搭载了 NVIDIA® GPU 且你的 TensorFlow Java 程序需要 GPU 加速，请将以下内容添加到项目的 `pom.xml`：

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow</artifactId>
  <version>1.10.0</version>
</dependency>
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow_jni_gpu</artifactId>
  <version>1.10.0</version>
</dependency>
```

只有当你的系统是 Linux 且满足 [GPU 的要求](./install_linux.md#determine_which_tensorflow_to_install)时，才能通过 Maven 使用 GPU 加速。

## 在 JDK 中使用 TensorFlow

这一节将介绍如何使用 JDK 安装得到的 java 和 javac 命令来使用 TensorFlow。如果你的项目中使用了 Apache Maven，请参考使用上一节更简单的安装方法。

### 在 Linux 或 macOS 上安装

采取以下步骤在 Linux 或 macOS 上安装 TensorFlow for Java：

1. 下载 [libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.10.0.jar)，它是 TensorFlow Java Archive (JAR)。

2. 决定你要只在 CPU 上运行 TensorFlow for Java 还是要在 GPU 的协助下运行。为了帮助你决定，请阅读以下指南中标题为“决定要安装哪个 TensorFlow”的部分：

* [在 Ubuntu 上安装 TensorFlow](./install_linux.md#determine_which_tensorflow_to_install)
* [在 macOS 上安装 TensorFlow](./install_mac.md#determine_which_tensorflow_to_install)

3. 通过运行以下 shell 命令，下载并提取相应的 Java Native Interface（JNI）文件，来为你的操作系统和处理器提供支持：

       TF_TYPE="cpu" # 默认处理器是 CPU。如果你想要使用 GPU，就将它设置成 "gpu"。
       OS=$(uname -s | tr '[:upper:]' '[:lower:]')
       mkdir -p ./jni
       curl -L \
         "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-${TF_TYPE}-${OS}-x86_64-1.10.0.tar.gz" |
         tar -xz -C ./jni

### 在 Windows 上安装

用如下几步在 Windows 上安装 TensorFlow for Java ：

  1. 下载 [libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.10.0.jar)，它是 TensorFlow Java Archive (JAR)。
  2. 下载适合 Windows 上的 TensorFlow for Java 的 [Java Native Interface (JNI) 文件](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.10.0.zip)。
  3. 解压此 .zip 文件。

**注意**：Native library `tensorflow_jni.dll` 在运行时需要 `msvcp140.dll`，它包含在 [Visual C++ 2015 Redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=48145) 包中。

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
           // 通常，可能存在多个输出 Tensor，必须关闭所有的输出 Tensor 以防止资源泄漏。
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

<pre><b>javac -cp libtensorflow-1.10.0.jar HelloTF.java</b></pre>

### 运行

要运行依赖 TensorFlow 的 Java 程序，保证下面的
两个文件对于 JVM 来说可用：

  * 下载好的 `.jar` 文件
  * 提取出的 JNI 库

例如，使用以下命令命令在 Linux 和 macOS X 上运行 `HelloTF` 程序：

<pre><b>java -cp libtensorflow-1.10.0.jar:. -Djava.library.path=./jni HelloTF</b></pre>

使用以下命令在 Windows 上运行 `HelloTF` 程序：

<pre><b>java -cp libtensorflow-1.10.0.jar;. -Djava.library.path=jni HelloTF</b></pre>

如果程序打印出 <tt>Hello from <i>version</i></tt>，说明你已经成功地安装了 TensorFlow for Java 并且可以使用 API 了。如果程序输出了其他内容，请访问  [Stack Overflow](http://stackoverflow.com/questions/tagged/tensorflow) 查找可行的解决方案。

### 高级示例

有关更复杂的示例，请参考 [LabelImage.java](https://www.tensorflow.org/code/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java)，它可以识别图像中的物体。

## 从源代码构建

TensorFlow 是开源的。你可以按照[另一份文档](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/README.md)中的说明从 TensorFlow 源代码构建适用于 Java 的 TensorFlow。
