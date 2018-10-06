# 如何在 Hadoop 上运行 Tensorflow

本文主要内容是如何在 Hadoop 上运行 Tensorflow。本文目前只写了 Tensorflow 在 Hadoop 分布式文件系统（HDFS）上如何运行，将来会扩充到在各种集群管理器上如何运行。

## HDFS

我们默认读者已经了解如何[读取数据](../api_guides/python/reading_data.md)。

为了在 HDFS 上使用 Tensorflow，需要将读写数据的文件路径改为 HDFS 路径。例如：

```python
filename_queue = tf.train.string_input_producer([
    "hdfs://namenode:8020/path/to/file1.csv",
    "hdfs://namenode:8020/path/to/file2.csv",
])
```

如果读者需要在 HDFS 配置文件中使用特定的 namenode，则可以将文件前缀改为 `hdfs://default/`.

当创建 Tensorflow 项目时，必须配置以下环境变量：

*   **JAVA_HOME**：JAVA 安装路径。
*   **HADOOP_HDFS_HOME**：HDFS 安装路径。也可以通过运行以下代码来配置该环境变量：

    ```shell
    source ${HADOOP_HOME}/libexec/hadoop-config.sh
    ```

*   **LD_LIBRARY_PATH**：包括 libjvm.so 和 libhdfs.so（可选）的路径。如果你的 Hadoop 版本没有在 `$HADOOP_HDFS_HOME/lib/natice` 路径下安装 libhdfs.so，则需要在 Linux 上执行以下操作：

    ```shell
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server
    ```

*   **CLASSPATH**：在运行 Tensoflow 项目之前不止需要添加 Hadoop 的 jar 包。通过 `${HADOOP_HOME}/libexec/hadoop-config.sh` 创建的 CLASSPATH 路径不满足条件，还需要在 libhdfs 文档中添加 glob：

    ```shell
    CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python your_script.py
    ```
    如果你使用的是比 Hadoop/libhdfs 2.6.0 更旧的版本，则应该自己添加 classpath 通配符。详情请见：[HADOOP-10903](https://issues.apache.org/jira/browse/HADOOP-10903).

如果 Hadoop 集群在安全模式下，要配置以下环境变量：

*   **KRB5CCNAME**：Kerberos 票据缓存文件路径。例如：

    ```shell
    export KRB5CCNAME=/tmp/krb5cc_10002
    ```

如果要运行[分布式 TensorFlow](../deploy/distributed.md)，那么所有的 workers 需要安装 Hadoop 并且配置相应的环境变量。
