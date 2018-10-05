# 定义以及运行基准

本指南包含定义以及运行一个 TensorFlow 基准的说明。这些基准将输出内容储存在[测试结果](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/test_log.proto)格式中。如果将这些基准测试添加到 TensorFLow 的 github 仓库中，我们将会持续每天构建运行并通过可视化的方式展示在仪表盘上：https://benchmarks-dot-tensorflow-testing.appspot.com/。

[TOC]


## 定义一个基准

定义一个 TensorFlow 基准测试需要继承 `tf.test.Benchmark` 类，并且调用 `self.report_benchmark` 方法。在下面，你可以找到基准代码的示例：

```python
import time

import tensorflow as tf


# Define a class that extends from tf.test.Benchmark.
class SampleBenchmark(tf.test.Benchmark):

  # Note: benchmark method name must start with `benchmark`.
  def benchmarkSum(self):
    with tf.Session() as sess:
      x = tf.constant(10)
      y = tf.constant(5)
      result = tf.add(x, y)

      iters = 100
      start_time = time.time()
      for _ in range(iters):
        sess.run(result)
      total_wall_time = time.time() - start_time

      # Call report_benchmark to report a metric value.
      self.report_benchmark(
          name="sum_wall_time",
          # This value should always be per iteration.
          wall_time=total_wall_time/iters,
          iters=iters)

if __name__ == "__main__":
  tf.test.main()
```
查看 [SampleBenchmark](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/benchmark/) 的完整示例。

在上述例子中需要注意的要点：

* 基准类需要从 `tf.test.Benchmark` 继承。
* 每个基准测试方法都应该以 `benchmark` 为前缀开头。
* 基准测试方法调用 `report_benchmark` 来报告度量的值。

## 使用 Python 来运行

使用 `--benchmarks` 标志来运行 Python 基准测试。将会打印[基准实例](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/util/test_log.proto)。

```
python sample_benchmark.py --benchmarks=SampleBenchmark
```

将该标志位设置为 `--benchmarks=.` 或 `--benchmarks=all` 也是可以的。

（请确保已安装 Tensorflow 并成功导入 `import tensorflow as tf` 行。有关安装说明，请查看 [Installing TensorFlow](../install/)。如果你使用 Bazel 来运行，这个步骤并不是必须的。）

## 添加一个 `bazel` 标志

我们在 TensorFlow github 仓库下，有一个特殊目标叫做 `tf_py_logged_benchmark` 来定义基准测试。`tf_py_logged_benchmark` 是依照常规 `py_test` 目标。 运行 `tf_py_logged_benchmark` 将会打印[测试结果](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/util/test_log.proto)。定义一个 `tf_py_logged_benchmark` 也可以让我们用 TensorFlow 持续构建运行它。

首先，定义一个常规的 `py_test` 目标。请看下面的例子：

```build
py_test(
  name = "sample_benchmark",
  srcs = ["sample_benchmark.py"],
  srcs_version = "PY2AND3",
  deps = [
    "//tensorflow:tensorflow_py",
  ],
)
```

你可以通过传递 `--benchmarks` 标志在 `py_test` 目标中进行基准测试。这个基准测试应该只会打印一个[基准实例](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/util/test_log.proto)原型。

```shell
bazel test :sample_benchmark --test_arg=--benchmarks=all
```


现在，添加 `tf_py_logged_benchmark` 目标（如果可用）。这个目标会将 `--benchmarks = all` 传递给包装后的 `py_test` 目标，并为我们的 TensorFlow 持续构建提供存储输出内容的方法。TensorFlow 存储库中允许提供 `tf_py_logged_benchmark` 目标。 

```build
load("//tensorflow/tools/test:performance.bzl", "tf_py_logged_benchmark")

tf_py_logged_benchmark(
    name = "sample_logged_benchmark",
    target = "//tensorflow/examples/benchmark:sample_benchmark",
)
```

使用以下命令运行基准目标：

```shell
bazel test :sample_logged_benchmark
```
