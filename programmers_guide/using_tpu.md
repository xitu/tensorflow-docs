# 使用 TPU

这份文档说明了有效使用 [Cloud TPU](https://cloud.google.com/tpu/) 时必需使用的关键 TensorFlow APIs，并强调了常规的 TensorFlow 和在 TPU 上使用区别。

这份文档针对以下用户：

* 熟悉 TensorFlow 的 `Estimator` 和 `Dataset` APIs
* 使用一个已有模型[尝试使用过 Cloud TPU](https://cloud.google.com/tpu/docs/quickstart)
* 浏览过 TPU 模型的样例代码 [[1]](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_tpu.py) [[2]](https://github.com/tensorflow/tpu-demos/tree/master/cloud_tpu/models)
* 对将一个现有的 `Estimator` 模型移植到 Cloud TPU 上运行感兴趣

## TPUEstimator

@{tf.estimator.Estimator$Estimators} 是 TensorFlow 的模型级抽象。标准的 `Estimators` 可以在 CPU 或者 GPU 上驱动模型。 你必须使用 @{tf.contrib.tpu.TPUEstimator} 在 TPU 上驱动模型。

使用 @{$get_started/premade_estimators$pre-made `Estimator`} 和 @{$get_started/custom_estimators$custom `Estimator`s} 的基础介绍可以参考 TensorFlow 的入门指南（Getting Started ）部分，

`TPUEstimator` 类和 `Estimator` 之间多少有些不一样。

要使一个模型可以在 CPU/GPU 或 Cloud TPU 上运行的最简单方法是在 `model_fn` 外定义模型的推理过程（从输入到预测）。然后继续分离 `Estimator` 设置和 `model_fn`，都包含这个推理步骤。这种模式的一个样例是 [tensorflow/models](https://github.com/tensorflow/models/tree/master/official/mnist) 中比较 `mnist.py` 和 `mnist_tpu.py` 的实现。

### 本地运行 `TPUEstimator`

要创建一个标准的 `Estimator` 你可以调用构造函数，并将它传递给 `model_fn`，例如：

```
my_estimator = tf.estimator.Estimator(
  model_fn=my_model_fn)
```
