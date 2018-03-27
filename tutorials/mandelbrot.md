# 曼德布洛特集合

对[曼德布洛特集合（Mandelbrot set）](https://en.wikipedia.org/wiki/Mandelbrot_set)进行可视化与机器学习并没有任何关系，但这个有趣的示例可以让您了解如何使用 TensorFlow 解决一般数学问题。本示例实际上是一个很幼稚的可视化实现，但它足以说明要点。（我们可能最终会给出一个更加精确的实现，以得到更加美丽的图像。）


## 基本设置

在开始前，我们需要导入一些库。

```python
# 导入用于模拟的库
import tensorflow as tf
import numpy as np

# 导入用于可视化的库
import PIL.Image
from io import BytesIO
from IPython.display import Image, display
```

现在我们将定义一个函数，用于在迭代指定次数后展示图像。

```python
def DisplayFractal(a, fmt='jpeg'):
  """将一个迭代累计的数组显示为一个分形彩色图片"""
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
                        30+50*np.sin(a_cyclic),
                        155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  display(Image(data=f.getvalue()))
```

## 初始化会话与变量

为了愉快的玩耍，我们会经常使用交互式会话（Interactive Session）。不过一般的会话也能圆满完成任务。

```python
sess = tf.InteractiveSession()
```

我们可以轻松地混合使用 NumPy 与 TensorFlow。

```python
# 使用 numpy 创建一个二维的复数数组
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X+1j*Y
```

现在我们将定义并初始化 TensorFlow 张量。

```python
xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))
```

TensorFlow 要求您在使用变量前，需要明确定义并初始化它们。

```python
tf.global_variables_initializer().run()
```

## 定义及运行运算

现在我们指定一系列的运算

```python
# 计算 z: z^2 + x
zs_ = zs*zs + xs

# 新值是否发散？
not_diverged = tf.abs(zs_) < 4

# 操作并更新 zs 并对迭代进行计数
#
# 注意: 始终在 zs 发散后进行计算，这是一种浪费，有更好的方法
#
step = tf.group(
  zs.assign(zs_),
  ns.assign_add(tf.cast(not_diverged, tf.float32))
)
```

并运行几百步

```python
for i in range(200): step.run()
```

再看看我们得到了什么。

```python
DisplayFractal(ns.eval())
```

![jpeg](../images/mandelbrot_output.jpg)

不错！


