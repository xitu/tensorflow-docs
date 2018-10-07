# 偏微分方程组

TensorFlow 并不仅限于机器学习使用。为此，我们提供了一个（很平凡的）示例，使用 TensorFlow 来模拟[偏微分方程（PDE）](https://en.wikipedia.org/wiki/Partial_differential_equation)的行为。我们将模拟当雨滴低落在一个池塘的水面上的情况。

## 基本设置

首先导入依赖库：

```python
# 引入用于模拟的库
import tensorflow as tf
import numpy as np

# 引入用于可视化的库
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
```

然后定义一个用于以图像形式展示池塘水面的函数。

```python
def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """将数组展示位一个图片"""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  clear_output(wait = True)
  display(Image(data=f.getvalue()))
```

现在，我们启动一个交互式 TensorFlow 会话。如果我们是使用 `.py` 文件进行这些操作，使用一般的会话也能完成任务。

```python
sess = tf.InteractiveSession()
```

## 便捷的计算函数


```python
def make_kernel(a):
  """将一个 2D 数组转换为一个卷积核"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """一个简化后的 2D 卷积操作"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """计算 2D 数组的拉普拉斯算子"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(x, laplace_k)
```

## 定义 PDE

与自然界中的大多数池塘一样，我们的池塘的尺寸是完美的 500 x 500。

```python
N = 500
```

现在我们将创建我们的池塘，并落入一些雨滴。

```python
# 初始条件：一些雨滴落在池塘水面上

# 所有变量初始化为 0
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

# 雨滴落在池塘水面上的某个随机的点上
for n in range(40):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

DisplayArray(u_init, rng=[-0.1, 0.1])
```

![jpeg](../images/pde_output_1.jpg)


现在让我们定义微分方程的一些细节。


```python
# 参数:
# eps: 时间精度
# damping -- 波阻尼
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# 创建模拟状态的随机变量
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# 离散 PDE 的更新规则
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# 更新状态的操作
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))
```

## 运行模型

使用一个简单的循环来不断地运行它 —— 这就是此程序的乐趣所在。

```python
# 初始化状态为初始条件
tf.global_variables_initializer().run()

# 运行 1000 步
for i in range(1000):
  # 单步模拟
  step.run({eps: 0.03, damping: 0.04})
  DisplayArray(U.eval(), rng=[-0.1, 0.1])
```

![jpeg](../../images/pde_output_2.jpg)

看，波纹！
