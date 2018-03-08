# 偏微分方程组

TensorFlow 不仅限于机器学习使用。在此，我们提供了一个（很平凡的）实例，使用 TensorFlow 模拟[偏微分方程](https://en.wikipedia.org/wiki/Partial_differential_equation)的表现。我们将模拟一个池塘的水面在雨滴落入时的情形。


## 基本设置

我们需要导入一些库。

```python
#Import libraries for simulation
import tensorflow as tf
import numpy as np

#Imports for visualization
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
```

定义一个用于以图像形式展示池塘水面的函数。

```python
def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  clear_output(wait = True)
  display(Image(data=f.getvalue()))
```

现在，我们启动一个 TensorFlow 交互式会话。如果我们是使用 .py 文件进行这些操作，使用一般的会话也能完成任务。

```python
sess = tf.InteractiveSession()
```

## 便捷的计算函数


```python
def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_k = make_kernel([[0.5, 1.0, 0.5],
                           [1.0, -6., 1.0],
                           [0.5, 1.0, 0.5]])
  return simple_conv(x, laplace_k)
```

## 定义偏微分方程 PDE

与自然界中的大多数池塘一样，我们的池塘的尺寸是完美的 500 x 500。

```python
N = 500
```

现在我们将创建我们的池塘，并落入一些雨滴。

```python
# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

# Some rain drops hit a pond at random points
for n in range(40):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

DisplayArray(u_init, rng=[-0.1, 0.1])
```

![jpeg](https://www.tensorflow.org/images/pde_output_1.jpg)


现在让我们定义微分方程的一些细节。


```python
# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# Discretized PDE update rules
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# Operation to update the state
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))
```

## 运行模型

使用一个简单的循环来不断地运行它 —— 这就是此程序的乐趣所在。

```python
# Initialize state to initial conditions
tf.global_variables_initializer().run()

# Run 1000 steps of PDE
for i in range(1000):
  # Step simulation
  step.run({eps: 0.03, damping: 0.04})
  DisplayArray(U.eval(), rng=[-0.1, 0.1])
```

![jpeg](../images/pde_output_2.jpg)

看！波纹！

