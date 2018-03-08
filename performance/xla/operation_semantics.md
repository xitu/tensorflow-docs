# 操作语义

本文档描述了操作的语义，即 [`ComputationBuilder`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h) 接口中所定义的那些操作的语义。通常来说，这些操作与 [`xla_data.proto`](https://www.tensorflow.org/code/tensorflow/compiler/xla/xla_data.proto) 中的 RPC 接口所定义的那些操作是一一对应的。

关于术语：XLA 通常所处理的数据类型为元素类型一致的 N-维数组（比如元素全是 32 比特浮点类型）。在本文档中，**数组** 一词用于表示任意维度的数组。为方便起见，那些特例则使用人们约定俗成的更具体的名称；比如，1维数组称为**向量**，2维数组称为**矩阵**。

## 广播（Broadcast）

另请参阅 [`ComputationBuilder::Broadcast`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

通过在数组中复制数据来增加其维度。

<b> `广播(operand, broadcast_sizes)` </b>

参数         | 类型                    | 语义
----------------- | ----------------------- | -------------------------------
`operand`         | `ComputationDataHandle` | 待复制的数组
`broadcast_sizes` | `ArraySlice<int64>`     | 新维度的形状大小

新的维度被插入在操作数（operand）的左侧，即，若 `broadcast_sizes` 的值为 `{a0, ..., aN}`，而操作数（operand）的维度形状为 `{b0, ..., bM}`，则广播后输出的维度形状为 `{a0, ..., aN, b0, ..., bM}`。

新的维度指标被插入到操作数（operand）副本中，即

```
output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
```

比如，若 `operand` 为一个值为 `2.0f` 的标量，而 `broadcast_sizes` 为 `{2, 3}`，则结果为形状为 `f32[2, 3]` 的一个数组，且它的所有元素的值都为 `2.0f`。

## 调用（Call）

另请参阅 [`ComputationBuilder::Call`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

给定参数情况下，会触发计算。

<b> `Call(computation, args...)` </b>

| 参数     | 类型                     | 语义                        |
| ------------- | ------------------------ | -------------------------------- |
| `computation` | `Computation`            | 类型为 `T_0, T_1, ..., T_N ->S` 的计算，它有 N 个任意类型的参数  |
| `args`        | N 个 `ComputationDataHandle` 的序列            | 任意类型的 N 个 参数 |

参数 `args` 的数目和类型必须与计算 `computation` 相匹配。当然，没有参数 `args` 也是允许的。

## 钳制（Clamp）

另请参阅 [`ComputationBuilder::Clamp`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

将一个操作数钳制到一个区间中，即在一个最小值和一个最大值之间。

<b> `Clamp(computation, args...)` </b>

| 参数     | 类型                    | 语义                        |
| ------------- | ----------------------- | -------------------------------- |
| `computation` | `Computation`           | 类型为 `T_0, T_1,..., T_N -> S` 的计算，它有 N 个任意类型的参数 |
| `operand`     | `ComputationDataHandle` | 类型为 T 的数组 |
| `min`         | `ComputationDataHandle` | 类型为 T 的数组 |
| `max`         | `ComputationDataHandle` | 类型为 T 的数组 |

输入是一个操作数和最大最小值，如果操作数位于最大最小值之间，则返回操作数，如果操作数小于最小值，则返回最小值，如果操作数大于最大值，则返回最大值。即 `clamp(x, a, b) =  max(min(x, a), b)`。

输入的三个数组的维度形状必须是一样的。不过，也可以采用一种最严格的[广播](broadcasting.md)形式，即 `min` 和/或 `max` 可以是类型为 `T` 的一个标量。

`min` 和 `max` 为标量的示例如下：

```
let operand: s32[3] = {-1, 5, 9};
let min: s32 = 0;
let max: s32 = 6;
==>
Clamp(operand, min, max) = s32[3]{0, 5, 6};
```

## 折叠（Collapse）

另请参阅 [`ComputationBuilder::Collapse`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h) 和 @{tf.reshape} 操作。

将一个数组的多个维度折叠为一个维度。

<b> `Collapse(operand, dimensions)` </b>

| 参数    | 类型                    | 语义                           |
| ------------ | ----------------------- | ----------------------------------- |
| `operand`    | `ComputationDataHandle` | 类型为 T 的数组   |
| `dimensions` | `int64` 矢量          | T 的维度形状的依次连续子集 |

折叠操作将操作数的指定的维度子集折叠为一个维度。输入参数为类型 T 的任意数组，和一个编译时为常数的维度指标。维度指标必须是依次排列的，即由低维到高维，且为 T 的维度形状的连续子集。因而，{0, 1, 2}，{0, 1}，或 {1, 2} 都是合规的维度子集，而 {1, 0} 和 {0, 2} 则不是。维度子集所表示的那部分维度会在同样的位置被替换一个新的维度，大小为被替换维度形状大小的乘积。`dimensions` 中的最低维度为折叠这些维度的循环中变化最慢的维度（主序），而最高维度为变化最快的那个维度（次序）。如果想了解更多的一般性的折叠次序问题，请参见 @{tf.reshape} 操作。

比如，令 v 为包含 24 个元素的数组：

```
let v = f32[4x2x3] {{{10, 11, 12},  {15, 16, 17}},
                    {{20, 21, 22},  {25, 26, 27}},
                    {{30, 31, 32},  {35, 36, 37}},
                    {{40, 41, 42},  {45, 46, 47}}};

// 折叠至一个维度，即只留下一个维度
let v012 = Collapse(v, {0,1,2});
then v012 == f32[24] {10, 11, 12, 15, 16, 17,
                      20, 21, 22, 25, 26, 27,
                      30, 31, 32, 35, 36, 37,
                      40, 41, 42, 45, 46, 47};

// 折叠两个较低维度，剩下两个维度
let v01 = Collapse(v, {0,1});
then v01 == f32[4x6] {{10, 11, 12, 15, 16, 17},
                      {20, 21, 22, 25, 26, 27},
                      {30, 31, 32, 35, 36, 37},
                      {40, 41, 42, 45, 46, 47}};

// 折叠两个较高维度，剩下两个维度
let v12 = Collapse(v, {1,2});
then v12 == f32[8x3] {{10, 11, 12},
                      {15, 16, 17},
                      {20, 21, 22},
                      {25, 26, 27},
                      {30, 31, 32},
                      {35, 36, 37},
                      {40, 41, 42},
                      {45, 46, 47}};

```

## 串连（Concatenate）

另请参阅 [`ComputationBuilder::ConcatInDim`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

串连操作是将多个数组操作数合并成一个数组。输出数组与输入数组的秩必须是一样的（即要求输入数组的秩也要相同），并且它按输入次序包含了输入数组的所有元素。

<b> `Concatenate(operands..., dimension)` </b>

| 参数 | 类型 | 语义 |
| ----------- | ----------------------- | ------------------------------------ |
| `operands`  | N 个 `ComputationDataHandle` 的序列 | 类型为 T 维度为 [L0, L1, ...] 的 N 个数组。要求 N>=1 |
| `dimension` | `int64` | 区间 `[0, N)` 中的一个整数值，令那些 `operands` 能够串连起来的维度名 |

除了 `dimension` 之外，其它维度都必须是一样的。这是因为 XLA 不支持 "不规则" 数组。还要注意的是，0-阶的标量值是无法串连在一起的（因为无法确定串连到底发生在哪个维度）。

1-维示例：

```
Concat({{2, 3}, {4, 5}, {6, 7}}, 0)
>>> {2, 3, 4, 5, 6, 7}
```

2-维示例：

```
let a = {
  {1, 2},
  {3, 4},
  {5, 6},
};
let b = {
  {7, 8},
};
Concat({a, b}, 0)
>>> {
  {1, 2},
  {3, 4},
  {5, 6},
  {7, 8},
}
```

图表：
<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ops_concatenate.png">
</div>

## ConvertElementType

另请参阅 [`ComputationBuilder::ConvertElementType`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

类似于 C++ 中逐个元素的 `static_cast` 运算，本操作也从一个数据形状到目标形状执行逐个元素的转换操作；比如，通过一个 `s32`-to-`f32` 的转换程序，`s32` 元素变成了 `f32` 元素。

<b> `ConvertElementType(operand, new_element_type)` </b>

 参数 | 类型 | 语义                                         
------------------ | ----------------------- | ---------------------------
`operand`          | `ComputationDataHandle` | 类型为 T 维度为 D 的数组
`new_element_type` | `PrimitiveType`         | 类型 U

如果操作数（operand）的维度和目标形状不匹配，或者执行一个非法的转换（比如输入或目标为一个元组），则会产生错误。

诸如 `T=s32` 至 `U=f32` 的转换，将执行通常的 int-to-float 的转换过程，比如 round-to-nearest-even。

> 注意：精确的 float-to-int 或反过程目前仍没有指定，但在将来，可能会在转换操作的额外参数中指定。不是所有目标的所有可能的转换都已经实现。

```
let a: s32[3] = {0, 1, 2};
let b: f32[3] = convert(a, f32);
then b == f32[3]{0.0, 1.0, 2.0}
```

## Conv (卷积)

另请参阅 [`ComputationBuilder::Conv`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

类似于 ConvWithGeneralPadding，但是边缘填充（padding）方式比较简单，要么是 SAME 要么是 VALID。SAME 方式将对输入（`lhs`）边缘填充零，使得不考虑步长（striding）的情况下输出与输入的维度形状一致。VALID 填充方式则表示没有填充。

## ConvWithGeneralPadding (卷积)

另请参阅 [`ComputationBuilder::ConvWithGeneralPadding`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

在神经网络中要做这种卷积的计算。在这里，一个卷积可想象为一个 n-维窗口在一个 n-维底空间上移动，而窗口的每个可能的位置都触发一次计算。

| 参数 | 类型 | 语义                                         |
| ---------------- | ----------------------- | ----------------------------- |
| `lhs`            | `ComputationDataHandle` | 秩为 n+2 的输入数组   |
| `rhs`            | `ComputationDataHandle` | 秩为 n+2 的内核权重数组 |
| `window_strides` | `ArraySlice<int64>`     | n-维内核步长数组 |
| `padding`        | `ArraySlice<pair<int64, int64>>` | n-维 (低, 高) 填充数据     |
| `lhs_dilation`   | `ArraySlice<int64>`     | n-维左边扩张因子数组 |
| `rhs_dilation`   | `ArraySlice<int64>`     | n-维右边扩张因子数组 |

令 n 为空间维度的数目。`lhs` 参数是一个 n+2 阶数组，它描述底空间区域的维度。它被为输入，其实 rhs 也是输入。在神经网络中，它们都属于输入激励。n+2 维的含义依次为：

*   `batch`: 此维中每个坐标表示执行卷积的一个独立输入
*   `z/depth/features`: 基空间区域中的每个 (y,x) 位置都指定有一个矢量，由这个维度来表示
*   `spatial_dims`: 描述了定义了底空间区域的那 `n` 个空间维度，窗口要在它上面移动

`rhs` 参数是一个 n+2 阶的数组，它描述了卷积过滤器/内核/窗口。这些维度的含义依次为：

*   `output-z`: 输出的 `z` 维度。
*   `input-z`: 此维度的大小等于 lhs 参数的 `z` 维度的大小。
*   `spatial_dims`: 描述了定义此 n-维窗口的那 `n` 个空间维度，此窗口用于在底空间上移动。

`window_strides` 参数指定了卷积窗口在空间维度上的步长。比如，如果步长为 3，则窗口只用放在第一个空间维度指标为 3 的倍数的那些位置上。

`padding` 参数指定了在底空间区域边缘填充多少个零。填充数目可以是负值 -- 这时数目绝对值表示执行卷积前要移除多少个元素。`padding[0]` 指定维度 `y` 的填充对子，`padding[1]` 指定的是维度 `x` 的填充对子。每个填充对子包含两个值，第一个值指定低位填充数目，第二个值指定高位填充数目。低位填充指的是低指标方向的填充，高位填充则是高指标方向的填充。比如，如果 `padding[1]` 为 `(2,3)`，则在第二个空间维度上，左边填充 2 个零，右边填充 3 个零。填充等价于在执行卷积前在输入 (`lhs`) 中插入这些零值。

`lhs_dilation` 和 `rhs_dilation` 参数指定了扩张系数，分别应用于 lhs 和 rhs 的每个空间维度上。如果在一个空间维度上的扩张系数为 d，则 d-1 个洞将被插入到这个维度的每一项之间，从而增加数组的大小。这些洞被填充上 no-op 值，对于卷积来说表示零值。

rhs 的扩张也被称为深黑卷积（atrous convolution）。更多细节请参考 @{tf.nn.atrous_conv2d}。 lhs 的扩张又被称为反卷积（deconvolution）。

输出形状的维度含义依次为：

*   `batch`: 和输入（`lhs`）具有相同的 `batch` 大小。
*   `z`: 和内核（`rhs`）具有相同的 `output-z` 大小。
*   `spatial_dims`: 卷积窗口的每个合法放置值。

卷积窗口的合法放置是由步长和填充后的底空间区域大小所决定的。

为描述卷积到底干了什么，考虑一个二维卷积，为输出选择某个固定的 `batch`，`z`，`y`，`x` 坐标。则 `(y,x)` 是底空间区域中的某个窗口的一个角的位置（比如左上角，具体是哪个要看你如何编码其空间维度）。现在，我们从底空间区域中得到了一个二维窗口，其中每一个二维点都指定有一个一维矢量，所以，我们得到一个三维盒子。对于卷积过滤器而言，因为我们固定了输出坐标 `z`，我们也有一个三维盒子。这两个盒子具有相同的维度，所以我们可以让它们逐个元素地相乘并相加（类似于点乘）。最后得到输出值。

注意，如果 `output-z` 等于一个数，比如 5，则此窗口的每个位置都在输出的 `z` 维 上产生 5 个值。这些值对应于卷积过滤器的不同部分，即每个 `output-z` 坐标，都由一个独立的三维盒子生成。所以，你可以将其想象成 5 个分立的卷积，每个都用了不同的过滤器。

下面是一个考虑了填充和步长的二维卷积伪代码：

```
for (b, oz, oy, ox) {  // 输出坐标
  value = 0;
  for (iz, ky, kx) {  // 内核坐标和输入 z
    iy = oy*stride_y + ky - pad_low_y;
    ix = ox*stride_x + kx - pad_low_x;
    if (底空间区域内的(iy, ix)是不在填充位置上的) {
      value += input(b, iz, iy, ix) * kernel(oz, iz, ky, kx);
    }
  }
  output(b, oz, oy, ox) = value;
}
```



## CrossReplicaSum

另请参阅 [`ComputationBuilder::CrossReplicaSum`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

跨多个副本（replica）的求和。

<b> `CrossReplicaSum(operand)` </b>

| 参数 | 类型 | 语义                                         |
| ------------ | ----------------------- | ---------------------------------- |
| `operand`    | `ComputationDataHandle` | 跨多个副本待求和的数组。   |

输出的维度形状与输入形状一样。比如，如果有两个副本，而操作数在这两个副本上的值分别为 `(1.0, 2.5)` 和 `(3.0, 5.1)`，则此操作在两个副本上的输出值都是 `(4.0, 7.6)`。

计算 CrossReplicaSum 的结果需要从每个副本中获得一个输入，所以，如果一个副本执行一个 CrossReplicaSum 结点的次数多于其它副本，则前一个副本将永久等待。因此这些副本都运行的是同一个程序，这种情况发生的机会并不多，其中一种可能的情况是，一个 while 循环的条件依赖于输入的数据，而被输入的数据导致此循环在一个副本上执行的次数多于其它副本。

## CustomCall

另请参阅 [`ComputationBuilder::CustomCall`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

在计算中调用由用户提供的函数。

<b> `CustomCall(target_name, args..., shape)` </b>

| 参数 | 类型 | 语义                                         |
| ------------- | ------------------------ | -------------------------------- |
| `target_name` | `string`                 | 函数名称。一个指向这个符号名称的调用指令会被发出 |
| `args`        | N 个 `ComputationDataHandle` 的序列            | 传递给此函数的 N 个任意类型的参数 |
| `shape`       | `Shape`                  | 此函数的输出维度形状  |

不管参数的数目和类型，此函数的签名（signature）都是一样的。

```
extern "C" void target_name(void* out, void** in);
```

比如，如果使用 CustomCall 如下：

```
let x = f32[2] {1,2};
let y = f32[2x3] {{10, 20, 30}, {40, 50, 60}};

CustomCall("myfunc", {x, y}, f32[3x3])
```

`myfunc` 实现的一个示例如下：

```
extern "C" void myfunc(void* out, void** in) {
  float (&x)[2] = *static_cast<float(*)[2]>(in[0]);
  float (&y)[2][3] = *static_cast<float(*)[2][3]>(in[1]);
  EXPECT_EQ(1, x[0]);
  EXPECT_EQ(2, x[1]);
  EXPECT_EQ(10, y[0][0]);
  EXPECT_EQ(20, y[0][1]);
  EXPECT_EQ(30, y[0][2]);
  EXPECT_EQ(40, y[1][0]);
  EXPECT_EQ(50, y[1][1]);
  EXPECT_EQ(60, y[1][2]);
  float (&z)[3][3] = *static_cast<float(*)[3][3]>(out);
  z[0][0] = x[1] + y[1][0];
  // ...
}
```

这个用户提供的函数不能有副作用，而且它的执行结果必须是确定的（即两次同样的调用不能有不同结果）。

> 注：用户函数的黑箱特点限制了编译器的优化潜力。所以，尽量使用原生的 XLA 操作来表示你的计算；只有在迫不得已的情况下才使用 CustomCall。

## 点乘（Dot）

另请参阅 [`ComputationBuilder::Dot`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

<b> `Dot(lhs, rhs)` </b>

 参数 | 类型 | 语义                                     
--------- | ----------------------- | ---------------
`lhs`     | `ComputationDataHandle` | 类型为 T 的数组
`rhs`     | `ComputationDataHandle` | 类型为 T 的数组

此操作的具体语义由它的两个操作数的秩来决定：

| 输入 | 输出 | 语义                                     |
| ----------------------- | --------------------- | ----------------------- |
| 矢量 [n] `dot` 矢量 [n] | 标量 | 矢量点乘 |
| 矩阵 [m x k] `dot` 矢量 [k]   | 矢量 [m]            | 矩阵矢量乘法 |
| 矩阵 [m x k] `dot` 矩阵 [k x n]   | 矩阵 [m x n]        | 矩阵矩阵乘法 |

此操作执行的是 `lhs` 的最后一维与 `rhs` 的倒数第二维之间的乘法结果的求和。因而计算结果会导致维度的 "缩减"。`lhs` 和 `rhs` 缩减的维度必须具有相同的大小。在实际中，我们会用到矢量之间的点乘，矢量/矩阵点乘，以及矩阵间的乘法。

## 逐个元素的二元算术操作

另请参阅 [`ComputationBuilder::Add`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

XLA 支持多个逐个元素的二元算术操作。

<b> `Op(lhs, rhs)` </b>

其中 `Op` 可以是如下操作之一：`Add` (加法), `Sub` (减法), `Mul` (乘法), `Div` (除法), `Rem` (余数), `Max` (最大值), `Min` (最小值), `LogicalAnd` (逻辑且), 或 `LogicalOr` (逻辑或)。

 参数 | 类型 | 语义                                     
--------- | ----------------------- | ----------------------------------------
`lhs`     | `ComputationDataHandle` | 左手边操作数：类型为 T 的数组
`rhs`     | `ComputationDataHandle` | 右手边操作数：类型为 T 的数组

这两个参数的维度形状要么相似，要么兼容。关于维度形状相似或兼容的准确含义，参见文档 @{$broadcasting$broadcasting}。 一个这样的二元操作的结果的维度形状为两个输入数组的广播的结果。虽然可以广播，但不同秩的数组之间的运算是不支持的，除非其中之一是标量。

当 `Op` 为 `Rem` 时，结果的符号与被除数一致，而结果的绝对值总是小于除数的绝对值。

不过，还是可以用如下接口来支持不同秩操作数的广播：

<b> `Op(lhs, rhs, broadcast_dimensions)` </b>

其中 `Op` 的含义同上。这种接口用于具有不同秩的数组之间的算术操作（比如将一个矩阵与一个矢量相加）。

额外的参数 `broadcast_dimensions` 为一个整数指标的切片，用于将低阶操作数的秩扩张至高阶操作数的秩。`broadcast_dimensions` 将低阶形状映射到高阶形状上。扩张后的形状的未被映射的维度将被填充为大小为 1 的退化维度。然后执行退化维度广播，即让维度形状沿这些退化维度扩大，使得与两个操作数的形状相等。更多细节请参阅 @{$broadcasting$广播页面}。

## 逐个元素的比较操作

另请参阅 [`ComputationBuilder::Eq`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

XLA 还支持标准的逐个元素的二元比较操作。注意：当比较浮点类型时，遵循的是标准的 IEEE 754 浮点数语义。

<b> `Op(lhs, rhs)` </b>

其中 `Op` 可以是如下操作之一：`Eq` (相等), `Ne` (不等), `Ge` (大于或等于), `Gt` (大于), `Le` (小于或等于), `Le` (小于)。

 参数 | 类型 | 语义                                     
--------- | ----------------------- | ----------------------------------------
`lhs`     | `ComputationDataHandle` | 左手边操作数：类型为 T 的数组
`rhs`     | `ComputationDataHandle` | 右手边操作数：类型为 T 的数组

这两个参数的维度形状要么相似要么兼容。维度形状的相似或兼容的具体含义参见文档 @{$broadcasting$broadcasting}。操作结果的维度形状为输入数组的形状广播的结果，结果中的元素类型为 `PERD`。在这类操作中，不同秩的数组之间的操作是不支持的，除非其中之一为标量。

要想用广播来比较不同秩的数组，需要用到如下接口：

<b> `Op(lhs, rhs, broadcast_dimensions)` </b>

其中 `Op` 含义同上。这种接口应该用于不同阶的数组之间的比较操作（比如将一个矩阵加到一个矢量上）。

额外的 `broadcast_dimensions` 操作数是一个整数的指标切片，用于指定将操作数广播时的维度。关于其语义的细节内容可参考 @{$broadcasting$广播页面}。

## 逐个元素的一元函数

ComputationBuilder 支持下列逐个元素的一元函数：

<b>`Abs(operand)`</b> 逐个元素的绝对值 `x -> |x|`。

<b>`Ceil(operand)`</b> 逐个元素的整数上界 `x -> ⌈x⌉`。

<b>`Cos(operand)`</b> 逐个元素的余弦 `x -> cos(x)`。

<b>`Exp(operand)`</b> 逐个元素的自然幂指数 `x -> e^x`。

<b>`Floor(operand)`</b> 逐个元素的整数下界 `x -> ⌊x⌋`。

<b>`IsFinite(operand)`</b> 测试 `operand` 的每个元素是否是有限的，即不是正无穷或负无穷，也不是 `NoN`。返回一个 `PRED` 值的数组，维度形状与输入一致，一个元素为 `true` 当且仅当相应的输入元素是有限的。

<b>`Log(operand)`</b> 逐个元素的自然对数 `x -> ln(x)`。

<b>`LogicalNot(operand)`</b> 逐个元素的逻辑非 `x -> !(x)`。

<b>`Neg(operand)`</b> 逐个元素取负值 `x -> -x`。

<b>`Sign(operand)`</b> 逐个元素求符号 `x -> sgn(x)`，其中 

$$\text{sgn}(x) = \begin{cases} -1 & x < 0\\ 0 & x = 0\\ 1 & x > 0 \end{cases}$$

它使用的是 `operand` 的元素类型的比较运算符。

<b>`Tanh(operand)`</b> 逐个元素的双曲正切 `x -> tanh(x)`。


 参数 | 类型 | 语义                                     
--------- | ----------------------- | ---------------------------
`operand` | `ComputationDataHandle` | 函数的操作数

此函数应用于 `operand` 数组的每个元素上，结果是同样形状的一个数组。`operand` 也可以是一个标量，即 0 阶张量。


## BatchNormTraining

关于此算法的细节描述，请参阅 [`ComputationBuilder::BatchNormTraining`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h) 和 [`原始批量标准化论文`](https://arxiv.org/abs/1502.03167)。

<b> 警告：尚没有在 GPU 后端上实现 </b>

在一个批次的多个空间维度上进行标准化。

<b> `BatchNormTraining(operand, scale, offset, epsilon, feature_index)` </b>

| 参数 | 类型 | 语义                                     |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | 待标准化的 n 维数组      |
| `scale`         | `ComputationDataHandle` | 1 维数组 (\\(\gamma\\)) |
| `offset`        | `ComputationDataHandle` | 1 维数组 (\\(\beta\\ )) |
| `epsilon`       | `float`                 | Epsilon 值 (\\(\epsilon\\))   |
| `feature_index` | `int64`                 | 在 `operand` 中的特征维度的索引   |

对于特征维度中的每个特征 (`feature_index` 为 `operand` 中的特征维度的索引)，此操作会计算出关于其它所有维度的数据的均值和方差，然后用这个均值和标准差来对 `operand` 中的每个元素进行标准化。如果传入一个非法的 `feature_index`，则会产生一个错误。

此算法对 `operand` \\(x\\) 中的每个批次做如下运算（假定 `operand` 是一个 4 维数组，它包含 `m` 个元素，`w` 和 `h` 为其空间维度的大小）：

- 对特征维度中的每个特征 `l` 计算批次的均值 \\(\mu_l\\) ：
\\(\mu_l=\frac{1}{mwh}\sum_{i=1}^m\sum_{j=1}^w\sum_{k=1}^h x_{ijkl}\\)

- 计算批次的方差 \\(\sigma^2_l\\)：
\\(\sigma^2_l=\frac{1}{mwh}\sum_{i=1}^m\sum_{j=1}^w\sum_{k=1}^h (x_{ijkl} - \mu_l)^2\\)

- 标准化，缩放和平移：
\\(y_{ijkl}=\frac{\gamma_l(x_{ijkl}-\mu_l)}{\sqrt[2]{\sigma^2_l+\epsilon}}+\beta_l\\)

epsilon 值通常是一个较小的数，加上它可以避免除以零。

输出类型是三个 ComputationDataHandles 的三元组：

| 输出 | 类型 | 语义 |
| ------------ | ----------------------- | -------------------------------------|
| `output`     | `ComputationDataHandle` | n 维数组，维度形状与输入 `operand` (y) 一样  |
| `batch_mean` | `ComputationDataHandle` | 1 维数组 (\\(\mu\\))      |
| `batch_var`  | `ComputationDataHandle` | 1 维数组 (\\(\sigma^2\\)) |

`batch_mean` 和 `batch_var` 是该批次的多个空间维度上用上述公式计算出来的统计矩（moment）。

## BatchNormInference

另请参阅 [`ComputationBuilder::BatchNormInference`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

<b> 警告：尚未实现 </b>

在一个批次的多个空间维度上对一个数组进行标准化。

<b> `BatchNormInference(operand, scale, offset, mean, variance, epsilon, feature_index)` </b>

| 参数 | 类型 | 语义                                     |
| --------------  | ----------------------- | ------------------------------- |
| `operand`       | `ComputationDataHandle` | 待标准化的 n 维数组 to be       |
| `scale`         | `ComputationDataHandle` | 1 维数组             |
| `offset`        | `ComputationDataHandle` | 1 维数组             |
| `mean`          | `ComputationDataHandle` | 1 维数组             |
| `variance`      | `ComputationDataHandle` | 1 维数组             |
| `epsilon`       | `float`                 | Epsilon 值 |
| `feature_index` | `int64`                 | 在 `operand` 中的特征维度的索引    |

对于特征维度中的每个特征（`feature_index` 为 `operand` 中的特征维度的索引），此操作会计算出关于其它所有维度的数据的均值和方差，然后用这个均值和方差对 `operand` 中的每个元素进行标准化。如果一个非法的 `feature_index` 传入了，则会产生一个错误。

`BatchNormInference` 等价于对每个批次不计算均值和方差而调用 `BatchNormTraining`，它使用的是输入的均值 `mean` 和方差 `variance`，而非估计值。这个操作的目的是减少推断时的延迟，因而得名 `BatchNormInference`。

输出是一个 n 维标准化过的数组，维度形状与输入 `operand` 一致。
`operand`.

## BatchNormGrad

另请参阅 [`ComputationBuilder::BatchNormGrad`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

<b> 警告：尚未实现 </b>

计算批次标准化的梯度。

<b> `BatchNormGrad(operand, scale, mean, variance, grad_output, epsilon, feature_index)` </b>

| 参数 | 类型 | 语义                                     |
| --------------  | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | 待标准化的 n 维数组 (x)     |
| `scale`         | `ComputationDataHandle` | 1 维数组 (\\(\gamma\\))       |
| `mean`          | `ComputationDataHandle` | 1 维数组 (\\(\mu\\))  |
| `variance`      | `ComputationDataHandle` | 1 维数组 (\\(\sigma^2\\))          |
| `grad_output`   | `ComputationDataHandle` | 传入到 `BatchNormTraining` 的梯度 (\\( \nabla y\\)) |
| `epsilon`       | `float`                 | Epsilon 值 (\\(\epsilon\\))   |
| `feature_index` | `int64`                 | `operand` 中的特征维度的索引  |

对于特征维度中的每个特征 (`feature_index` 为 `operand` 中的特征维度的索引），此操作计算出其它所有维度的关于 `operand`、`offset` 和 `scale` 的梯度。如果传入一个非法的 `feature_index`，则会产生一个错误。

这三个梯度由下列公式来定义：

\\( \nabla x = \nabla y * \gamma * \sqrt{\sigma^2+\epsilon} \\)

\\( \nabla \gamma = sum(\nabla y * (x - \mu) * \sqrt{\sigma^2 + \epsilon}) \\)

\\( \nabla \beta = sum(\nabla y) \\)

输入的均值和方差表示对于一个批次和多个空间维度的统计矩。

输出类型是 ComputationDataHandle 的三元组：

| 输出 | 类型 | 语义 |
|------------- | ----------------------- | ------------------------------------|
|`grad_operand`| `ComputationDataHandle` | 关于输入 `operand` 的梯度   |
|`grad_offset` | `ComputationDataHandle` | 关于输入 `offset` 的梯度    |
|`grad_scale`  | `ComputationDataHandle` | 关于输入 `scale` 的梯度    |


## GetTupleElement

另请参阅 [`ComputationBuilder::GetTupleElement`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

对编译时常量的元组，该操作能提供索引的功能。

输入值必须是编译时常量值，这样才可以通过形状推理获得结果值的类型。

概念上，这类似于 C++ 中的 `std::get<int N>(t)`：

```
let v: f32[10] = f32[10]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
let s: s32 = 5;
let t: (f32[10], s32) = tuple(v, s);
let element_1: s32 = gettupleelement(t, 1);  // 推断出的形状匹配 s32.
```

另见 @{tf.tuple}。

## Infeed

另请参阅 [`ComputationBuilder::Infeed`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h).

<b> `Infeed(shape)` </b>

| 参数 | 类型 | 语义                                              |
| -------- | ------- | ----------------------------------------------------- |
| `shape`  | `Shape` | 从 Infeed 界面读得的数据的维度形状。此形状的数据布局必须与发送到设备上的数据相匹配；否则此行为是未定义的 |

从设备的隐式 Infeed 流界面读取单个数据项，根据给定的形状和布局来进行解析，并返回一个此数据的一个 `ComputationDataHandle`。在一个计算中允许有多个 Infeed 操作，但这些 Infeed 操作之间必须有全序。比如，下面代码中两个 Infeed 是有全序的，因为在不同 while 循环之间有依赖关系。如果没有全序，编译器会产生一个错误。

```
result1 = while (condition, init = init_value) {
  Infeed(shape)
}

result2 = while (condition, init = result1) {
  Infeed(shape)
}
```

嵌套的元组形状是不支持的。对于一个空的元组形状，Infeed 操作直接是一个 nop，因而不会从设备的 Infeed 中读取任何数据。

> 注意：我们计划支持没有全序的多个 Infeed 操作，在这种情况下，编译器将提供信息，确定这些 Infeed 操作在编译后的程序中如何串行化。


## 映射（Map）

另请参阅 [`ComputationBuilder::Map`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

<b> `Map(operands..., computation)` </b>

| 参数 | 类型 | 语义                      |
| ----------------- | ------------------------ | ----------------------------- |
| `operands`        | N 个 `ComputationDataHandle` 的序列 | 类型为 T_0..T_{N-1} 的 N 个数组 |
| `computation`     | `Computation`            | 类型为`T_0, T_1, ..., T_{N + M -1} -> S` 的计算，有 N 个类型为 T 的参数，和 M 个任意类型的参数 |
| `dimensions`       | `int64` array           | 映射维度的数组  |
| `static_operands` | M 个 `ComputationDataHandle` 的序列  | 任意类型的 M 个数组  |

将一个标量函数作用于给定的 `operands` 数组，可产生相同维度的数组，其中每个元素都是映射函数（mapped function）作用于相应输入数组中相应元素的结果，而 `static_operands` 是 `computation` 的额外输入。

此映射函数可以是任意计算过程，只不过它必须有 N 类类型为 `T` 的标量参数，和类型为 `S` 的输出。输出的维度与输入 `operands` 相同，只不过元素类型 T 换成了 S。

比如，`Map(op1, op2, op3, computation, par1)` 用 `elem_out <-
computation(elem1, elem2, elem3, par1)` 将输入数组中的每个（多维）指标映射产生输出数组。

## 填充（Pad）

另请参阅 [`ComputationBuilder::Pad`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

<b> `Pad(operand, padding_value, padding_config)` </b>

| 参数 | 类型 | 语义                      |
| ---------------- | ----------------------- | ----------------------------- |
| `operand`        | `ComputationDataHandle` | 类型为 `T` 的标量的数组 |
| `padding_value`  | `ComputationDataHandle` | 类型为 `T` 的标量，用于填充 |
| `padding_config` | `PaddingConfig`         | 每个维度的两端的填充量 (low, high) |

通过在数组周围和数组之间进行填充，可以将给定的 `operand` 数组扩大，其中 `padding_value` 和 `padding_config` 用于配置每个维度的边缘填充和内部填充的数目。

`PaddingConfig` 是 `PaddingConfigDimension` 的一个重复字段，它对于每个维度都包含有三个字段：`edge_padding_low`, `edge_padding_high` 和 `interior_padding`。`edge_padding_low` 和 `edge_padding_high` 分别指定了该维度上低端（指标为 0 那端）和高端（最高指标那端）上的填充数目。边缘填充数目可以是负值 -- 负的填充数目的绝对值表示从指定维度移除元素的数目。`interior_padding` 指定了在每个维度的任意两个相邻元素之间的填充数目。逻辑上，内部填充应发生在边缘填充之前，所有在负边缘填充时，会从经过内部填充的操作数之上再移除边缘元素。如果边缘填充配置为 (0, 0)，且内部填充值都是 0，则此操作是一个 no-op。下图展示的是二维数组上不同 `edge_padding` 和 `interior_padding` 值的示例。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ops_pad.png">
</div>


## Reduce

另请参阅 [`ComputationBuilder::Reduce`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

将一个归约函数作用于一个数组。

<b> `Reduce(operand, init_value, computation, dimensions)` </b>

| 参数 | 类型 | 语义                        |
| ------------- | ----------------------- | -------------------------------- |
| `operand`     | `ComputationDataHandle` | 类型为 `T` 的数组            |
| `init_value`  | `ComputationDataHandle` | 类型为 `T` 的标量        |
| `computation` | `Computation`           | 类型为 `T, T -> T`的计算  |
| `dimensions`  | `int64` 数组 | 待归约的未排序的维度数组 |

概念上看，归约（Reduce）操作将输入数组中的一个或多个数组归约为标量。结果数组的秩为 `rank(operand) - len(dimensions)`。 `init_value` 是用于每次归约的初值，如果后端有需求也可以在计算中插入到任何地方。所以，在大多数情况下，`init_value` 应该为归约函数的一个单位元（比如，加法中的 0）。

归约函数的执行顺序是任意的，即可能是非确定的。因而，约化函数不应该对运算的结合性敏感。

有些归约函数，比如加法，对于浮点数并没有严格遵守结合率。不过，如果对数据的值域进行限制，大多数实际情况中，浮点加法已经足够满足结合率。当然，我们也可以构造出完全不遵守结合率的归约函数，这时，XLA 归约就会产生不正确或不可预测的结果。

下面是一个示例，对 1D 数组 [10, 11, 12, 13] 进行归约，归约函数为 `f` （即参数 `computation`），则计算结果为：

`f(10, f(11, f(12, f(init_value, 13)))`

但它还有其它很多种可能性，比如：

`f(init_value, f(f(10, f(init_value, 11)), f(f(init_value, 12), f(13,
init_value))))`

下面是一段实现归约的伪代码，归约计算为求和，初值为 0。

```python
result_shape <- 从 operand_shape 的维度中移除所有待归约的维度

# 遍历 result_shape 中的所有元素，这里，r 的数目等于 result 的秩
for r0 in range(result_shape[0]), r1 in range(result_shape[1]), ...:
  # 初始化 result 的元素
  result[r0, r1...] <- 0

  # 遍历所有的归约维度
  for d0 in range(dimensions[0]), d1 in range(dimensions[1]), ...:
    # 用 operand 的元素的值来增加 result 中的元素的值
    # operand 的元素的索引由所有的 ri 和 di 按正确的顺序构造而来
    # （构造得到的索引用来访问 operand 的整个形状）
    result[r0, r1...] += operand[ri... di]
```

下面是一个对 2D 数组（矩阵）进行归约的示例。其形状的秩为 2，0 维大小为 2，1 维大小为 3：

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/ops_2d_matrix.png">
</div>

对 0 维或 1 维进行求和归约：

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/ops_reduce_from_2d_matrix.png">
</div>

注意，两个归约结果都是一维数组。图中将一个显示为行，另一个显示为列，但这只是为了可视化效果。

下面是一个更复杂的 3D 数组的例子。它的秩为 3 ，形状为 (4,2,3)。为简单起见，我们让 1 到 6 这几个数字沿 0 维复制 4 份。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/ops_reduce_from_3d_matrix.png">
</div>

类似于二维的情况，我们可以只归约一个维度。如果我们归约第 0 维，我们得到一个二阶数组，它沿第 0 维的所有值会合并为一个标量：

```text
|  4   8  12 |
| 16  20  24 |
```

如果我们归约第 2 维，结果仍然是一个二阶数组，沿第 2 维的所有值合并为一个标量：

```text
| 6  15 |
| 6  15 |
| 6  15 |
| 6  15 |
```

注意，输出中剩下的维度的顺序与它们在输入中的相对顺序保持一致，只不过维度的名称（数字）会发生变化，因为数组的秩发生了变化。

我们也可以归约多个维度。对 0 维和 1 维进行求和归约，将得到一个一维数组 `| 20 28 36 |`。

对这个三维数组的所有元素进行求和归约，得到一个标量 `84`。

## ReducePrecision

另请参阅 [`ComputationBuilder::ReducePrecision`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

当浮点数转换为低精度格式（比如 IEEE-FP16）然后转换回原格式时，值可能会发生变化，ReducePrecision 对这种变化进行建模。低精度格式中的指数（exponent）和尾数（mantissa）的比特数目是可以任意指定的，不过不是所有硬件实现都支持所有的比特数设置。

<b> `ReducePrecision(operand, mantissa_bits, exponent_bits)` </b>

| 参数 | 类型 | 语义                    |
| ------------------- | ----------------------- | ---------------------------- |
| `operand`           | `ComputationDataHandle` | 浮点类型 `T` 的数组 |
| `exponent_bits`     | `int32`                 | 低精度格式中的指数比特数目 |
| `mantissa_bits`     | `int32`                 | 低精度格式中的尾数比特数目 |

结果为类型为 `T` 的数组。输入值被舍入至与给定尾数比特的数字最接近的那个值（采用的是"偶数优先"原则）。而超过指数比特所允许的值域时，输入值会被视为正无穷或负无穷。`NaN` 值会保留，不过它可能会被转换为规范化的 NaN 值。

低精度格式必须至少有一个指数比特（为了区分零和无穷，因为两者的尾数比特数都为零），且尾数比特数必须是非负的。指数或尾数的比特数目可能会超过类型 `T`；这种情况下，相应部分的转换就仅仅是一个 no-op 了。


## ReduceWindow

另请参阅 [`ComputationBuilder::ReduceWindow`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

将一个归约函数应用于输入多维数组的每个窗口内的所有元素上，输出一个多维数组，其元素个数等于合法窗口的位置数目。一个池化层可以表示为一个 `ReduceWindow`。

<b> `ReduceWindow(operand, init_value, computation, window_dimensions,
window_strides, padding)` </b>

| 参数 | 类型 | 语义                    |
| ------------------- | ----------------------- | ---------------------------- |
| `operand`           | `ComputationDataHandle` | 类型为 T 的 N 维数组。这是窗口放置的底空间区域  |
| `init_value`        | `ComputationDataHandle` | 归约的初始值。细节请参见 [规约](#reduce)。 |
| `computation`       | `Computation`           | 类型为 `T, T -> T`的归约函数，应用于每个窗口内的所有元素  |
| `window_dimensions` | `ArraySlice<int64>`     | 表示窗口维度值的整数数组  |
| `window_strides`    | `ArraySlice<int64>`     | 表示窗口步长值的整数数组 |
| `padding`           | `Padding`               | 窗口的边缘填充类型（Padding\:\:kSame 或 Padding\:\:kValid） |

下列代码和图为一个使用 `ReduceWindow` 的示例。输入是一个大小为 [4x6] 的矩阵，window_dimensions 和 window_stride_dimensions 都是 [2x3]。

```
// 创建一个归约计算（求最大值）
Computation max;
{
  ComputationBuilder builder(client_, "max");
  auto y = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "y");
  auto x = builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "x");
  builder.Max(y, x);
  max = builder.Build().ConsumeValueOrDie();
}

// 用最大值归约计算来创建一个 ReduceWindow 计算
ComputationBuilder builder(client_, "reduce_window_2x3");
auto shape = ShapeUtil::MakeShape(F32, {4, 6});
auto input = builder.Parameter(0, shape, "input");
builder.ReduceWindow(
    input, *max,
    /*init_val=*/builder.ConstantLiteral(LiteralUtil::MinValue(F32)),
    /*window_dimensions=*/{2, 3},
    /*window_stride_dimensions=*/{2, 3},
    Padding::kValid);
```

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:35%" src="https://www.tensorflow.org/images/ops_reduce_window.png">
</div>

在一个维度上步长为 1 表示在此维度上两个相邻窗口间隔一个元素，为了让窗口互相不重叠，window_stride_dimensions 和 window_dimensions 应该要相等。下图给出了两种不同步长设置的效果。边缘填充应用于输入的每个维度，计算过程实际发生在填充之后的数组上。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:75%" src="https://www.tensorflow.org/images/ops_reduce_window_stride.png">
</div>

归约函数的执行顺序是任意的，因而结果可能是非确定性的。所以，归约函数应该不能对计算的结合性太过敏感。更多细节，参见 [`Reduce`](#reduce) 关于结合性的讨论。

## Reshape

另请参阅 [`ComputationBuilder::Reshape`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h) 和 [`Collapse`](#collapse) 操作。

变形操作（reshape）是将一个数组的维度变成另外一种维度设置。

<b> `Reshape(operand, new_sizes)` </b>
<b> `Reshape(operand, dimensions, new_sizes)` </b>

参数 | 类型 | 语义
------------ | ----------------------- | ---------------------------------------
`operand`    | `ComputationDataHandle` | 类型为 T 的数组
`dimensions` | `int64` vector          | 维度折叠的顺序
`new_sizes`  | `int64` vector          | 新维度大小的矢量

从概念上看，变形操作首先将一个数组拉平为一个一维矢量，然后将此矢量展开为一个新的形状。输入参数是一个类型为 T 的任意数组，一个编译时常量的维度指标数组，以及表示结果维度大小的一个编译时常量的数组。
如果给出了 `dimensions` 参数，这个矢量中的值必须是 T 的所有维度的一个置换，其默认值为 `{0, ..., rank - 1}`。`dimensions` 中的维度的顺序是从最慢变化维（最主序）到最快变化维（最次序），按照这个顺序依次将所有元素折叠到一个维度上。`new_sizes` 矢量决定了输出数组的维度大小。`new_sizes[0]` 表示第 0 维的大小，`new_sizes[1]` 表示的是第 1 维的大小，依此类推。`new_sizes` 中的维度值的乘积必须等于 operand 的维度值的乘积。将折叠的一维数组展开为由 `new_sizes` 定义的多维数组时，`new_sizes` 中的维度的顺序也是最慢变化维（最主序）到最快变化维（最次序）。

比如，令 v 为包含 24 个元素的数组：

```
let v = f32[4x2x3] {{{10, 11, 12}, {15, 16, 17}},
                    {{20, 21, 22}, {25, 26, 27}},
                    {{30, 31, 32}, {35, 36, 37}},
                    {{40, 41, 42}, {45, 46, 47}}};

依次折叠:
let v012_24 = Reshape(v, {0,1,2}, {24});
then v012_24 == f32[24] {10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27,
                         30, 31, 32, 35, 36, 37, 40, 41, 42, 45, 46, 47};

let v012_83 = Reshape(v, {0,1,2}, {8,3});
then v012_83 == f32[8x3] {{10, 11, 12}, {15, 16, 17},
                          {20, 21, 22}, {25, 26, 27},
                          {30, 31, 32}, {35, 36, 37},
                          {40, 41, 42}, {45, 46, 47}};

乱序折叠:
let v021_24 = Reshape(v, {1,2,0}, {24});
then v012_24 == f32[24]  {10, 20, 30, 40, 11, 21, 31, 41, 12, 22, 32, 42,
                          15, 25, 35, 45, 16, 26, 36, 46, 17, 27, 37, 47};

let v021_83 = Reshape(v, {1,2,0}, {8,3});
then v021_83 == f32[8x3] {{10, 20, 30}, {40, 11, 21},
                          {31, 41, 12}, {22, 32, 42},
                          {15, 25, 35}, {45, 16, 26},
                          {36, 46, 17}, {27, 37, 47}};


let v021_262 = Reshape(v, {1,2,0}, {2,6,2});
then v021_262 == f32[2x6x2] {{{10, 20}, {30, 40},
                              {11, 21}, {31, 41},
                              {12, 22}, {32, 42}},
                             {{15, 25}, {35, 45},
                              {16, 26}, {36, 46},
                              {17, 27}, {37, 47}}};
```

作为特例，单元素数组和标量之间可以用变形操作相互转化。比如：

```
Reshape(f32[1x1] {{5}}, {0,1}, {}) == 5;
Reshape(5, {}, {1,1}) == f32[1x1] {{5}};
```

## Rev (倒置)

另请参阅 [`ComputationBuilder::Rev`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

<b>`Rev(operand, dimensions)`</b>

参数 | 类型 | 语义
------------ | ----------------------- | ---------------------
`operand`    | `ComputationDataHandle` | 类型为 T 的数组 
`dimensions` | `ArraySlice<int64>`     | 待倒置的维度

倒置操作将 `operand` 数组沿指定的维度 `dimensions` 将元素的顺序反转，产生一个形状相同的数组。operand 数组的每个元素被存储在输出数组的变换后的位置上。元素的原索引位置在每个待倒置维度上都被反转了，得到其在输出数组中的索引位置（即，如果一个大小为 N 的维度是待倒置的，则索引 i 被变换为 N-i-i）。

`Rev` 操作的一个用途是在神经网络的梯度计算时沿两个窗口维度对卷积权重值进行倒置。

## RngBernoulli

另请参阅 [`ComputationBuilder::RngBernoulli`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

RngBernoulli 构造一个符合 Bernoulli 随机分布的指定形状的随机数组。输入参数是一个 F32 类型的标量，和一个表示输出形状的类型为 `U32` 的数组。

<b>`RngBernoulli(mean, shape)`</b>

| 参数 | 类型 | 语义                             |
| --------- | ----------------------- | ------------------------------------- |
| `mean`    | `ComputationDataHandle` | 类型为 F32 的标量，指定生成的数的均值 |
| `shape`   | `Shape`                 | 类型为 U32 的输出的形状 |

## RngNormal

另请参阅 [`ComputationBuilder::RngNormal`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

RngNormal 构造一个符合 $$(\mu, \sigma)$$ 正态随机分布的指定形状的随机数组。参数 `mu` 和 `sigma` 为 F32 类型的标量值，而输出形状为 U32 的数组。

<b>`RngNormal(mean, sigma, shape)`</b>

| 参数 | 类型 | 语义                              |
| --------- | ----------------------- | -------------------------------------- |
| `mu`      | `ComputationDataHandle` | 类型为 F32 的标量，指定生成的数的均值  |
| `sigma`   | `ComputationDataHandle` | 类型为 F32 的标量，指定生成的数的标准差  |
| `shape`   | `Shape`                 | 类型为 U32 的输出的形状 |

## RngUniform

另请参阅 [`ComputationBuilder::RngUniform`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

RngNormal 构造一个符合区间 $$[a,b)$$ 上的均匀分布的指定形状的随机数组。参数和输出形状可以是 F32、S32 或 U32，但是类型必须是一致的。此外，参数必须是标量值。如果 $$b <= a$$，输出结果与具体的实现有关。

<b>`RngUniform(a, b, shape)`</b>

| 参数 | 类型 | 语义                         |
| --------- | ----------------------- | --------------------------------- |
| `a`       | `ComputationDataHandle` | 类型为 T 的标量，指定区间的下界 |
| `b`       | `ComputationDataHandle` | 类型为 T 的标量，指定区间的上界 |
| `shape`   | `Shape`                 | 类型为 T 的输出的形状 |

## SelectAndScatter

另请参阅 [`ComputationBuilder::SelectAndScatter`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

这个操作可视为一个复合操作，它先在 `operand` 数组上计算 `ReduceWindow`，以从每个窗口中选择一个数，然后将 `source` 数组散布到选定元素的指标位置上，从而构造出一个与 `operand` 数组形状一样的输出数组。二元函数 `select` 用于从每个窗口中选出一个元素，当调用此函数时，第一个参数的指标矢量的字典序小于第二个参数的指标矢量。如果第一个参数被选中，则 `select` 返回 `true`，如果第二个参数被选中，则返回 `false`。而且该函数必须满足传递性，即如果 `select(a, b)` 和 `select(b, c)` 都为 `true`，则 `select(a, c)` 也为 `true`。这样，被选中的元素不依赖于指定窗口中元素访问的顺序。

`scatter` 函数作用在输出数组的每个选中的指标上。它有两个标量参数：

1. 输出数组中选中指标处的值
2. `source` 中被放置到选中指标处的值

它根据这两个参数返回一个标量值，用于更新输出数组中选中指标处的值。最开始的时候，输出数组所有指标处的值都被设为 `init_value`。

输出数组与 `operand` 数组的形状相同，而 `source` 数组必须与 `operand` 上应用 `ReduceWindow` 之后的形状相同。 `SelectAndScatter` 可用于神经网络池化层中梯度值的反向传播。

<b>`SelectAndScatter(operand, select, window_dimensions, window_strides,
padding, source, init_value, scatter)`</b>

| 参数 | 类型 | 语义                    |
| ------------------- | ----------------------- | ---------------------------- |
| `operand`           | `ComputationDataHandle` | 类型为 T 的数组，窗口在它上面滑动 |
| `select`            | `Computation`           | 类型为 `T, T -> PRED` 的二元计算，它被应用到每个窗口中的所有元素上；如果选中第一个元素返回 `true`，如果选中第二个元素返回 `false` |
| `window_dimensions` | `ArraySlice<int64>`     | 表示窗口维度值的整数数组 |
| `window_strides`    | `ArraySlice<int64>`     | 表示窗口步长值的整数数组 |
| `padding`           | `Padding`               | 窗口边缘填充类型（Padding\:\:kSame 或 Padding\:\:kValid）|
| `source`            | `ComputationDataHandle` | 类型为 T 的数组，它的值用于散布 |
| `init_value`        | `ComputationDataHandle` | 类型为 T 的标量值，用于输出数组的初值 |
| `scatter`           | `Computation`           | 类型为 `T, T -> T` 的二元计算，应用于 source 的每个元素和它的目标元素 |

下图为 `SelectAndScatter` 的示例，其中 `select` 函数计算它的参数中的最大值。注意，当窗口重叠时，如图 (2) 所示，`operand` 的一个指标可能会被不同窗口多次选中。在此图中，值为 9 的元素被顶部的两个窗口（蓝色和红色）选中，从而二元加法函数 `scatter` 产生值为 8 的输出值（2+6）。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%"
    src="https://www.tensorflow.org/images/ops_scatter_to_selected_window_element.png">
</div>

`scatter` 函数的执行顺序是任意的，因而可能会出现不确定的结果。所以，`scatter` 函数不应该对计算的结合性过于敏感。更多细节，参见 [`Reduce`](#reduce) 一节中关于结合性的讨论。

## Select

另请参阅 [`ComputationBuilder::Select`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

`Select` 根据一个预测数组的值，由两个输入数组的元素构造出一个输出数组。

<b> `Select(pred, on_true, on_false)` </b>

参数 | 类型 | 语义
---------- | ----------------------- | ------------------
`pred`     | `ComputationDataHandle` | 类型为 PRED 的数组
`on_true`  | `ComputationDataHandle` | 类型为 T 的数组
`on_false` | `ComputationDataHandle` | 类型为 T 的数组

`on_true` 数组和 `on_false` 数组的形状必须一样，这也是输出数组的形状。`pred` 数组的维度必须与 `on_true` 和 `on_false` 相同，元素类型应为 `PRED`。

对于 `pred` 的每个元素 `P`，若其值为 `true`，则输出数组中的相应元素取值于 `on_true`，若其值为 `false`，则取值于 `on_false`。这里还支持一种受限的 [广播]{broadcasting.md) 形式，即 `pred` 可以是类型为 `PRED` 的标量。在这种情况下，如果 `pred` 为 `true`，则输出数组完全取值于 `on_true`，如果 `pred` 为 `false`，则完全取值于 `on_false`。

非标量的 `pred` 示例：

```
let pred: PRED[4] = {true, false, false, true};
let v1: s32[4] = {1, 2, 3, 4};
let v2: s32[4] = {100, 200, 300, 400};
==>
Select(pred, v1, v2) = s32[4]{1, 200, 300, 4};
```

标量 `pred` 示例：

```
let pred: PRED = true;
let v1: s32[4] = {1, 2, 3, 4};
let v2: s32[4] = {100, 200, 300, 400};
==>
Select(pred, v1, v2) = s32[4]{1, 2, 3, 4};
```

XLA 还支持两个元组之间的选择，这时元组被视为标量。如果 `on_true` 和 `on_false` 是元组（必须是相同形状的），则 `pred` 必须是类型为 `PRED` 的标量。


## Slice

另请参阅 [`ComputationBuilder::Slice`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

`Slice` 用于从输入数组中提取出一个子数组。子数组与输入数组的秩相同，它的值在输入数组的包围盒中，此包围盒的维度和指标作为 slice 操作的参数而给出。

<b> `Slice(operand, start_indices, limit_indices)` </b>

| 参数 | 类型 | 语义                        |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | 类型为 T 的 N 维数组 |
| `start_indices` | `ArraySlice<int64>`     | N 个整数的数组，包含每个维度的切片的起始指标。值必须大于等于零 |
| `limit_indices` | `ArraySlice<int64>`     | N 个整数的数组，包含每个维度的切片的结束指标（不包含）。每个维度的结束指标必须严格大于其起始指标，且小于等于维度大小 |

1-维示例：

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
Slice(a, {2}, {4}) produces:
  {2.0, 3.0}
```

2-维示例：

```
let b =
 { {0.0,  1.0,  2.0},
   {3.0,  4.0,  5.0},
   {6.0,  7.0,  8.0},
   {9.0, 10.0, 11.0} }

Slice(b, {2, 1}, {4, 3}) produces:
  { { 7.0,  8.0},
    {10.0, 11.0} }
```

## DynamicSlice

另请参阅 [`ComputationBuilder::DynamicSlice`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

DynamicSlice 在动态的 `start_indices` 处从输出数组中提取出子数组。每个维度切片的大小由 `size_indices` 数组给出，它指定了每个维度切片的区间大小：[start, start + size)。`start_indices` 必须是一维的，维度大小等于 `operand` 的秩。
注意：目前，对于切片指标越界的处理（运行时生成了错误的 `start_indices`）是由具体实现决定的。当前的做法是，让切片指标对输入维度大小取模，从而避免数组的越界访问，但这种行为方式在未来的实现中可能会发生变化。

<b> `DynamicSlice(operand, start_indices, size_indices)` </b>

| 参数 | 类型 | 语义                        |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | 类型为 T 的 N 维数组   |
| `start_indices` | `ComputationDataHandle` | N 个整数构成的 1 阶数组，包含每个维度的切片的起始指标。值必须大于等于零 |
| `size_indices`  | `ArraySlice<int64>`     | N 个整数的列表，包含每个维度的切片大小。每个值必须严格大于零，且 start + size 必须小于等于维度大小，避免发生对维度大小取模的运算 |

1-维示例:

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
let s = {2}

DynamicSlice(a, s, {2}) produces:
  {2.0, 3.0}
```

2-维示例:

```
let b =
 { {0.0,  1.0,  2.0},
   {3.0,  4.0,  5.0},
   {6.0,  7.0,  8.0},
   {9.0, 10.0, 11.0} }
let s = {2, 1}

DynamicSlice(b, s, {2, 2}) produces:
  { { 7.0,  8.0},
    {10.0, 11.0} }
```
## DynamicUpdateSlice

另请参阅 [`ComputationBuilder::DynamicUpdateSlice`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

DynamicUpdateSlice 将输入数组 `operand` 中的一部分值更新，即在 `start_indices` 处用 `update` 覆盖原数据。`update` 的形状决定了结果中被更新的子数组的形状。`start_indices` 的形状必须是一维的，维度大小等于 `operand` 的秩。

注意：目前，对于切片指标越界的处理（运行时生成了错误的 `start_indices`）是由具体实现决定的。当前的做法是，让切片指标对输入维度大小取模，从而避免数组的越界访问，但这种行为方式在未来的实现中可能会发生变化。

<b> `DynamicUpdateSlice(operand, update, start_indices)` </b>

| 参数 | 类型 | 语义                        |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | 类型为 T 的 N 维    |
| `update`        | `ComputationDataHandle` | 类型为 T 的 N 维，包含切片更新。更新形状的每个维度必须严格大于零， start+update 必须小于 operand 每个维度的大小，避免产生越界更新指标  |
| `start_indices` | `ComputationDataHandle` | N 个整数的 1 阶数组，包含每个维度的切片的起始指标。值必须大于或等于零 |

1-维示例:

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
let u = {5.0, 6.0}
let s = {2}

DynamicUpdateSlice(a, u, s) produces:
  {0.0, 1.0, 5.0, 6.0, 4.0}
```

2-维示例:

```
let b =
 { {0.0,  1.0,  2.0},
   {3.0,  4.0,  5.0},
   {6.0,  7.0,  8.0},
   {9.0, 10.0, 11.0} }
let u =
 { {12.0,  13.0},
   {14.0,  15.0},
   {16.0,  17.0} }

let s = {1, 1}

DynamicUpdateSlice(b, u, s) produces:
 { {0.0,  1.0,  2.0},
   {3.0, 12.0, 13.0},
   {6.0, 14.0, 15.0},
   {9.0, 16.0, 17.0} }
```

## Sort

另请参阅 [`ComputationBuilder::Sort`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

`Sort` 用于对输入数组中的元素进行排序。

<b>`Sort(operand)`</b>

参数 | 类型 | 语义
--------- | ----------------------- | -------------------
`operand` | `ComputationDataHandle` | 待排序数组

## Transpose

另请参阅 @{tf.reshape} 操作。

<b>`Transpose(operand)`</b>

参数 | 类型 | 语义
---------     | ----------------------- | -------------------------
`operand`     | `ComputationDataHandle` | 待转置的数组
`permutation` | `ArraySlice<int64>`     | 指定维度重排列的方式


Transpose 将 operand 数组的维度重排列，所以
`∀ i . 0 ≤ i < rank ⇒ input_dimensions[permutation[i]] = output_dimensions[i]`。

这等价于 Reshape(operand, permutation, Permute(permutation, operand.shape.dimensions))。


## Tuple

另请参阅 [`ComputationBuilder::Tuple`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

一个元组（tuple）包含一些数据句柄，它们各自都有自己的形状。

概念上看，它类似于 C++ 中的 `std::tuple`：

```
let v: f32[10] = f32[10]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
let s: s32 = 5;
let t: (f32[10], s32) = tuple(v, s);
```

元组可通过 [`GetTupleElement`](#gettupleelement) 操作来解析（访问）。

## While

另请参阅 [`ComputationBuilder::While`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/computation_builder.h)。

<b> `While(condition, body, init)` </b>

| 参数 | 类型 | 语义                                      |
| ----------- | ------------- | ---------------------------------------------- |
| `condition` | `Computation` | 类型为 `T -> PRED` 的计算，它定义了循环终止的条件 |
| `body`      | `Computation` | 类型为 `T -> T` 的计算，它定义了循环体 |
| `init`      | `T`           | `condition` 和 `body` 的参数的初始值 |

`While` 顺序执行循环体 `body` ，直到 `condition` 失败。这类似于很多语言中的 while 循环，不过，它有如下的区别和限制：

*   一个 `While` 结点有一个类型为 `T` 的返回值，它是最后一次执行 `body` 的结果。
*   类型为 `T` 的形状是由统计确定的，在整个迭代过程中，它都是保持不变的。
*   `While` 结点之间不允许嵌套。这个限制可能会在未来某些目标平台上取消。

该计算的类型为 T 的那些参数使用 `init` 作为迭代的第一次计算的初值，并在接下来的迭代中由 `body` 来更新。

`While` 结点的一个主要使用安例是实现神经网络中的训练的重复执行。下面是一个简化版的伪代码，和一个表示计算过程的图。实际代码可以在 [`while_test.cc`](https://www.tensorflow.org/code/tensorflow/compiler/xla/tests/while_test.cc) 中找到。此例中的 `T` 类型为一个 `Tuple`，它包含一个 `int32` 值，表示迭代次数，还有一个 `vector[10]`，用于累加结果。它有 1000 次迭代，每一次都会将一个常数矢量累加到 result(1) 上。

```
// Pseudocode for the computation.
init = {0, zero_vector[10]} // Tuple of int32 and float[10].
result = init;
while (result(0) < 1000) {
  iteration = result(0) + 1;
  new_vector = result(1) + constant_vector[10];
  result = {iteration, new_vector};
}
```

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ops_while.png">
</div>


