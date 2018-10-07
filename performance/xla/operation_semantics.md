# 操作语义

本文档介绍了在 [`XlaBuilder`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) 接口中定义的操作语义。通常来说，这些操作与 [`xla_data.proto`](https://www.tensorflow.org/code/tensorflow/compiler/xla/xla_data.proto) 中 RPC 接口所定义的操作是一一对应的。

关于术语：广义数据类型 XLA 处理的是一个 N - 维数组，其元素均为某种数据类型（如 32 位浮点数）。在本文档中，**数组** 表示任意维度的数组。为方便起见，有些特例使用人们约定俗成的更具体和更熟悉的名称；比如，1 维数组称为**向量**，2 维数组称为**矩阵**。

## AllToAll

See also [`XlaBuilder::AllToAll`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h).

Alltoall is a collective operation that sends data from all cores to all cores. It has two phases:

1.  the scatter phase. On each core, the operand is split into `split_count` number of blocks along the `split_dimensions`, and the blocks are scattered to all cores, e.g., the ith block is send to the ith core.
2.  the gather phase. Each core concatenates the received blocks along the `concat_dimension`.

The participating cores can be configured by:

-   `replica_groups`: each ReplicaGroup contains a list of replica id. If empty, all replicas belong to one group in the order of 0 - (n-1). Alltoall will be applied within subgroups in the specified order. For example, replica groups = {{1,2,3},{4,5,0}} means, an Alltoall will be applied within replica 1, 2, 3, and in the gather phase, the received blocks will be concatenated in the order of 1, 2, 3; another Alltoall will be applied within replica 4, 5, 0, and the concatenation order is 4, 5, 0.

Prerequisites:

-   The dimension size of the operand on the split_dimension is divisible by split_count.
-   The operand's shape is not tuple.

<b> `AllToAll(operand, split_dimension, concat_dimension, split_count,
replica_groups)` </b>

| Arguments          | Type                  | Semantics                       |
| ------------------ | --------------------- | ------------------------------- |
| `operand`          | `XlaOp`               | n dimensional input array       |
| `split_dimension`  | `int64`               | A value in the interval `[0,    |
:                    :                       : n)` that names the dimension    :
:                    :                       : along which the operand is      :
:                    :                       : split                           :
| `concat_dimension` | `int64`               | a value in the interval `[0,    |
:                    :                       : n)` that names the dimension    :
:                    :                       : along which the split blocks    :
:                    :                       : are concatenated                :
| `split_count`      | `int64`               | the number of cores that        |
:                    :                       : participate this operation. If  :
:                    :                       : `replica_groups` is empty, this :
:                    :                       : should be the number of         :
:                    :                       : replicas; otherwise, this       :
:                    :                       : should be equal to the number   :
:                    :                       : of replicas in each group.      :
| `replica_groups`   | `ReplicaGroup` vector | each group contains a list of   |
:                    :                       : replica id.                     :

Below shows an example of Alltoall.

```
XlaBuilder b("alltoall");
auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4, 16}), "x");
AllToAll(x, /*split_dimension=*/1, /*concat_dimension=*/0, /*split_count=*/4);
```

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/xla/ops_alltoall.png">
</div>

In this example, there are 4 cores participating the Alltoall. On each core, the operand is split into 4 parts along dimension 0, so each part has shape f32[4,4]. The 4 parts are scattered to all cores. Then each core concatenates the received parts along dimension 1, in the order or core 0-4. So the output on each core has shape f32[16,4].

## BatchNormGrad

算法详情参见 [`XlaBuilder::BatchNormGrad`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) 和 [batch normalization 原始论文](https://arxiv.org/abs/1502.03167)。

计算 batch norm 的梯度

<b> `BatchNormGrad(operand, scale, mean, variance, grad_output, epsilon, feature_index)` </b>

| 类型             | 类型   | 语义                              |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `XlaOp` | 待归一化的 n 维数组 （x）            |
| `scale`         | `XlaOp` | 1 维数组 (\\(\gamma\\))           |
| `mean`          | `XlaOp` | 1 维数组 (\\(\mu\\))              |
| `variance`      | `XlaOp` | 1 维数组 (\\(\sigma^2\\))         |
| `grad_output`   | `XlaOp` | 传入 `BatchNormTraining` 的梯度(\\( \nabla y\\)) |
| `epsilon`       | `float` | ε 值 (\\(\epsilon\\))            |
| `feature_index` | `int64` |`operand` 中的特征维数索引          |

对于特征维数中的每一个特征（`feature_index` 即 `operand` 中特征维度的索引），此操作计算 `operand` 的梯度、在所有其他维度上的 `offset` 和 `scale`。`feature_index` 必须是 `operand` 中特征维度的合法索引。

The three gradients are defined by the following formulas (assuming a 4-dimensional tensor as `operand` and with feature dimension index \\(l\\), batch size `m` and spatial sizes `w` and `h`):

\\[ \begin{split} c_l&=
\frac{1}{mwh}\sum_{i=1}^m\sum_{j=1}^w\sum_{k=1}^h
\left( \nabla y_{ijkl} \frac{x_{ijkl} - \mu_l}{\sigma^2_l+\epsilon} \right)
\\\\
\nabla x_{ijkl} &= \frac{\gamma_{l}}{\sqrt{\sigma^2_{l}+\epsilon}}
\left( \nabla y_{ijkl} - \mathrm{mean}(\nabla y) - c_l (x_{ijkl} - \mu_{l})
\right)
\\\\
\nabla \gamma_l &= \sum_{i=1}^m\sum_{j=1}^w\sum_{k=1}^h \left( \nabla y_{ijkl}
\frac{x_{ijkl} - \mu_l}{\sqrt{\sigma^2_{l}+\epsilon}} \right)
\\\\\
\nabla \beta_l &= \sum_{i=1}^m\sum_{j=1}^w\sum_{k=1}^h \nabla y_{ijkl}
\end{split} \\]

The inputs `mean` and `variance` represent moments value across batch and spatial dimensions.

输出类型是包含三个句柄的元组：

|输出          | 类型    | 语义                                 |
|------------- | ------- | ------------------------------------|
|`grad_operand`| `XlaOp` | 输入 `operand` 的梯度 (\\( \nabla x\\)) |
|`grad_scale`  | `XlaOp` | 输入 `scale` 的梯度 (\\( \nabla \gamma\\)) |
|`grad_offset` | `XlaOp` | 输入 `offset` 的梯度 (\\( \nabla \beta\\)) |

## BatchNormInference

算法详情参见 [`XlaBuilder::BatchNormInference`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) 和 [batch normalization 原始论文](https://arxiv.org/abs/1502.03167)。

在批处理和空间维度上归一化数组。

<b> `BatchNormInference(operand, scale, offset, mean, variance, epsilon, feature_index)` </b>

| 参数             | 类型    | 语义              |
| --------------  | ------- | ----------------- |
| `operand`       | `XlaOp` | 待归一化的 n 维数组 |
| `scale`         | `XlaOp` | 1 维数组           |
| `offset`        | `XlaOp` | 1 维数组           |
| `mean`          | `XlaOp` | 1 维数组           |
| `variance`      | `XlaOp` | 1 维数组           |
| `epsilon`       | `float` | ε 值               |
| `feature_index` | `int64`  | `operand` 中的特征维数索引 |

对于特征维数中的每一个特征（`feature_index` 即 `operand` 中特征维度的索引），此操作计算在所有其他维度上的均值和方差，以及使用均值和方差归一化 `operand` 中的每个元素。`feature_index` 必须是 `operand` 中特征维度的合法索引。

`BatchNormInference` 等价于在每批次中不计算 `mean` 和 `variance` 的情况下调用 `BatchNormTraining`。它使用 `mean` 和 `variance` 作为估计值。此操作的目的是减少推断中的延迟，因此命名为 `BatchNormInference`。

输出是一个 N 维的标准化数组，与输入 `operand` 的形状相同。

## BatchNormTraining

算法详情参见 [`XlaBuilder::BatchNormTraining`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) 和 [`batch normalization 原始论文`](https://arxiv.org/abs/1502.03167)。

在批处理和空间维度上归一化数组。

<b> `BatchNormTraining(operand, scale, offset, epsilon, feature_index)` </b>

| 参数             | 类型    | 语义                             |
| --------------- | ------- | -------------------------------- |
| `operand`       | `XlaOp` | 待归一化的 N 维数组 normalized (x) |
| `scale`         | `XlaOp` | 1 维数组 (\\(\gamma\\))           |
| `offset`        | `XlaOp` | 1 维数组 (\\(\beta\\))            |
| `epsilon`       | `float` | Epsilon 值 (\\(\epsilon\\))       |
| `feature_index` | `int64` | `operand` 中的特征维数索引         |

对于特征维数中的每一个特征（`feature_index` 即 `operand` 中特征维度的索引），此操作计算在所有其他维度上的均值和方差，以及使用均值和方差归一化 `operand` 中的每个元素。`feature_index` 必须是 `operand` 中特征维度的合法索引。

该算法对 `operand` \\(x\\) 中的每批次数据（包含 `w` 和 `h` 的 `m` 元素作为空间维度的大小）按如下次序执行：

- 在特征维度中，对每个特征 `l` 计算批处理均值 \\(\mu_l\\):
\\(\mu_l=\frac{1}{mwh}\sum_{i=1}^m\sum_{j=1}^w\sum_{k=1}^h x_{ijkl}\\)

- 计算批处理方差 \\(\sigma^2_l\\):
\\(\sigma^2_l=\frac{1}{mwh}\sum_{i=1}^m\sum_{j=1}^w\sum_{k=1}^h (x_{ijkl} - \mu_l)^2\\)

- 归一化、缩放和平移:
\\(y_{ijkl}=\frac{\gamma_l(x_{ijkl}-\mu_l)}{\sqrt[2]{\sigma^2_l+\epsilon}}+\beta_l\\)

ε 值，通常是一个很小的数字，以避免 divide-by-zero 错误

输出类型是一个包含三个 `XlaOp` 类型元素的元组：

| 输出          | 类型   | 语义                                  |
| ------------ | ------- | -------------------------------------|
| `output`     | `XlaOp` | 与输入 `operand` (y)具有相同形状的 N 维数组   |
| `batch_mean` | `XlaOp` | 1 维数组 (\\(\mu\\))      |
| `batch_var`  | `XlaOp` | 1 维数组 (\\(\sigma^2\\)) |

输入 `batch_mean ` 和 `batch_var ` 表示使用上述公式在批处理和空间维度上计算的矩值。

## BitcastConvertType

同样参见 [`XlaBuilder::BitcastConvertType`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

类似于 TensorFlow 中的 `tf.bitcast`，对输入数据的每个元素进行 bitcast 操作，从而转化为目标形状。维度必须匹配，且转换是一对一的；如 `s32` 元素通过 bitcast 操作转化为 `f32`。Bitcast 采用底层 cast 操作，所以不同浮点数表示法的机器会产生不同的结果。

<b> `BitcastConvertType(operand, new_element_type)` </b>

参数                | 类型   | 语义
------------------ | ------- | --------------------
`operand`          | `XlaOp` | D 维，类型为 T 的数组
`new_element_type` | `PrimitiveType`   | 类型 U

operand 和 目标形状的维度必须匹配。源和目标元素类型的位宽必须一致。源和目标元素类型不能是元组。

## 广播（Broadcast）

另请参阅 [`XlaBuilder::Broadcast`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

通过在数组中复制数据来增加其维度。

<b> `Broadcast(operand, broadcast_sizes)` </b>

参数               | 类型                    | 语义
----------------- | ----------------------- | -------------------------------
`operand`         | `XlaOp`                 | 待复制的数组
`broadcast_sizes` | `ArraySlice<int64>`     | 新维度的形状大小

新的维度被插入在操作数（operand）的左侧，即，若 `broadcast_sizes` 的值为 `{a0, ..., aN}`，而操作数（operand）的维度形状为 `{b0, ..., bM}`，则广播后输出的维度形状为 `{a0, ..., aN, b0, ..., bM}`。

新的维度指标被插入到操作数（operand）副本中，即

```
output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
```

比如，若 `operand` 为一个值为 `2.0f` 的标量，且 `broadcast_sizes` 为 `{2, 3}`，则结果形状为 `f32[2, 3]` 的一个数组，且它的所有元素的值都为 `2.0f`。

## 调用（Call）

另请参阅 [`XlaBuilder::Call`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

给定参数情况下，触发计算。

<b> `Call(computation, args...)` </b>

| 参数           | 类型                     | 语义                              |
| ------------- | ------------------------ | -------------------------------- |
| `computation` | `XlaComputation`         | 类型为 `T_0, T_1, ..., T_N ->S` 的计算，它有 N 个任意类型的参数  |
| `args`        | N 个 `XlaOp` 的序列       | 任意类型的 N 个 参数 |

参数 `args` 的数目和类型必须与计算 `computation` 相匹配。当然，没有参数 `args` 也是允许的。

## 钳制（Clamp）

另请参阅 [`XlaBuilder::Clamp`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

将一个操作数钳制在最小值和最大值之间的范围内。

<b> `Clamp(min, operand, max)` </b>

| 参数           | 类型                    | 语义                              |
| ------------- | ----------------------- | -------------------------------- |
| `min`         | `XlaOp` | 类型为 T 的数组 |
| `operand`     | `XlaOp` | 类型为 T 的数组 |
| `max`         | `XlaOp` | 类型为 T 的数组 |

给定操作数，最小和最大值，如果操作数位于最小值和最大值之间，则返回操作数，否则，如果操作数小于最小值，则返回最小值，如果操作数大于最大值，则返回最大值。即 `clamp(a, x, b) =  min(max(a, x), b)`。

输入的三个数组的维度形状必须是一样的。另外，也可以采用一种严格的[广播](./broadcasting.md)形式，即 `min` 和/或 `max` 可以是类型为 `T` 的一个标量。

`min` 和 `max` 为标量的示例如下：

```
let operand: s32[3] = {-1, 5, 9};
let min: s32 = 0;
let max: s32 = 6;
==>
Clamp(min, operand, max) = s32[3]{0, 5, 6};
```

## 折叠（Collapse）

另请参阅 [`XlaBuilder::Collapse`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) 和 `tf.reshape` 操作。

将一个数组的多个维度折叠为一个维度。

<b> `Collapse(operand, dimensions)` </b>

| 参数         | 类型          | 语义                    |
| ------------ | ---   ------ | ----------------------- |
| `operand`    | `XlaOp`      | 类型为 T 的数组          |
| `dimensions` | `int64` 矢量 | T 的维度形状的依次连续子集 |

折叠操作将操作数的指定的维度子集折叠为一个维度。输入参数为类型 T 的任意数组，和一个编译时为常数的维度指标。维度指标必须是依次排列的，即由低维到高维，且为 T 的维度形状的连续子集。因而，{0, 1, 2}，{0, 1}，或 {1, 2} 都是合规的维度子集，而 {1, 0} 和 {0, 2} 则不是。维度子集所表示的那部分维度会在同样的位置被替换一个新的维度，大小为被替换维度形状大小的乘积。`dimensions` 中的最低维度为折叠这些维度的循环中变化最慢的维度（主序），而最高维度为变化最快的那个维度（次序）。如果想了解更多的一般性的折叠次序问题，请参见 `tf.reshape` 操作。

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

另请参阅 [`XlaBuilder::ConcatInDim`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

串连操作是将多个数组操作数合并成一个数组。输出数组与输入数组的秩必须是一样的（即要求输入数组的秩也要相同），并且它按输入次序包含了输入数组的所有元素。

<b> `Concatenate(operands..., dimension)` </b>

| 参数 | 类型 | 语义 |
| ----------- | ----------------------- | ------------------------------------ |
| `operands`  | N 个 `XlaOp` 的序列 | 类型为 T 维度为 [L0, L1, ...] 的 N 个数组。要求 N>=1 |
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

## Conditional

另请参阅 [`XlaBuilder::Conditional`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b> `Conditional(pred, true_operand, true_computation, false_operand, false_computation)` </b>

| 参数                 | 类型                    | 语义                         |
| ------------------- | ----------------------- | --------------------------- |
| `pred`              | `XlaOp`                 | 类型为 `PRED` 的标量          |
| `true_operand`      | `XlaOp`                 | 类型为 `T_0` 的参数           |
| `true_computation`  | `XlaComputation`        | 类型为 `T_0 -> S` 的计算      |
| `false_operand`     | `XlaOp`                 | 类型为 `T_1` 的参数           |
| `false_computation` | `XlaComputation`        | 类型为 `T_0 -> S` 的计算      |

如果 `pred` 为 `true`，执行 `true_computation`，如果 `pred` 为 `false`，则返回结果。

`true_computation` 必须接受一个类型为 `T_0` 的单参数，并使用 `true_operand` 来调用，它们必须类型相同。  `false_computation` 必须接受一个类型为 `T_1` 的单参数，并使用 `false_operand` 来调用，它们必须类型相同。 `true_computation` 和 `false_computation` 的返回值的类型必须相同。

注意，根据 `pred` 的值，`true_computation` 和 `false_computation` 只能执行其中一个。

## Conv (卷积)

另请参阅 [`XlaBuilder::Conv`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

类似于 ConvWithGeneralPadding，但是边缘填充（padding）方式比较简单，要么是 SAME 要么是 VALID。SAME 方式将对输入（`lhs`）边缘填充零，使得在不考虑步长（striding）的情况下输出与输入的维度形状一致。VALID 填充方式则表示没有填充。

## ConvWithGeneralPadding (卷积)

另请参阅 [`XlaBuilder::ConvWithGeneralPadding`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

计算神经网络中使用的卷积。此处，一个卷积可被认为是一个 n-维窗口在一个 n-维底空间上移动，并对窗口的每个可能的位置执行一次计算。

| 参数 | 类型 | 语义                                         |
| ---------------- | ----------------------- | ----------------------------- |
| `lhs`            | `XlaOp` | 秩为 n+2 的输入数组   |
| `rhs`            | `XlaOp` | 秩为 n+2 的内核权重数组 |
| `window_strides` | `ArraySlice<int64>`     | n-维内核步长数组 |
| `padding`        | `ArraySlice<pair<int64, int64>>` | n-维 (低, 高) 填充数据     |
| `lhs_dilation`   | `ArraySlice<int64>`     | n-维左边扩张因子数组 |
| `rhs_dilation`   | `ArraySlice<int64>`     | n-维右边扩张因子数组 |
| `feature_group_count` | int64               | 特征组的数量  |

设 n 为空间维数。`lhs` 参数是一个 n+2 阶数组，它描述底空间区域的维度。它被称为输入，其实 rhs 也是输入。在神经网络中，它们都属于输入激励。n+2 维的含义依次为：

*   `batch`：此维中每个坐标表示执行卷积的一个独立输入
*   `z/depth/features`：基空间区域中的每个 (y,x) 位置都指定有一个矢量，由这个维度来表示
*   `spatial_dims`：描述了定义了底空间区域的那 `n` 个空间维度，窗口要在它上面移动

`rhs` 参数是一个 n+2 阶的数组，它描述了卷积过滤器/内核/窗口。这些维度的含义依次为：

*   `output-z`：输出的 `z` 维度。
*   `input-z`：此维度的大小乘以 `feature_group_count` 应该等于 lhs 参数的 `z` 维度的大小。
*   `spatial_dims`：描述了定义此 n-维窗口的那 `n` 个空间维度，此窗口用于在底空间上移动。

`window_strides` 参数指定了卷积窗口在空间维度上的步长。比如，如果步长为 3，则窗口只用放在第一个空间维度指标为 3 的倍数的那些位置上。

`padding` 参数指定了在底空间区域边缘填充多少个零。填充数目可以是负值 -- 这时数目绝对值表示执行卷积前要移除多少个元素。`padding[0]` 指定维度 `y` 的填充对子，`padding[1]` 指定的是维度 `x` 的填充对子。每个填充对子包含两个值，第一个值指定低位填充数目，第二个值指定高位填充数目。低位填充指的是低指标方向的填充，高位填充则是高指标方向的填充。比如，如果 `padding[1]` 为 `(2,3)`，则在第二个空间维度上，左边填充 2 个零，右边填充 3 个零。填充等价于在执行卷积前在输入 (`lhs`) 中插入这些零值。

`lhs_dilation` 和 `rhs_dilation` 参数指定了扩张系数，分别应用于 lhs 和 rhs 的每个空间维度上。如果在一个空间维度上的扩张系数为 d，则 d-1 个洞将被插入到这个维度的每一项之间，从而增加数组的大小。这些洞被填充上 no-op 值，对于卷积来说表示零值。

rhs 的扩张也称为无功卷积。有关更多细节，请参见 `tf.nn.atrous_conv2d`。lhs 的扩张也称为转置卷积。要了解更多细节，请参见`tf.nn.conv2d_transpose`。

The `feature_group_count` argument (default value 1) can be used for grouped convolutions. `feature_group_count` needs to be a divisor of both the input and the output feature dimension. If `feature_group_count` is greater than 1, it means that conceptually the input and output feature dimension and the `rhs` output feature dimension are split evenly into `feature_group_count` many groups, each group consisting of a consecutive subsequence of features. The input feature dimension of `rhs` needs to be equal to the `lhs` input feature dimension divided by `feature_group_count` (so it already has the size of a group of input features). The i-th groups are used together to compute `feature_group_count` many separate convolutions. The results of these convolutions are concatenated together in the output feature dimension.

For depthwise convolution the `feature_group_count` argument would be set to the input feature dimension, and the filter would be reshaped from `[filter_height, filter_width, in_channels, channel_multiplier]` to `[filter_height, filter_width, 1, in_channels * channel_multiplier]`. For more details, see `tf.nn.depthwise_conv2d`.

输出形状的维度含义依次为：

*   `batch`：和输入（`lhs`）具有相同的 `batch` 大小。
*   `z`：和内核（`rhs`）具有相同的 `output-z` 大小。
*   `spatial_dims`：每个卷积窗口的有效位置的值。

卷积窗口的有效位置是由步长和填充后的底空间区域大小所决定的。

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

## ConvertElementType

另请参阅 [`XlaBuilder::ConvertElementType`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

与 C++ 中逐元素的 `static_cast` 类似，对输入数据的每个元素进行转换操作，从而转化为目标形状。维度必须匹配，且转换是一对一的；如 `s32` 元素通过 `s32`-to-`f32` 转换过程转换为 `f32`。

<b> `ConvertElementType(operand, new_element_type)` </b>

参数          | 类型                 | 语义
------------------ | --------------- | ------------------
`operand`          | `XlaOp`         | D 维类型为 T 的数组
`new_element_type` | `PrimitiveType` | 类型 U

操作数和目标形状的维度必须匹配。源和目标元素类型不能是元组。

一个 `T=s32` 到 `U=f32` 的转换将执行标准化的 int-to-float 转化过程，如 round-to-nearest-even。

> 注意：目前没有指定精确的 float-to-int 和 visa-versa 转换，但是将来可能作为转换操作的附加参数。不是所有的目标都实现了所有可能的转换。

```
let a: s32[3] = {0, 1, 2};
let b: f32[3] = convert(a, f32);
then b == f32[3]{0.0, 1.0, 2.0}
```

## CrossReplicaSum

另请参阅 [`XlaBuilder::CrossReplicaSum`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

跨多个副本（replica）的求和。

<b> `CrossReplicaSum(operand)` </b>

| 参数 | 类型 | 语义                      |
| ------------ | ------- | ---------------- |
| `operand`    | `XlaOp` | 跨多个副本待求和的数组。  |
| `replica_group_ids`    | `int64` 向量 | 每个副本的 Group ID |

输出的维度形状与输入形状一样。比如，如果有两个副本，而操作数在这两个副本上的值分别为 `(1.0, 2.5)` 和 `(3.0, 5.25)`，则此操作在两个副本上的输出值都是 `(4.0, 7.75)`。

`replica_group_ids` identifies the group ID of each replica. The group ID must either be empty (all replicas belong to a single group), or contain the same number of elements as the number of replicas. For example, if `replica_group_ids` = {0, 1, 2, 3, 0, 1, 2, 3} has eight replicas, there are four subgroups of replica IDs: {0, 4}, {1, 5}, {2, 6}, and {3, 7}. The size of each subgroup *must* be identical, so, for example, using: `replica_group_ids` = {0, 1, 2, 0} for four replicas is invalid.

计算 CrossReplicaSum 的结果需要从每个副本中获得一个输入，所以，如果一个副本执行一个 CrossReplicaSum 结点的次数多于其它副本，则前一个副本将永久等待。因此这些副本都运行的是同一个程序，这种情况发生的机会并不多，其中一种可能的情况是，一个 while 循环的条件依赖于输入的数据，而被输入的数据导致此循环在一个副本上执行的次数多于其它副本。

## CustomCall

另请参阅 [`XlaBuilder::CustomCall`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

在计算中调用由用户提供的函数。

<b> `CustomCall(target_name, args..., shape)` </b>

| 参数 | 类型 | 语义                                         |
| ------------- | ------------------ | -------------------------------- |
| `target_name` | `string`           | 函数名称。一个指向这个符号名称的调用指令会被发出 |
| `args`        | N 个 `XlaOp` 的序列 | 传递给此函数的 N 个任意类型的参数 |
| `shape`       | `Shape`            | 此函数的输出维度形状  |

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

另请参阅 [`XlaBuilder::Dot`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b> `Dot(lhs, rhs)` </b>

 参数 | 类型 | 语义                                     
--------- | ------- | ---------------
`lhs`     | `XlaOp` | 类型为 T 的数组
`rhs`     | `XlaOp` | 类型为 T 的数组

此操作的具体语义由它的两个操作数的秩来决定：

| 输入 | 输出 | 语义                                     |
| ----------------------- | --------------------- | ----------------------- |
| 矢量 [n] `dot` 矢量 [n] | 标量 | 矢量点乘 |
| 矩阵 [m x k] `dot` 矢量 [k]   | 矢量 [m]            | 矩阵矢量乘法 |
| 矩阵 [m x k] `dot` 矩阵 [k x n]   | 矩阵 [m x n]        | 矩阵矩阵乘法 |

此操作执行的是 `lhs` 的最后一维与 `rhs` 的倒数第二维之间的乘法结果的求和。因而计算结果会导致维度的 "缩减"。`lhs` 和 `rhs` 缩减的维度必须具有相同的大小。在实际中，我们会用到矢量之间的点乘，矢量/矩阵点乘，以及矩阵间的乘法。

## DotGeneral

另请参阅 [`XlaBuilder::DotGeneral`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b> `DotGeneral(lhs, rhs, dimension_numbers)` </b>

| 参数 | 类型                    | 语义
| --------- | ----------------------- | ---------------
| `lhs`     | `XlaOp` | 类型为 T 的数组
| `rhs`     | `XlaOp` | 类型为 T 的数组
| `dimension_numbers` | `DotDimensionNumbers` | 类型为 T 的数组

和点乘一样，但是对于 'lhs' 和 'rhs' 允许收缩和指定批处理维数。

| DotDimensionNumbers 成员 | 类型                    | 语义
| --------- | ----------------------- | ---------------
| 'lhs_contracting_dimensions' | repeated int64 | 'lhs' 转换维数 |
| 'rhs_contracting_dimensions' | repeated int64 | 'rhs' 转换维数 |
| 'lhs_batch_dimensions' | repeated int64 | 'lhs' 批处理维数     |
| 'rhs_batch_dimensions' | repeated int64 | 'rhs' 批处理维数     |

DotGeneral 根据 'dimension_numbers' 指定的维数进行转换操作，然后计算点积和。 

与 'lhs' 和 'rhs' 有关的转换维数不需要相同，但是在 'lhs_contracting_dimensions' 和 'rhs_contracting_dimensions' 数组必须按照相同的顺序列出，同时具有相同的维数大小。且需要同时与 'lhs' 和 'rhs' 在同一个维度上。

以转换维数为例：

```
lhs = { {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0} }

rhs = { {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0} }

DotDimensionNumbers dnums;
dnums.add_lhs_contracting_dimensions(1);
dnums.add_rhs_contracting_dimensions(1);

DotGeneral(lhs, rhs, dnums) -> { {6.0, 12.0},
                                 {15.0, 30.0} }
```

'lhs' 和 'rhs' 的批处理维数必须相同，在两个数组中必须以相同的顺序列出，同时维数大小必须相同。必须在合约和非合约/非批次维度之前订阅。

批处理维数的例子（批处理大小为 2，2x2 矩阵）：

```
lhs = { { {1.0, 2.0},
          {3.0, 4.0} },
        { {5.0, 6.0},
          {7.0, 8.0} } }

rhs = { { {1.0, 0.0},
          {0.0, 1.0} },
        { {1.0, 0.0},
          {0.0, 1.0} } }

DotDimensionNumbers dnums;
dnums.add_lhs_contracting_dimensions(2);
dnums.add_rhs_contracting_dimensions(1);
dnums.add_lhs_batch_dimensions(0);
dnums.add_rhs_batch_dimensions(0);

DotGeneral(lhs, rhs, dnums) -> { { {1.0, 2.0},
                                   {3.0, 4.0} },
                                 { {5.0, 6.0},
                                   {7.0, 8.0} } }
```

| Input                               | Output            | Semantics        |
| ----------------------------------- | ----------------- | ---------------- |
| [b0, m, k] `dot` [b0, k, n]         | [b0, m, n]        |  batch matmul    |
| [b0, b1, m, k] `dot` [b0, b1, k, n] | [b0, b1, m, n]    |  batch matmul    |

由此得出的结果维数是从批处理维度开始，然后是 `lhs` 非收缩/非批处理维数，最后是 `rhs` 非收缩/非批处理维数。

## DynamicSlice

另请参阅 [`XlaBuilder::DynamicSlice`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

DynamicSlice 从动态 `start_indices` 输入数组中提取子数组。`size_indices` 为每个维度的切片大小，它在每个维度上指定了切片范围：[start, start + size)。`start_indices` 的秩必须为 1，且维数大小等于 `operand` 的秩。

<b> `DynamicSlice(operand, start_indices, size_indices)` </b>

| 参数       | 类型                    | 语义                        |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `XlaOp`                 | 类型为 T 的 N 维数组    |
| `start_indices` | `XlaOp`                 | N 个整数组成的秩为 1 的数组，其中包含每个维度的起始切片索引。值必须大于等于0      |
| `size_indices`  | `ArraySlice<int64>`     | N 个整数组成的列表，其中包含每个维度的切片大小。值必须大于 0，且 start + size 必须小于等于维度大小，从而避免封装维数大小的模运算    |

The effective slice indices are computed by applying the following transformation for each index `i` in `[1, N)` before performing the slice:

```
start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i] - size_indices[i])
```

This ensures that the extracted slice is always in-bounds with respect to the operand array. If the slice is in-bounds before the transformation is applied, the transformation has no effect.

1 维示例如下：

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
let s = {2}

DynamicSlice(a, s, {2}) produces:
  {2.0, 3.0}
```

2 维示例如下：

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

另请参见 [`XlaBuilder::DynamicUpdateSlice`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

DynamicUpdateSlice 是在输入数组 `operand` 上，通过切片 `update` 操作覆盖 `start_indices` 后生成的结果。`update` 的形状决定了更新后结果的子数组的形状。 `start_indices` 的秩必须为 1，且维数大小等于 `operand` 的秩。

注意：当前实现未定义切片索引越界（错误的运行时生成的'start_indices'）的情况。

<b> `DynamicUpdateSlice(operand, update, start_indices)` </b>

| 参数       | 类型         | 语义                        |
| --------------- | ------- | -------------------------------- |
| `operand`       | `XlaOp` | 类型为 T 的 N 维数组    |
| `update`        | `XlaOp` | 类型为 T 的包含切片更新的 N 维数组，每个维度的更新形状必须大于 0 ，且 start + update 必须小于维度大小，从而避免越界更新索引    |
| `start_indices` | `XlaOp` | N 个整数组成的秩为 1 的数组，其中包含每个维度的起始切片索引。值必须大于等于0       |

The effective slice indices are computed by applying the following transformation for each index `i` in `[1, N)` before performing the slice:

```
start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i] - update.dimension_size[i])
```

This ensures that the updated slice is always in-bounds with respect to the operand array. If the slice is in-bounds before the transformation is applied, the transformation has no effect.

1 维示例如下：

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
let u = {5.0, 6.0}
let s = {2}

DynamicUpdateSlice(a, u, s) produces:
  {0.0, 1.0, 5.0, 6.0, 4.0}
```

2 维示例如下：

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

## 逐个元素的二元算术操作

另请参阅 [`XlaBuilder::Add`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

XLA 支持多个逐个元素的二元算术操作。

<b> `Op(lhs, rhs)` </b>

其中 `Op` 可以是如下操作之一：`Add` (加法), `Sub` (减法), `Mul` (乘法), `Div` (除法), `Rem` (余数), `Max` (最大值), `Min` (最小值), `LogicalAnd` (逻辑且), 或 `LogicalOr` (逻辑或)。

 参数 | 类型 | 语义                                     
--------- | ----------------------- | ----------------------------------------
`lhs`     | `XlaOp` | 左操作数：类型为 T 的数组
`rhs`     | `XlaOp` | 右操作数：类型为 T 的数组

这两个参数的维度形状要么相似，要么兼容。关于维度形状相似或兼容的准确含义，参见[广播](../../performance/xla/broadcasting.md)文档。二元操作的结果有一个形状，它是广播两个输入数组的结果。虽然可以广播，但不同秩的数组之间的运算是不支持的，除非其中之一是标量。

当 `Op` 为 `Rem` 时，结果的符号与被除数一致，而结果的绝对值总是小于除数的绝对值。

Integer division overflow (signed/unsigned division/remainder by zero or signed divison/remainder of `INT_SMIN` with `-1`) produces an implementation defined value.

不过，还是可以用如下接口来支持不同秩操作数的广播：

<b> `Op(lhs, rhs, broadcast_dimensions)` </b>

其中 `Op` 的含义同上。这种接口用于具有不同秩的数组之间的算术操作（比如将一个矩阵与一个矢量相加）。

附加参数 `broadcast_dimensions` 为一个整数切片，用于将低阶操作数的秩扩张至高阶操作数的秩。`broadcast_dimensions` 将低阶形状映射到高阶形状上。扩张后的形状的未被映射的维度将被填充为大小为 1 的退化维度。然后执行退化维度广播，即让维度形状沿这些退化维度扩大，使得与两个操作数的形状相等。更多细节请参阅[广播页面](../../performance/xla/broadcasting.md)。

## 逐个元素的比较操作

另请参阅 [`XlaBuilder::Eq`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

XLA 还支持标准的逐个元素的二元比较操作。注意：当比较浮点类型时，遵循的是标准的 IEEE 754 浮点数语义。

<b> `Op(lhs, rhs)` </b>

其中 `Op` 可以是如下操作之一：`Eq` (相等), `Ne` (不等), `Ge` (大于或等于), `Gt` (大于), `Le` (小于或等于), `Lt` (小于)。

 参数 | 类型 | 语义                                     
--------- | ------- | -----------------------
`lhs`     | `XlaOp` | 左操作数：类型为 T 的数组
`rhs`     | `XlaOp` | 右操作数：类型为 T 的数组

这两个参数的维度形状要么相似要么兼容。维度形状的相似或兼容的具体含义参见[广播](../../performance/xla/broadcasting.md)文档。二元操作的结果有一个形状，它是广播两个输入数组的结果。其中元素类型为 `PERD`。在这类操作中，不同秩的数组之间的操作是不支持的，除非其中之一为标量。

要想用广播来比较不同秩的数组，需要用到如下接口：

<b> `Op(lhs, rhs, broadcast_dimensions)` </b>

其中 `Op` 含义同上。这种接口应该用于不同阶的数组之间的比较操作（比如将一个矩阵加到一个矢量上）。

附加参数 `broadcast_dimensions` 为一个整数切片，用于指定将操作数广播时的维度。关于其语义的细节内容可参考[广播页面](../../performance/xla/broadcasting.md)。

## 逐个元素的一元函数

XlaBuilder 支持下列逐个元素的一元函数：

<b>`Abs(operand)`</b> 逐个元素的绝对值 `x -> |x|`。

<b>`Ceil(operand)`</b> 逐个元素的整数上界 `x -> ⌈x⌉`。

<b>`Cos(operand)`</b> 逐个元素的余弦 `x -> cos(x)`。

<b>`Exp(operand)`</b> 逐个元素的自然幂指数 `x -> e^x`。

<b>`Floor(operand)`</b> 逐个元素的整数下界 `x -> ⌊x⌋`。

<b>`IsFinite(operand)`</b> 测试 `operand` 的每个元素是否是有限的，即不是正无穷或负无穷，也不是 `NaN`。该操作返回一个 `PRED` 值的数组，维度形状与输入一致，数组中的元素当且仅当相应的输入是有限时为 `true`，否则为 `false`。

<b>`Log(operand)`</b> 逐个元素的自然对数 `x -> ln(x)`。

<b>`LogicalNot(operand)`</b> 逐个元素的逻辑非 `x -> !(x)`。

<b>`Neg(operand)`</b> 逐个元素取负值 `x -> -x`。

<b>`Sign(operand)`</b> 逐个元素求符号 `x -> sgn(x)`，其中 

$$\text{sgn}(x) = \begin{cases} -1 & x < 0\\ 0 & x = 0\\ 1 & x > 0 \end{cases}$$

它使用的是 `operand` 的元素类型的比较运算符。

<b>`Tanh(operand)`</b> 逐个元素的双曲正切 `x -> tanh(x)`。


 参数 | 类型 | 语义                                     
--------- | ----------------------- | ---------------------------
`operand` | `XlaOp` | 函数的操作数

该函数应用于 `operand` 数组的每个元素，从而形成具有相同形状的数组。它允许操作数为标量（秩 0 ）

## 收集

XLA 收集操作将一个输入数组的几个片（每个片在一个可能不同的运行时偏移量上）拼接成起来。

### 一般语义

也可以在 [`XlaBuilder::Gather`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) 进行查阅。更直观的描述，请参阅下面的“非正式描述”部分。

<b> `gather(operand, start_indices, offset_dims, collapsed_slice_dims, slice_sizes, start_index_map)` </b>

|参数      | 类型                    | 语义                       |
|----------------- | ----------------------- | --------------------------------|
|`operand`         | `XlaOp` | 我们收集的数组。|
|`start_indices`   | `XlaOp`                 | Array containing the starting indices of the slices we gather.:
|`index_vector_dim` | `int64`                | The dimension in `start_indices` that "contains" the starting indices. See below for a description.  |
|`offset_dims`     | `ArraySlice<int64>`     | The set of dimensions in  the output shape that offset into a array sliced from operand. |
|`slice_sizes`     | `ArraySlice<int64>`      | `slice_sizes[i]` is the bounds for the slice on dimension `i`. |
|`collapsed_slice_dims` | `ArraySlice<int64>` | The set of dimensions in each slice that are collapsed away. These dimensions must have size: 1.                             |
|`start_index_map` | `ArraySlice<int64>`      | A map that describes how to map indices in `start_indices` to to legal indices into operand. |

For convenience, we label dimensions in the output array not in `offset_dims` as `batch_dims`.

The output is an array of rank `batch_dims.size` + `operand.rank` - `collapsed_slice_dims`.size.

If `index_vector_dim` is equal to `start_indices.rank` we implicitly consider `start_indices` to have a trailing `1` dimension (i.e. if `start_indices` was of shape `[6,7]` and `index_vector_dim` is `2` then we implicitly consider the shape of `start_indices` to be `[6,7,1]`).

The bounds for the output array along dimension `i` is computed as follows:

   1. If `i` is present in `batch_dims` (i.e. is equal to `batch_dims[k]` for some `k`) then we pick the corresponding dimension bounds out of `start_indices.shape`, skipping `index_vector_dim` (i.e. pick `start_indices.shape.dims`[`k`] if `k` < `index_vector_dim` and  `start_indices.shape.dims`[`k`+`1`] otherwise).
   2. If `i` is present in `offset_dims` (i.e. equal to `offset_dims`[`k`] for some `k`) then we pick the corresponding bound out of `slice_sizes` after accounting for `collapsed_slice_dims` (i.e. we pick `adjusted_slice_sizes`[`k`] where `adjusted_slice_sizes` is `slice_sizes` with the bounds at indices `collapsed_slice_dims` removed).

Formally, the operand index `In` corresponding to an output index `Out` is computed as follows:

   1. Let `G` = { `Out`[`k`] for `k` in `batch_dims` }.  Use `G` to slice out vector `S` such that `S`[`i`] = `start_indices`[Combine(`G`, `i`)] where Combine(A, b) inserts b at position `index_vector_dim` into A.  Note that this is well defined even if `G` is empty -- if `G` is empty then `S` = `start_indices`.
   2. Create a starting index, `S`<sub>`in`</sub>, into `operand` using `S` by scattering `S` using `start_index_map`.  More precisely:
       1. `S`<sub>`in`</sub>[`start_index_map`[`k`]] = `S`[`k`] if `k` < `start_index_map.size`.
       2. `S`<sub>`in`</sub>[`_`] = `0` otherwise.
   3. Create an index `O`<sub>`in`</sub> into `operand` by scattering the indices at the offset dimensions in `Out` according to the `collapsed_slice_dims` set.  More precisely:
       1. `O`<sub>`in`</sub>[`expand_offset_dims`(`k`)] = `Out`[`offset_dims`[`k`]] if `k` < `offset_dims.size` (`expand_offset_dims` is defined below).
       2. `O`<sub>`in`</sub>[`_`] = `0` otherwise.
  4. `In` 是 `O`<sub>`in`</sub> + `S`<sub>`in`</sub>，是元素级加法。

`expand_offset_dims` is the monotonic function with domain [`0`, `offset.size`) and range [`0`, `operand.rank`) \ `collapsed_slice_dims`.  So if, e.g., `offset.size` is `4`, `operand.rank` is `6` and `collapsed_slice_dims` is {`0`, `2`} then `expand_offset_dims` is {`0`→`1`, `1`→`3`, `2`→`4`, `3`→`5`}.

### 非正式说明和实例

Informally, every index `Out` in the output array corresponds to an element `E` in the operand array, computed as follows:
   - We use the batch dimensions in `Out` to look up a starting index from `start_indices`.
   - We use `start_index_map` to map the starting index (which may have size less than operand.rank) to a "full" starting index into operand.
   - We dynamic-slice out a slice with size `slice_sizes` using the full starting index.
   - We reshape the slice by collapsing the `collapsed_slice_dims` dimensions. Since all collapsed slice dimensions have to have bound 1 this reshape is always legal.
   - We use the offset dimensions in `Out` to index into this slice to get the input element, `E`, corresponding to output index `Out`.

`index_vector_dim` is set to `start_indices.rank` - `1` in all of the examples that follow.  More interesting values for `index_vector_dim` does not change the operation fundamentally, but makes the visual representation more cumbersome.

为了直观地了解所有上述情况如何结合在一起，我们来看一个例子，它从一个 `[16,11]` 数组中收集 5 片形状为 `[8,6]` 的数组。切片到 `[16,11]` 数组中的位置可以表示为形状为 `S64[2]` 的索引向量，所有以 5 个位置的集合可以表示 `S64[5,2]` 数组。

集合操作的行为可以被描述为一个索引转换，采用 [`G`,`O`<sub>`0`</sub>,`O`<sub>`1`</sub>] 输出形状中的索引，并按以下方式将其映射到输入数组中的元素：

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ops_xla_gather_1.svg">
</div>

We first select an (`X`,`Y`) vector from the gather indices array using `G`. The element in the output array at index [`G`,`O`<sub>`0`</sub>,`O`<sub>`1`</sub>] is then the element in the input array at index [`X`+`O`<sub>`0`</sub>,`Y`+`O`<sub>`1`</sub>].

`slice_sizes` 是 `[8,6]`，它决定 W<sub>`0`</sub> 和 W<sub>`1`</sub> 的范围，这反过来决定切片的边界。

此集合操作充当批处理动态切片，`G` 作为批处理维度。

集合指数可能是多方面的，例如，使用形状 `[4,5,2]` 的 "gather indices" 数组的上述示例的一个更一般的版本可以翻译成这样的指数：

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/ops_xla_gather_1.svg">
</div>

同样，这是一个批处理动态切片 `G`<sub>`0`</sub> 和 `G`<sub>`1`</sub>，切片大小仍然是 `[8,6]`。

XLA 中收集的数据操作概括了以上概述的非正式语义：

 1. 在最后一个示例中，我们可以配置输出形状中的哪些维度是 offset 维度（上一个示例中包含 `O`<sub>`0`</sub>，`O`<sub>`1`</sub> 的维数）。输出 batch 的维度（上一个示例中包含 `G`<sub>`0`</sub>，`G`<sub>`1`</sub> 的维数）被定义为不是 offset 的输出维度。

 2. 输出形状中显式显示的输出 offset 维数可能小于输入等级。这些“缺失”的维度显式地列为 `collapsed_slice_dims`，必须有一个切片大小为 `1`。由于它们的切片大小为 `1`，因此它们的唯一有效索引是 `0`，而对它们进行赋值并不会引入歧义。

 3. 从 "Gather Indices" 数组（最后一个示例中的（`X`, `Y`）中提取的切片可能比输入数组级别有更少的元素，并且一个明确的映射指示如何扩展索引，使其与输入具有相同的等级。

最后一个例子，我们使用（2）和（3）来实现 `tf.gather_nd`：

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/ops_xla_gather_2.svg">
</div>

from the gather indices array as usual, except the starting index has only one element, `X`. Similarly, there is only one output offset index with the value `O`<sub>`0`</sub>.  However, before being used as indices into the input array, these are expanded in accordance to "Gather Index Mapping" (`start_index_map` in the formal description) and "Offset Mapping" (`expand_offset_dims` in the formal description) into  [`X`,`0`] and [`0`,`O`<sub>`0`</sub>] respectively, adding up to [`X`,`O`<sub>`0`</sub>].  In other words, the output index [`G`<sub>`0`</sub>,`G`<sub>`1`</sub>,`O`<sub>`0`</sub>] maps to the input index [`GatherIndices`[`G`<sub>`0`</sub>,`G`<sub>`1`</sub>,`0`],`X`] which gives us the semantics for `tf.gather_nd`.

在这种情况下，`slice_sizes` 是 `[1,11]`。直觉上这意味着集合索引数组中的每一个索引 `X` 都会选择整行，结果是所有这些行连在一起。

## GetTupleElement

另请参阅 [`XlaBuilder::GetTupleElement`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

将索引添加到编译时常量的元组中。

该值必须是编译时常量，这样才可以通过形状推断确定结果值的类型。

概念上，这类似于 C++ 中的 `std::get<int N>(t)`：

```
let v: f32[10] = f32[10]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
let s: s32 = 5;
let t: (f32[10], s32) = tuple(v, s);
let element_1: s32 = gettupleelement(t, 1);  // 推断出的形状匹配 s32.
```

另见 `tf.tuple`。

## Infeed

另请参阅 [`XlaBuilder::Infeed`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b> `Infeed(shape)` </b>

| 参数 | 类型 | 语义                                              |
| -------- | ------- | ----------------------------------------------------- |
| `shape`  | `Shape` | 从 Infeed 接口读取数据的维度形状。此形状的数据布局必须与发送到设备上的数据相匹配；否则行为是未定义的 |

从设备的隐式 Infeed 流接口读取单个数据项，根据给定的形状和布局来进行解析，并返回一个此数据的 `XlaOp`。在一个计算中允许有多个 Infeed 操作，但这些 Infeed 操作之间必须是全序的。比如，下面代码中两个 Infeed 是全序的，因为在不同 while 循环之间有依赖关系。

```
result1 = while (condition, init = init_value) {
  Infeed(shape)
}

result2 = while (condition, init = result1) {
  Infeed(shape)
}
```

不支持嵌套的元组形状。对于一个空的元组形状，Infeed 操作通常是一个 no-op，因而不会从设备的 Infeed 中读取任何数据。

> 注意：我们计划允许支持没有全序的多个 Infeed 操作，在这种情况下，编译器将提供信息，确定这些 Infeed 操作在编译后的程序中如何串行化。

## Iota

<b> `Iota()` </b>

Builds a constant literal on device rather than a potentially large host transfer.  Creates a rank 1 tensor of values starting at zero and incrementing by one.

Arguments          | Type            | Semantics
------------------ | --------------- | ---------------------------
`type`             | `PrimitiveType` | type U
`size`             | `int64`         | The number of elements in the tensor.

## 映射（Map）

另请参阅 [`XlaBuilder::Map`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b> `Map(operands..., computation)` </b>

| 参数 | 类型 | 语义                      |
| ----------------- | ------------------------ | ----------------------------- |
| `operands`        | N 个 `XlaOp` 的序列 | 类型为 T_0..T_{N-1} 的 N 个数组 |
| `computation`     | `XlaComputation`    | 类型为`T_0, T_1, ..., T_{N + M -1} -> S` 的计算，有 N 个类型为 T 的参数，和 M 个任意类型的参数 |
| `dimensions`       | `int64` array      | 映射维度的数组  |

将一个标量函数作用于给定的 `operands` 数组，可产生相同维度的数组，其中每个元素都是映射函数（mapped function）作用于相应输入数组中相应元素的结果。

此映射函数可以是任意计算过程，只不过它必须有 N 个类型为 `T` 的标量参数，和单个类型为 `S` 的输出。输出的维度与输入 `operands` 相同，只不过元素类型 T 换成了 S。

比如，`Map(op1, op2, op3, computation, par1)` 用 `elem_out <-
computation(elem1, elem2, elem3, par1)` 将输入数组中的每个（多维）指标映射产生输出数组。

## 填充（Pad）

另请参阅 [`XlaBuilder::Pad`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b> `Pad(operand, padding_value, padding_config)` </b>

| 参数 | 类型 | 语义                  |
| ---------------- | --------------- | ----------------------------- |
| `operand`        | `XlaOp`         | 类型为 `T` 的数组 |
| `padding_value`  | `XlaOp`         | 类型为 `T` 的标量，用于填充 |
| `padding_config` | `PaddingConfig` | 每个维度的两端的填充量 (low, high) |

通过在数组周围和数组之间进行填充，可以将给定的 `operand` 数组扩大，其中 `padding_value` 和 `padding_config` 用于配置每个维度的边缘填充和内部填充的数目。

`PaddingConfig` 是 `PaddingConfigDimension` 的一个重复字段，它对于每个维度都包含有三个字段：`edge_padding_low`, `edge_padding_high` 和 `interior_padding`。

`edge_padding_low` 和 `edge_padding_high` 分别指定了该维度上低端（指标为 0 那端）和高端（最高指标那端）上的填充数目。边缘填充数目可以是负值 — 负的填充数目的绝对值表示从指定维度移除元素的数目。

`interior_padding` 指定了在每个维度的任意两个相邻元素之间的填充数目。逻辑上，内部填充应发生在边缘填充之前，所有在负边缘填充时，会从经过内部填充的操作数之上再移除边缘元素。

如果边缘填充配置为 (0, 0)，且内部填充值都是 0，则此操作是一个 no-op。下图展示的是二维数组上不同 `edge_padding` 和 `interior_padding` 值的示例。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ops_pad.png">
</div>

## Recv

另请参阅 [`XlaBuilder::Recv`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b> `Recv(shape, channel_handle)` </b>

| 参数        | 类型            | 语义                            |
| ---------------- | --------------- | ------------------------------------ |
| `shape`          | `Shape`         | 要接收的数据的形状         |
| `channel_handle` | `ChannelHandle` | 发送/接收对的唯一标识 |

从另一台共享相同通道句柄的计算机的 `Send` 指令接收指定形状的数据，返回一个接收数据的 XlaOp。

客户端 `Recv` 操作的客户端 API 是同步通信。但是，指令内分解成 2 个 HLO 指令（`Recv` 和 `RecvDone`）用于异步数据传输。请参考 [`HloInstruction::CreateRecv` 和 `HloInstruction::CreateRecvDone`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/hlo_instruction.h)

<b>`Recv(const Shape& shape, int64 channel_id)`</b>

分配资源从具有相同 channel_id 的 `Send` 指令接收数据。返回已分配资源的上下文，该上下文随后通过 `RecvDone` 指令等待数据传输完成。上下文是 {接收缓冲区 (形状), 请求标识符（U32）} 的元组，且只能用于 `RecvDone` 指令。

<b> `RecvDone(HloInstruction context)` </b>

给定一个由 `Recv` 指令创建的上下文，等待数据传输完成并返回接收的数据。

## Reduce

另请参阅 [`XlaBuilder::Reduce`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

将一个归约函数作用于一个或多个并行数组。

<b> `Reduce(operands..., init_values..., computation, dimensions)` </b>

Arguments     | Type                  | Semantics
------------- | --------------------- | ---------------------------------------
`operands`    | Sequence of N `XlaOp` | N arrays of types `T_0, ..., T_N`.
`init_values` | Sequence of N `XlaOp` | N scalars of types `T_0, ..., T_N`.
`computation` | `XlaComputation`      | computation of type
              :                       : `T_0, ..., T_N, T_0, ..., T_N -> Collate(T_0, ..., T_N)`
`dimensions`  | `int64` array         | unordered array of dimensions to reduce

Where:

* N is required to be greater or equal to 1.
* All input arrays must have the same dimensions.
* If `N = 1`, `Collate(T)` is `T`.
* If `N > 1`, `Collate(T_0, ..., T_N)` is a tuple of `N` elements of type `T`.

The output of the op is `Collate(Q_0, ..., Q_N)` where `Q_i` is an array of type `T_i`, the dimensions of which are described below.

This operation reduces one or more dimensions of each input array into scalars. The rank of each returned array is `rank(operand) - len(dimensions)`. `init_value` is the initial value used for every reduction and may be inserted anywhere during computation by the back-end. In most cases, `init_value` is an identity of the reduction function (for example, 0 for addition). The applied `computation` is always passed the `init_value` on the left-hand side.

归约函数的执行顺序是任意的，即可能是非确定的。因而，归约函数不应对运算的结合性敏感。

有些归约函数，比如加法，对于浮点数并没有严格遵守结合率。不过，如果数据的范围是有限的，则在大多数实际情况中，浮点加法已经足够满足结合率。当然，我们也可以构造出完全不遵守结合率的归约函数，这时，XLA 归约就会产生不正确或不可预测的结果。

下面是一个示例，对 独立的 1D 数组 [10, 11, 12, 13] 进行归约，归约函数为 `f` （即参数 `computation`），则计算结果为：

`f(10, f(11, f(12, f(init_value, 13)))`

但它还有其它很多种可能性，比如：

`f(init_value, f(f(10, f(init_value, 11)), f(f(init_value, 12), f(init_value, 13))))`

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

When `N > 1`, reduce function application is slightly more complex, as it is applied simultaneously to all inputs. For example, consider the following reduction function, which can be used to compute the max and the argmax of a a 1-D tensor in parallel:

```
f: (Float, Int, Float, Int) -> Float, Int
f(max, argmax, value, index):
  if value >= argmax:
    return (value, index)
  else:
    return (max, argmax)
```

For 1-D Input arrays `V = Float[N], K = Int[N]`, and init values `I_V = Float, I_K =  Int`, the result `f_(N-1)` of reducing across the only input dimension is equivalent to the following recursive application:

```
f_0 = f(I_V, I_K, V_0, K_0)
f_1 = f(f_0.first, f_0.second, V_1, K_1)
...
f_(N-1) = f(f_(N-2).first, f_(N-2).second, V_(N-1), K_(N-1))
```

Applying this reduction to an array of values, and an array of sequential indices (i.e. iota), will co-iterate over the arrays, and return a tuple containing the maximal value and the matching index.

## ReducePrecision

另请参阅 [`XlaBuilder::ReducePrecision`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

当浮点数转换为低精度格式（比如 IEEE-FP16）然后转换回原格式时，值可能会发生变化，ReducePrecision 对这种变化进行建模。低精度格式中的指数（exponent）和尾数（mantissa）的位数目是可以任意指定的，不过不是所有硬件实现都支持所有的位大小。

<b> `ReducePrecision(operand, mantissa_bits, exponent_bits)` </b>

| 参数 | 类型 | 语义                                    |
| ------------------- | ------- | -------------------  |
| `operand`           | `XlaOp` | 浮点类型 `T` 的数组   |
| `exponent_bits`     | `int32` | 低精度格式中的指数位数 |
| `mantissa_bits`     | `int32` | 低精度格式中的尾数位数 |

结果为类型为 `T` 的数组。输入值被舍入至与给定尾数位的数字最接近的那个值（采用的是"偶数优先"原则）。而超过指数位所允许的值域时，输入值会被视为正无穷或负无穷。`NaN` 值会保留，不过它可能会被转换为规范化的 NaN 值。

低精度格式必须至少有一个指数位（为了区分零和无穷，因为两者的尾数位都为零），且尾数位必须是非负的。指数或尾数位可能会超过类型 `T`；这种情况下，相应部分的转换就仅仅是一个 no-op 了。

## ReduceWindow

另请参阅 [`XlaBuilder::ReduceWindow`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

将一个归约函数应用于输入多维数组的每个窗口内的所有元素上，输出一个多维数组，其元素个数等于合法窗口的元素数目。一个池化层可以表示为一个 `ReduceWindow`。Similar to [`Reduce`](#reduce), the applied `computation` is always passed the `init_value` on the left-hand side.

<b> `ReduceWindow(operand, init_value, computation, window_dimensions,
window_strides, padding)` </b>

| 参数 | 类型 | 语义                    |
| ------------------- | ----------------------- | ---------------------------- |
| `operand`           | `XlaOp` | 类型为 T 的 N 维数组。这是窗口放置的底空间区域  |
| `init_value`        | `XlaOp` | 归约的初始值。细节请参见 [规约](#reduce)。 |
| `computation`       | `XlaComputation`           | 类型为 `T, T -> T`的归约函数，应用于每个窗口内的所有元素  |
| `window_dimensions` | `ArraySlice<int64>`     | 表示窗口维度值的整数数组  |
| `window_strides`    | `ArraySlice<int64>`     | 表示窗口步长值的整数数组 |
| `padding`           | `Padding`               | 窗口的边缘填充类型（Padding\:\:kSame 或 Padding\:\:kValid） |

下列代码和图为一个使用 `ReduceWindow` 的示例。输入是一个大小为 [4x6] 的矩阵，window_dimensions 和 window_stride_dimensions 都是 [2x3]。

```
// 创建一个归约计算（求最大值）
XlaComputation max;
{
  XlaBuilder  builder(client_, "max");
  auto y = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "y");
  auto x = builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "x");
  builder.Max(y, x);
  max = builder.Build().ConsumeValueOrDie();
}

// 用最大值归约计算来创建一个 ReduceWindow 计算
XlaBuilder  builder(client_, "reduce_window_2x3");
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

在维度中，步长为 1 表示在此维度上两个相邻窗口间隔一个元素，为了让窗口互相不重叠，window_stride_dimensions 和 window_dimensions 应该要相等。下图给出了两种不同步长设置的效果。边缘填充应用于输入的每个维度，计算过程实际发生在填充之后的数组上。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:75%" src="https://www.tensorflow.org/images/ops_reduce_window_stride.png">
</div>

归约函数的执行顺序是任意的，因而结果可能是非确定性的。所以，归约函数应该不能对计算的结合性太过敏感。更多细节，参见 [`Reduce`](#reduce) 关于结合性的讨论。

## Reshape

另请参阅 [`XlaBuilder::Reshape`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) 和 [`Collapse`](#collapse) 操作。

变形操作（reshape）是将一个数组的维度变成另外一种维度设置。

<b> `Reshape(operand, new_sizes)` </b>
<b> `Reshape(operand, dimensions, new_sizes)` </b>

参数 | 类型 | 语义
------------ | ----------------------- | ---------------------------------------
`operand`    | `XlaOp` | 类型为 T 的数组
`dimensions` | `int64` vector          | 维度折叠的顺序
`new_sizes`  | `int64` vector          | 新维度大小的矢量

从概念上看，变形操作首先将一个数组拉平为一个一维矢量，然后将此矢量展开为一个新的形状。输入参数是一个类型为 T 的任意数组，一个编译时常量的维度指标数组，以及表示结果维度大小的一个编译时常量的数组。如果给出了 `dimensions` 参数，这个矢量中的值必须是 T 的所有维度的一个置换，其默认值为 `{0, ..., rank - 1}`。`dimensions` 中的维度的顺序是从最慢变化维（最主序）到最快变化维（最次序），按照这个顺序依次将所有元素折叠到一个维度上。`new_sizes` 矢量决定了输出数组的维度大小。`new_sizes[0]` 表示第 0 维的大小，`new_sizes[1]` 表示的是第 1 维的大小，依此类推。`new_sizes` 中的维度值的乘积必须等于 operand 的维度值的乘积。将折叠的一维数组展开为由 `new_sizes` 定义的多维数组时，`new_sizes` 中的维度的顺序也是最慢变化维（最主序）到最快变化维（最次序）。

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

## Rev（反转）

另请参阅 [`XlaBuilder::Rev`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b>`Rev(operand, dimensions)`</b>

参数 | 类型 | 语义
------------ | ----------------------- | ---------------------
`operand`    | `XlaOp` | 类型为 T 的数组 
`dimensions` | `ArraySlice<int64>`     | 待反转的维度

反转操作是将 `operand` 数组沿指定的维度 `dimensions` 对元素的顺序反转，产生一个形状相同的数组。operand 数组的每个元素被存储在输出数组的变换后的位置上。元素的原索引位置在每个待倒置维度上都被反转了，得到其在输出数组中的索引位置（即，如果一个大小为 N 的维度是待倒置的，则索引 i 被变换为 N-i-i）。

`Rev` 操作的一个用途是在神经网络的梯度计算时沿两个窗口维度对卷积权重值进行倒置。

## RngNormal

另请参阅 [`XlaBuilder::RngNormal`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。


Constructs an output of a given shape with random numbers generated following the $$N(\mu, \sigma)$$ normal distribution. The parameters $$\mu$$ and $$\sigma$$, and output shape have to have a floating point elemental type. The parameters furthermore have to be scalar valued.

<b>`RngNormal(mu, sigma, shape)`</b>

| Arguments | Type    | Semantics                                           |
| --------- | ------- | --------------------------------------------------- |
| `mu`      | `XlaOp` | Scalar of type T specifying mean of generated numbers |
| `sigma`   | `XlaOp` | Scalar of type T specifying standard deviation of generated numbers |
| `shape`   | `Shape` | Output shape of type T                              |

## RngUniform

See also [`XlaBuilder::RngUniform`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h).

Constructs an output of a given shape with random numbers generated following the uniform distribution over the interval $$[a,b)$$. The parameters and output element type have to be a boolean type, an integral type or a floating point types, and the types have to be consistent. The CPU and GPU backends currently only support F64, F32, F16, BF16, S64, U64, S32 and U32. Furthermore, the parameters need to be scalar valued. If $$b <= a$$ the result is implementation-defined.

<b>`RngUniform(a, b, shape)`</b>

| Arguments | Type                    | Semantics                         |
| --------- | ----------------------- | --------------------------------- |
| `a`       | `XlaOp`                 | Scalar of type T specifying lower limit of interval |
| `b`       | `XlaOp`                 | Scalar of type T specifying upper limit of interval |
| `shape`   | `Shape`                 | Output shape of type T            |

## Scatter

The XLA scatter operation generates a result which is the value of the input
tensor `operand`, with several slices (at indices specified by
`scatter_indices`) updated with the values in `updates` using
`update_computation`.

See also
[`XlaBuilder::Scatter`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h).

<b> `scatter(operand, scatter_indices, updates, update_computation, index_vector_dim, update_window_dims, inserted_window_dims, scatter_dims_to_operand_dims)` </b>

|Arguments         | Type                   | Semantics                        |
|------------------|------------------------|----------------------------------|
|`operand`         | `XlaOp`                | Tensor to be scattered into.     |
|`scatter_indices` | `XlaOp`                | Tensor containing the starting indices of the slices that must be scattered to. |
|`updates`         | `XlaOp`                | Tensor containing the values that must be used for scattering. |
|`update_computation`| `XlaComputation`     | Computation to be used for       |
:                  :                        : combining the existing values in :
:                  :                        : the input tensor and the updates :
:                  :                        : during scatter. This computation :
:                  :                        : should be of type `T, T -> T`.   :
|`index_vector_dim`| `int64`                | The dimension in                 |
:                  :                        : `scatter_indices` that contains  :
:                  :                        : the starting indices.            :
|`update_window_dims`| `ArraySlice<int64>`  | The set of dimensions in         |
:                  :                        : `updates` shape that are _window :
:                  :                        : dimensions_.                     :
|`inserted_window_dims`| `ArraySlice<int64>`| The set of _window dimensions_   |
:                  :                        : that must be inserted into       :
:                  :                        : `updates` shape.                 :
|`scatter_dims_to_operand_dims`| `ArraySlice<int64>`  | A dimensions map from  |
:                  :                        : the scatter indices to the       :
:                  :                        : operand index space. This array  :
:                  :                        : is interpreted as mapping `i` to :
:                  :                        : `scatter_dims_to_operand_dims[i]`:
:                  :                        : . It has to be one-to-one and    :
:                  :                        : total.                           :

If `index_vector_dim` is equal to `scatter_indices.rank` we implicitly consider `scatter_indices` to have a trailing `1` dimension.

We define `update_scatter_dims` of type `ArraySlice<int64>` as the set of dimensions in `updates` shape that are not in `update_window_dims`, in ascending order.

The arguments of scatter should follow these constraints:

  - `updates` tensor must be of rank `update_window_dims.size + scatter_indices.rank - 1`.

  - Bounds of dimension `i` in `updates` must conform to the following:
      - If `i` is present in `update_window_dims` (i.e. equal to `update_window_dims`[`k`] for some `k`), then the bound of dimension `i` in `updates` must not exceed the corresponding bound of `operand` after accounting for the `inserted_window_dims` (i.e.  `adjusted_window_bounds`[`k`], where `adjusted_window_bounds` contains the bounds of `operand` with the bounds at indices `inserted_window_dims` removed).
      - If `i` is present in `update_scatter_dims` (i.e. equal to `update_scatter_dims`[`k`] for some `k`), then the bound of dimension `i` in `updates` must be equal to the corresponding bound of `scatter_indices`, skipping `index_vector_dim` (i.e. `scatter_indices.shape.dims`[`k`], if `k` < `index_vector_dim` and `scatter_indices.shape.dims`[`k+1`] otherwise).

  - `update_window_dims` must be in ascending order, not have any repeating dimension numbers, and be in the range `[0, updates.rank)`.

  - `inserted_window_dims` must be in ascending order, not have any repeating dimension numbers, and be in the range `[0, operand.rank)`.

  - `scatter_dims_to_operand_dims.size` must be equal to `scatter_indices`[`index_vector_dim`], and its values must be in the range  `[0, operand.rank)`.

For a given index `U` in the `updates` tensor, the corresponding index `I` in the `operand` tensor into which this update has to be applied is computed as follows:

  1. Let `G` = { `U`[`k`] for `k` in `update_scatter_dims` }. Use `G` to look up an index vector `S` in the `scatter_indices` tensor such that `S`[`i`] = `scatter_indices`[Combine(`G`, `i`)] where Combine(A, b) inserts b at positions `index_vector_dim` into A.
  2. Create an index `S`<sub>`in`</sub> into `operand` using `S` by scattering `S` using the `scatter_dims_to_operand_dims` map. More formally:
       1. `S`<sub>`in`</sub>[`scatter_dims_to_operand_dims`[`k`]] = `S`[`k`] if `k` < `scatter_dims_to_operand_dims.size`.
       2. `S`<sub>`in`</sub>[`_`] = `0` otherwise.
  3. Create an index `W`<sub>`in`</sub> into `operand` by scattering the indices at `update_window_dims` in `U` according to `inserted_window_dims`.
     More formally:
       1. `W`<sub>`in`</sub>[`window_dims_to_operand_dims`(`k`)] = `U`[`k`] if `k` < `update_window_dims.size`, where `window_dims_to_operand_dims` is the monotonic function with domain [`0`, `update_window_dims.size`) and range [`0`, `operand.rank`) \\ `inserted_window_dims`. (For example, if `update_window_dims.size` is `4`, `operand.rank` is `6`, and `inserted_window_dims` is {`0`, `2`} then `window_dims_to_operand_dims` is {`0`→`1`, `1`→`3`, `2`→`4`, `3`→`5`}).
       2. `W`<sub>`in`</sub>[`_`] = `0` otherwise.
  4. `I` is `W`<sub>`in`</sub> + `S`<sub>`in`</sub> where + is element-wise addition.

In summary, the scatter operation can be defined as follows.

   - Initialize `output` with `operand`, i.e. for all indices `O` in the `operand` tensor:\
       `output`[`O`] = `operand`[`O`]
   - For every index `U` in the `updates` tensor and the corresponding index `O` in the `operand` tensor:\
       `output`[`O`] = `update_computation`(`output`[`O`], `updates`[`U`])

The order in which updates are applied is non-deterministic. So, when multiple indices in `updates` refer to the same index in `operand`, the corresponding value in `output` will be non-deterministic.

Note that the first parameter that is passed into the `update_computation` will always be the current value from the `output` tensor and the second parameter will always be the value from the `updates` tensor. This is important specifically for cases when the `update_computation` is _not commutative_.

Informally, the scatter op can be viewed as an _inverse_ of the gather op, i.e. the scatter op updates the elements in the input that are extracted by the corresponding gather op.

For a detailed informal description and examples, refer to the "Informal Description" section under `Gather`.

## Select

See also
[`XlaBuilder::Select`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h).

Constructs an output array from elements of two input arrays, based on the values of a predicate array.

<b> `Select(pred, on_true, on_false)` </b>

Arguments  | Type    | Semantics
---------- | ------- | ------------------
`pred`     | `XlaOp` | array of type PRED
`on_true`  | `XlaOp` | array of type T
`on_false` | `XlaOp` | array of type T

The arrays `on_true` and `on_false` must have the same shape. This is also the shape of the output array. The array `pred` must have the same dimensionality as `on_true` and `on_false`, with the `PRED` element type.

For each element `P` of `pred`, the corresponding element of the output array is taken from `on_true` if the value of `P` is `true`, and from `on_false` if the value of `P` is `false`. As a restricted form of [broadcasting](broadcasting.md), `pred` can be a scalar of type `PRED`. In this case, the output array is taken wholly from `on_true` if `pred` is `true`, and from `on_false` if `pred` is `false`.

Example with non-scalar `pred`:

```
let pred: PRED[4] = {true, false, false, true};
let v1: s32[4] = {1, 2, 3, 4};
let v2: s32[4] = {100, 200, 300, 400};
==>
Select(pred, v1, v2) = s32[4]{1, 200, 300, 4};
```

Example with scalar `pred`:

```
let pred: PRED = true;
let v1: s32[4] = {1, 2, 3, 4};
let v2: s32[4] = {100, 200, 300, 400};
==>
Select(pred, v1, v2) = s32[4]{1, 2, 3, 4};
```

Selections between tuples are supported. Tuples are considered to be scalar types for this purpose. If `on_true` and `on_false` are tuples (which must have the same shape!) then `pred` has to be a scalar of type `PRED`.

## SelectAndScatter

See also
[`XlaBuilder::SelectAndScatter`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h).

This operation can be considered as a composite operation that first computes `ReduceWindow` on the `operand` array to select an element from each window, and then scatters the `source` array to the indices of the selected elements to construct an output array with the same shape as the operand array. The binary `select` function is used to select an element from each window by applying it across each window, and it is called with the property that the first parameter's index vector is lexicographically less than the second parameter's index vector. The `select` function returns `true` if the first parameter is selected and returns `false` if the second parameter is selected, and the function must hold transitivity (i.e., if `select(a, b)` and `select(b, c)` are `true`, then `select(a, c)` is also `true`) so that the selected element does not depend on the order of the elements traversed for a given window.

The function `scatter` is applied at each selected index in the output array. It takes two scalar parameters:

1.  Current value at the selected index in the output array
2.  The scatter value from `source` that applies to the selected index

It combines the two parameters and returns a scalar value that's used to update the value at the selected index in the output array. Initially, all indices of the output array are set to `init_value`.

The output array has the same shape as the `operand` array and the `source` array must have the same shape as the result of applying a `ReduceWindow` operation on the `operand` array. `SelectAndScatter` can be used to backpropagate the gradient values for a pooling layer in a neural network.

<b>`SelectAndScatter(operand, select, window_dimensions, window_strides,
padding, source, init_value, scatter)`</b>

| Arguments           | Type                | Semantics                        |
| ------------------- | ------------------- | -------------------------------- |
| `operand`           | `XlaOp`             | array of type T over which the   |
:                     :                     : windows slide                    :
| `select`            | `XlaComputation`    | binary computation of type `T, T |
:                     :                     : -> PRED`, to apply to all        :
:                     :                     : elements in each window; returns :
:                     :                     : `true` if the first parameter is :
:                     :                     : selected and returns `false` if  :
:                     :                     : the second parameter is selected :
| `window_dimensions` | `ArraySlice<int64>` | array of integers for window     |
:                     :                     : dimension values                 :
| `window_strides`    | `ArraySlice<int64>` | array of integers for window     |
:                     :                     : stride values                    :
| `padding`           | `Padding`           | padding type for window          |
:                     :                     : (Padding\:\:kSame or             :
:                     :                     : Padding\:\:kValid)               :
| `source`            | `XlaOp`             | array of type T with the values  |
:                     :                     : to scatter                       :
| `init_value`        | `XlaOp`             | scalar value of type T for the   |
:                     :                     : initial value of the output      :
:                     :                     : array                            :
| `scatter`           | `XlaComputation`    | binary computation of type `T, T |
:                     :                     : -> T`, to apply each scatter     :
:                     :                     : source element with its          :
:                     :                     : destination element              :

The figure below shows examples of using `SelectAndScatter`, with the `select` function computing the maximal value among its parameters. Note that when the windows overlap, as in the figure (2) below, an index of the `operand` array may be selected multiple times by different windows. In the figure, the element of value 9 is selected by both of the top windows (blue and red) and the binary addition `scatter` function produces the output element of value 8 (2 + 6).

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%"
    src="https://www.tensorflow.org/images/ops_scatter_to_selected_window_element.png">
</div>

The evaluation order of the `scatter` function is arbitrary and may be non-deterministic. Therefore, the `scatter` function should not be overly sensitive to reassociation. See the discussion about associativity in the context of [`Reduce`](#reduce) for more details.

## Send

See also
[`XlaBuilder::Send`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h).

<b> `Send(operand, channel_handle)` </b>

Arguments        | Type            | Semantics
---------------- | --------------- | -----------------------------------------
`operand`        | `XlaOp`         | data to send (array of type T)
`channel_handle` | `ChannelHandle` | unique identifier for each send/recv pair

Sends the given operand data to a `Recv` instruction in another computation that shares the same channel handle. Does not return any data.

Similar to the `Recv` operation, the client API of `Send` operation represents synchronous communication, and is internally decomposed into 2 HLO instructions (`Send` and `SendDone`) to enable asynchronous data transfers. See also [`HloInstruction::CreateSend` and `HloInstruction::CreateSendDone`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/hlo_instruction.h).

<b>`Send(HloInstruction operand, int64 channel_id)`</b>

Initiates an asynchronous transfer of the operand to the resources allocated by the `Recv` instruction with the same channel id. Returns a context, which is used by a following `SendDone` instruction to wait for the completion of the data transfer. The context is a tuple of {operand (shape), request identifier (U32)} and it can only be used by a `SendDone` instruction.

<b> `SendDone(HloInstruction context)` </b>

Given a context created by a `Send` instruction, waits for the data transfer to complete.  The instruction does not return any data.

<b> Scheduling of channel instructions </b>

The execution order of the 4 instructions for each channel (`Recv`, `RecvDone`, `Send`, `SendDone`) is as below.

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:70%" src="../../images/send_recv_order.png">
</div>

* `Recv` happens before `Send`
* `Send` happens before `RecvDone`
* `Recv` happens before `RecvDone`
* `Send` happens before `SendDone`

When the backend compilers generate a linear schedule for each computation that communicates via channel instructions, there must not be cycles across the computations. For example, below schedules lead to deadlocks.

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/send_recv_schedule.png">
</div>

## Slice

See also
[`XlaBuilder::Slice`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h).

Slicing extracts a sub-array from the input array. The sub-array is of the same rank as the input and contains the values inside a bounding box within the input array where the dimensions and indices of the bounding box are given as arguments to the slice operation.

<b> `Slice(operand, start_indices, limit_indices)` </b>

| Arguments       | Type                | Semantics                            |
| --------------- | ------------------- | ------------------------------------ |
| `operand`       | `XlaOp`             | N dimensional array of type T        |
| `start_indices` | `ArraySlice<int64>` | List of N integers containing the    |
:                 :                     : starting indices of the slice for    :
:                 :                     : each dimension. Values must be       :
:                 :                     : greater than or equal to zero.       :
| `limit_indices` | `ArraySlice<int64>` | List of N integers containing the    |
:                 :                     : ending indices (exclusive) for the   :
:                 :                     : slice for each dimension. Each value :
:                 :                     : must be greater than or equal to the :
:                 :                     : respective `start_indices` value for :
:                 :                     : the dimension and less than or equal :
:                 :                     : to the size of the dimension.        :

1-dimensional example:

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
Slice(a, {2}, {4}) produces:
  {2.0, 3.0}
```

2-dimensional example:

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

## Sort

See also [`XlaBuilder::Sort`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h).

There are two versions of the Sort instruction: a single-operand and a two-operand version.

<b>`Sort(operand)`</b>

Arguments   | Type    | Semantics
----------- | ------- | --------------------
`operand`   | `XlaOp` | The operand to sort.
`dimension` | `int64` | The dimension along which to sort.

Sorts the elements in the operand in ascending order along the provided dimension. For example, for a rank-2 (matrix) operand, a `dimension` value of 0 will sort each column independently, and a `dimension` value of 1 will sort each row independently. If the operand's elements have floating point type, and the operand contains NaN elements, the order of elements in the output is implementation-defined.

<b>`Sort(key, value)`</b>

Sorts both the key and the value operands. The keys are sorted as in the single-operand version. The values are sorted according to the order of their corresponding keys. For example, if the inputs are `keys = [3, 1]` and `values = [42, 50]`, then the output of the sort is the tuple `{[1, 3], [50, 42]}`.

The sort is not guaranteed to be stable, that is, if the keys array contains duplicates, the order of their corresponding values may not be preserved.

Arguments   | Type    | Semantics
----------- | ------- | -------------------
`keys`      | `XlaOp` | The sort keys.
`values`    | `XlaOp` | The values to sort.
`dimension` | `int64` | The dimension along which to sort.

The `keys` and `values` must have the same dimensions, but may have different element types.

## Transpose

See also the `tf.reshape` operation.

<b>`Transpose(operand)`</b>

Arguments     | Type                | Semantics
------------- | ------------------- | ------------------------------
`operand`     | `XlaOp`             | The operand to transpose.
`permutation` | `ArraySlice<int64>` | How to permute the dimensions.


Permutes the operand dimensions with the given permutation, so
`∀ i . 0 ≤ i < rank ⇒ input_dimensions[permutation[i]] = output_dimensions[i]`.

This is the same as Reshape(operand, permutation,
                            Permute(permutation, operand.shape.dimensions)).

## Tuple

See also [`XlaBuilder::Tuple`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h).

A tuple containing a variable number of data handles, each of which has its own shape.

This is analogous to `std::tuple` in C++. Conceptually:

```
let v: f32[10] = f32[10]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
let s: s32 = 5;
let t: (f32[10], s32) = tuple(v, s);
```

Tuples can be deconstructed (accessed) via the [`GetTupleElement`](#gettupleelement) operation.

## While

See also [`XlaBuilder::While`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h).

<b> `While(condition, body, init)` </b>

| Arguments   | Type             | Semantics                                |
| ----------- | ---------------- | ---------------------------------------- |
| `condition` | `XlaComputation` | XlaComputation of type `T -> PRED` which defines the termination condition of the loop. |
| `body`      | `XlaComputation` | XlaComputation of type `T -> T` which defines the body of the loop. |
| `init`      | `T`              | Initial value for the parameter of `condition` and `body`. |

Sequentially executes the `body` until the `condition` fails. This is similar to a typical while loop in many other languages except for the differences and restrictions listed below.

*   A `While` node returns a value of type `T`, which is the result from the last execution of the `body`.
*   The shape of the type `T` is statically determined and must be the same across all iterations.

The T parameters of the computations are initialized with the `init` value in the first iteration and are automatically updated to the new result from `body` in each subsequent iteration.

One main use case of the `While` node is to implement the repeated execution of training in neural networks. Simplified pseudocode is shown below with a graph that represents the computation. The code can be found in [`while_test.cc`](https://www.tensorflow.org/code/tensorflow/compiler/xla/tests/while_test.cc). The type `T` in this example is a `Tuple` consisting of an `int32` for the iteration count and a `vector[10]` for the accumulator. For 1000 iterations, the loop keeps adding a constant vector to the accumulator.

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
