# 操作语义

本文档介绍了在 [`XlaBuilder`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) 接口中定义的操作语义。通常来说，这些操作与 [`xla_data.proto`](https://www.tensorflow.org/code/tensorflow/compiler/xla/xla_data.proto) 中 RPC 接口所定义的操作是一一对应的。

关于术语：广义数据类型 XLA 处理的是一个 N - 维数组，其元素均为某种数据类型（如 32 位浮点数）。在本文档中，**数组** 表示任意维度的数组。为方便起见，有些特例使用人们约定俗成的更具体和更熟悉的名称；比如，1 维数组称为**向量**，2 维数组称为**矩阵**。

## AllToAll

也可查看 [`XlaBuilder::AllToAll`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

Alltoall 是一个将数据在所有核心间互相传送的集合操作。它可分为两个阶段：

1. 分散阶段。在每个核心上，操作数会按 `split_dimensisons` 分割成 `split_count` 个块，并且这些块会分散到所有核心上，比如，第 i 个块会被送至第 i 个核心。
2. 聚合阶段。每个核心会根据 `concat_dimension` 联结所收到的块。

参与的核心可由以下参数进行配置：

-   `replica_groups`：每个 ReplicaGroup 都有一个包含所有副本 id 的数组。如果其为空，所有副本会按照 0-(n-1) 的顺序编入一个组中。Alltoall 将会按照特定的顺序应用到子组中。例如： 副本组为 {{1, 2, 3}, {4, 5, 0}}，Alltoall 操作会应用到副本 1, 2, 3 中，并在聚合阶段 ，所有收到的块会按照 1, 2, 3 的顺序联结；另一个 Alltoall 则会应用到副本 4, 5, 0 中，并按 4, 5, 0 的顺序联结。

先决条件：

-   split_dimission 中操作数的维度大小要能被 split_count 整除。
-   操作数的形状不能是 tuple。

<b> `AllToAll(operand, split_dimension, concat_dimension, split_count,
replica_groups)` </b>

| 参数               | 类型                   | 语义                            |
| ------------------ | --------------------- | ------------------------------- |
| `operand`          | `XlaOp`               | n 维输入操作数                   |
| `split_dimension`  | `int64`               | 要将操作数分割成的维度数，其介于 `[0,  n]` 间  |
| `concat_dimension` | `int64`               | 将分割块联结起来的维度数，其介于 `[0, n]` 间  |
| `split_count`      | `int64`               | 参与操作的核心数，如果 `replica_groups` 为空，其为副本数；否则，其为每个组中的副本数。|
| `replica_groups`   | `ReplicaGroup` 向量   | 每一个组中包含一个副本 id 的数组  |

下面是 Alltoall 的一个样例。

```
XlaBuilder b("alltoall");
auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4, 16}), "x");
AllToAll(x, /*split_dimension=*/1, /*concat_dimension=*/0, /*split_count=*/4);
```

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/xla/ops_alltoall.png">
</div>

这个例子中，共有 4 个核心参与到 Alltoall 操作中。在每个核心上，操作数会按维度 0 被切割成 4 份，所以每一部分的形状是 f32[4,4]。这 4 部分会分散到所有核心中。然后每个核心会按维度 1 联结接收到的数据，这里的顺序为核心 0-4。所以每个核心的输出的形状都应该是 f32[16,4]。

## BatchNormGrad

算法详情参见 [`XlaBuilder::BatchNormGrad`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) 和 [batch normalization 原始论文](https://arxiv.org/abs/1502.03167)。

计算 batch norm 的梯度

<b> `BatchNormGrad(operand, scale, mean, variance, grad_output, epsilon, feature_index)` </b>

| 参数             | 类型   | 语义                              |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `XlaOp` | 待归一化的 n 维数组 （x）            |
| `scale`         | `XlaOp` | 1 维数组 (\\(\gamma\\))           |
| `mean`          | `XlaOp` | 1 维数组 (\\(\mu\\))              |
| `variance`      | `XlaOp` | 1 维数组 (\\(\sigma^2\\))         |
| `grad_output`   | `XlaOp` | 传入 `BatchNormTraining` 的梯度(\\( \nabla y\\)) |
| `epsilon`       | `float` | ε 值 (\\(\epsilon\\))            |
| `feature_index` | `int64` |`operand` 中的特征维数索引          |

对于特征维数中的每一个特征（`feature_index` 即 `operand` 中特征维度的索引），此操作计算 `operand` 的梯度、在所有其他维度上的 `offset` 和 `scale`。`feature_index` 必须是 `operand` 中特征维度的合法索引。

这三个梯度按照以下规则定义（假设一个 `operand` 的四维张量并有特征维度索引 \\(l\\)，批大小 `m` 和空间大小 `w` 和 `h`。

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

|参数               | 类型                    | 语义|
|----------------- | ----------------------- | -------------------------------|
|`operand`         | `XlaOp`                 | 待复制的数组|
|`broadcast_sizes` | `ArraySlice<int64>`     | 新维度的形状大小|

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

`feature_group_count` 参数（默认值为 1）可被用于分组卷积。`feature_group_count` 应为一个结合输入和输出特征维度的因数。如果 `feature_group_count` 大于 1，其意味着理论上输入和输出特征维度以及 `rhs` 输出特征维度均匀的分散在 `feature_group_count` 个分组中，并且这些组都包含连贯的特征序列。`rhs` 的输入特征维度需要等于 `lhs` 输入特征维度按 `feature_group_count` 分割而得的维度(所以它已经包含输入特征的分组大小)。这 i 个分组会一起计算 `feature_group_count` 分离的卷积。这些卷积额输出会在输出特征维度上联结起来。

对 depthwise 卷积而言，`feature_group_count` 参数将会被设为输入特征维度，并且过滤器会从 `[filter_height, filter_width, in_channels, channel_multiplier]` 重整为 `[filter_height, filter_width, 1, in_channels * channel_multiplier]`。更多细节请参考 `tf.nn.depthwise_conv2d`。

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

|参数                | 类型           | 语义               |
|------------------ | --------------- | ------------------|
|`operand`          | `XlaOp`         | D 维类型为 T 的数组|
|`new_element_type` | `PrimitiveType` | 类型 U            |

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

| 参数         |类型       | 语义                      |
| ------------ | ------- | ---------------- |
| `operand`    | `XlaOp` | 跨多个副本待求和的数组。  |
| `replica_group_ids`    | `int64` 向量 | 每个副本的 Group ID |

输出的维度形状与输入形状一样。比如，如果有两个副本，而操作数在这两个副本上的值分别为 `(1.0, 2.5)` 和 `(3.0, 5.25)`，则此操作在两个副本上的输出值都是 `(4.0, 7.75)`。

`replica_group_ids` 明确每一个副本的分组 id。分组 id 必须为空（所有副本都属于同一组）或每组包含相同数量的副本数。例如，如果 `replica_group_ids` = {0, 1, 2, 3, 0, 1, 2, 3} 既有八个副本，且有四个副本 ID 的子组：{0, 4}、{1, 5}、{2, 6} 和 {3, 7}。每一个子组的大小**必须**一致，例如，对四个副本使用：`replica_group_ids` = {0, 1, 2, 0} 是无效的。

计算 CrossReplicaSum 的结果需要从每个副本中获得一个输入，所以，如果一个副本执行一个 CrossReplicaSum 结点的次数多于其它副本，则前一个副本将永久等待。因此这些副本都运行的是同一个程序，这种情况发生的机会并不多，其中一种可能的情况是，一个 while 循环的条件依赖于输入的数据，而被输入的数据导致此循环在一个副本上执行的次数多于其它副本。

## CustomCall

另请参阅 [`XlaBuilder::CustomCall`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

在计算中调用由用户提供的函数。

<b> `CustomCall(target_name, args..., shape)` </b>

| 参数          | 类型                | 语义                                 |
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

| 参数 | 类型 | 语义 |
|--------- | ------- | ---------------|
|`lhs`     | `XlaOp` | 类型为 T 的数组|
|`rhs`     | `XlaOp` | 类型为 T 的数组|

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

| 参数 | 类型                    | 语义|
| --------- | ----------------------- | ---------------|
| `lhs`     | `XlaOp` | 类型为 T 的数组|
| `rhs`     | `XlaOp` | 类型为 T 的数组|
| `dimension_numbers` | `DotDimensionNumbers` | 类型为 T 的数组|

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

| 输入                               | 输出            | 语义        |
| ----------------------------------- | ----------------- | ---------------- |
| [b0, m, k] `dot` [b0, k, n]         | [b0, m, n]        |  批进行矩阵相乘    |
| [b0, b1, m, k] `dot` [b0, b1, k, n] | [b0, b1, m, n]    |  批进行矩阵相乘    |

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

在执行切片操作之前，通过对 `[1, N)` 中的每个索引 `i`  应用以下转换来计算有效切片索引： 

```
start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i] - size_indices[i])
```

这可确保提取的切片相对于操作数组处于边界内。如果切片在应用变换之前处于边界内，则变换不起作用。

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
| `start_indices` | `XlaOp` | N 个整数组成的秩为 1 的数组，其中包含每个维度的起始切片索引。值必须大于等于 0       |

在执行切片操作之前，通过对 `[1, N)` 中的每个索引 `i` 应用以下转换来计算有效切片索引：

```
start_indices[i] = clamp(start_indices[i], 0, operand.dimension_size[i] - update.dimension_size[i])
```

这可确保更新后的切片始终相对于操作组处于边界内。如果切片在应用变换之前处于边界内，则变换不起作用。

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

|参数 | 类型 | 语义|                                     
| ------------ | ------- | ------ |
|`lhs`     | `XlaOp` | 左操作数：类型为 T 的数组|
|`rhs`     | `XlaOp` | 右操作数：类型为 T 的数组|

这两个参数的维度形状要么相似，要么兼容。关于维度形状相似或兼容的准确含义，参见[广播](../../performance/xla/broadcasting.md)文档。二元操作的结果有一个形状，它是广播两个输入数组的结果。虽然可以广播，但不同秩的数组之间的运算是不支持的，除非其中之一是标量。

当 `Op` 为 `Rem` 时，结果的符号与被除数一致，而结果的绝对值总是小于除数的绝对值。

整数除法溢出（有符号/无符号除或取余零或有符号数除或取余使用 `-1` 的 `INT_SMIN`）会产生一个由实现过程定义的值。

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


| 参数 | 类型 | 语义|                                     
|--------- | ----------------------- | ---------------------------|
|`operand` | `XlaOp` | 函数的操作数|

该函数应用于 `operand` 数组的每个元素，从而形成具有相同形状的数组。它允许操作数为标量（秩 0 ）

## 收集

XLA 收集操作将一个输入数组的几个片（每个片在一个可能不同的运行时偏移量上）拼接成起来。

### 一般语义

也可以在 [`XlaBuilder::Gather`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h) 进行查阅。更直观的描述，请参阅下面的“非正式描述”部分。

<b> `gather(operand, start_indices, offset_dims, collapsed_slice_dims, slice_sizes, start_index_map)` </b>

|参数      | 类型                    | 语义                       |
|----------------- | ----------------------- | --------------------------------|
|`operand`         | `XlaOp` | 我们收集的数组。|
|`start_indices`   | `XlaOp`                 | 包含我们收集的切片的起始索引。|
|`index_vector_dim` | `int64`                | `start_indices` 中的维度，其中“包含”了起始索引，请参考下面的详细解释。|
|`offset_dims`     | `ArraySlice<int64>`     | 输出形状中从操作数分割成数组的一组维数。 |
|`slice_sizes`     | `ArraySlice<int64>`      | `slice_sizes[i]` 是维度 `i` 上切片的界限。|
|`collapsed_slice_dims` | `ArraySlice<int64>` | 折叠起来的每个切片中的一组维度。这些标注的尺寸必须为：1。   |
|`start_index_map` | `ArraySlice<int64>`      | 描述如何将 `start_indices` 中的索引映射到操作数中合法索引的一个映射。 |

为了方便起见，我们将输出数组中的维度标记为 `batch_dims`，而不是 `offset_dims`。

输出是一个数组，其大小为秩 `batch_dims.size` + `operand.rank` - `collapsed_slice_dims`.size。

如果 `index_vector_dim` 等于 `start_indices.rank` ，我们默认 `start_indices` 其后会有一个 `1` 维度（即如果 `start_indices` 形状为 `[6,7]` 且 `index_vector_dim` 为 `2`，那么我们默认  `start_indices` 形状为 `[6,7,1]`）。

输出数组沿维度 `i` 的界限计算如下：

   1. 如果 `i` 存在于 `batch_dims` 中（例如，对于某些 `k`，等于 `batch_dims[k]`），则我们从 `start_indices,shape` 中选择相应的维边界，跳过 `index_vector_dim`（即如果是 `k` < `index_vector_dim`，选择 `start_indices.shape.dims`[`k`]，否则选择 `start_indices.shape.dims`[`k`+`1`]）。
   2. 如果 `i` 出现在 `offset_dims`（例如，对于某些 `k`，等于  `offset_dims`[`k`]），那么在考虑 `collapsed_slice_dims` 后我们从 `slice_sizes` 选择相应的绑定（即我们选择 `adjusted_slice_sizes`[`k`] ，其中 `adjusted_slice_sizes` 是将索引 `collapsed_slice_dims` 界限外移除后的 `slice_sizes`）。 

形式上，对应与输出索引的 `Out` 的操作数索引 `In` 按照以下方式计算：

   1. 使 `G` = { `Out`[`k`] for `k` in `batch_dims` }。用 `G` 将 向量 `S` 分离，比如 `S`[`i`] = `start_indices`[Combine(`G`, `i`)]，其中 Combine(A, b)  将 b 插入 A 中的 `index_vector_dim` 位置。注意这在 `G` 为空时也成立，如果 `G` 为空则 `S` = `start_indices`。
   2. 创建一个起始索引，`S`<sub>`in`</sub>，通过 `start_index_map` 分散 `S` 来将 `S` 插入 `operand`。更确切得来说：
       1. `S`<sub>`in`</sub>[`start_index_map`[`k`]] = `S`[`k`] 如果 `k` < `start_index_map.size`。
       2. 否则，`S`<sub>`in`</sub>[`_`] = `0`。
  3. 创建索引 `O`<sub>`in`</sub>，通过将 `Out` 中偏移维度中的索引按照 `collapsed_slice_dims` 分散到 `operand` 中。更确切的来说：
       1. `O`<sub>`in`</sub>[`expand_offset_dims`(`k`)] = `Out`[`offset_dims`[`k`]] 如果 `k` < `offset_dims.size`（`expand_offset_dims`  的定义在下方）。
       2. 否则，`O`<sub>`in`</sub>[`_`] = `0`。
  4. `In` 是 `O`<sub>`in`</sub> + `S`<sub>`in`</sub>，是元素级加法。

`expand_offset_dims` 是定义域为 [`0`, `offset.size`) 且值域为 [`0`, `operand.rank`) \ `collapsed_slice_dims` 的单调函数。所以如果，`offset.size` 是 `4`，`operand.rank` 是 `6` 且 `collapsed_slice_dims`为 {`0`, `2`} 那么 `expand_offset_dims` 则为 {`0`→`1`, `1`→`3`, `2`→`4`, `3`→`5`}。

### 非正式说明和实例

非正式情况下，输出数组中的每个索引 `Out` 对应于操作数组中的元素 `E`， 计算方法如下：

    - 我们使用 `Out` 中的批处理维度从 `start_indedices` 中查找起始索引。
    - 我们使用 `start_index_map` 将起始索引（其大小可能小于 operand.rank）映射到“完整”的起始索引到操作数。
    - 我们使用完整的起始索引动态切片大小为 `Slice_sizes` 的切片。
    - 我们通过折叠 `collapsed_slice_dims` 维度来重塑切片。因为所有折叠的切片维度都必须绑定为 1，所以这种重塑总是合法的。
    - 我们使用 `Out` 中的偏移量维度索引到此切片中，以获取与输出索引 `Out` 对应的输入元素 `E`。
    
在下面的所有示例中，`index_vector_dim` 被设置为 `start_indices.rank` - `1`，`index_vector_dim` 的更有趣的值不会从根本上改变操作，但会使可视化表示更麻烦。

为了直观地了解所有上述情况如何结合在一起，我们来看一个例子，它从一个 `[16,11]` 数组中收集 5 片形状为 `[8,6]` 的数组。切片到 `[16,11]` 数组中的位置可以表示为形状为 `S64[2]` 的索引向量，所有以 5 个位置的集合可以表示 `S64[5,2]` 数组。

集合操作的行为可以被描述为一个索引转换，采用 [`G`,`O`<sub>`0`</sub>,`O`<sub>`1`</sub>] 输出形状中的索引，并按以下方式将其映射到输入数组中的元素：

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="https://www.tensorflow.org/images/ops_xla_gather_1.svg">
</div>

我们首先使用 `G` 从聚集索引数组中选择一个 (`X`,`Y`) 向量。索引处的输出数组 [`G`,`O`<sub>`0`</sub>,`O`<sub>`1`</sub>] 中的元素是索引 [`X`+`O`<sub>`0`</sub>,`Y`+`O`<sub>`1`</sub>] 处的输入数组中的元素。

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

除了起始索引只有一个元素 `X` 之外，其他元素通常都来自聚集索引数组。类似地，只有一个输出偏移量索引的值为 `O`<sub>`0`</sub>。但是，在将它们用作输入数组的索引之前，将根据 “Gather Index Mapping”(正式描述中的 `start_index_map`)和 “Offset Mapping”（正式描述中的 `expand_offset_dims`）扩展为 [`X`,`0`] and [`0`,`O`<sub>`0`</sub>]，加起来分别为 [`X`,`O`<sub>`0`</sub>]。换句话说，输出索引为 [`G`<sub>`0`</sub>,`G`<sub>`1`</sub>,`O`<sub>`0`</sub>] 映射到输入索引 [`GatherIndices`[`G`<sub>`0`</sub>,`G`<sub>`1`</sub>,`0`],`X`]，它为我们提供了 `tf.gather_nd` 的语义。


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

| 参数      | 类型    | 语义                                              |
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

|参数               | 类型            | 语义            |
|------------------ | --------------- | ---------------------------
|`type`             | `PrimitiveType` | 类型 U           |
|`size`             | `int64`         | 张量中的元素个数。|
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

|参数     | 类型                  |语义                                       |
|------------- | --------------------- | ---------------------------------------|
|`operands`    | Sequence of N `XlaOp` | 类型为 `T_0, ..., T_N` 的 N 维数组。 |
|`init_values` | Sequence of N `XlaOp` | 类型为 `T_0, ..., T_N` 的 N 标量。|
|`computation` | `XlaComputation`      | 类型 `T_0, ..., T_N, T_0, ..., T_N -> Collate(T_0, ..., T_N)` 的计算|
|`dimensions`  | `int64` array         | 降维数量的无序数组 |

这里：

* N必须大于或等于1。
* 所有输入数组必须具有相同的维度。
* 如果 `N = 1`，`Collate(T)` 为 `T`。
* 如果 `N > 1`，`Collate(T_0, ..., T_N)` 是 `N` 类型为 `T` 元素的元组。

OP的输出是 `Collate(Q_0, ..., Q_N)`，其中 `Q_i` 是一个类型为 `T_i` 的数组，其维数如下所述。

此操作将每个输入数组的一个或多个维度降为为标量。每个返回的数组的秩是 `rank(operand) - len(dimensions)`。`init_value` 是用于每次减少的初始值，可以在后端计算过程中插入到任何位置。在大多数情况下，`init_value` 则是缩减函数的标识(例如，0表示加法)。应用的 `computation` 总是在左侧传递 `init_value`。

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

当“ `N > 1` 时，Reduce函数应用程序稍微复杂一些，因为它同时应用于所有输入。例如，考虑以下简化函数，该函数可用于并行计算一维张量的最大值和最大值：

```
f: (Float, Int, Float, Int) -> Float, Int
f(max, argmax, value, index):
  if value >= argmax:
    return (value, index)
  else:
    return (max, argmax)
```

对于一维输入数组 `V = Float[N], K = Int[N]` 和 init 值 `I_V = Float, I_K =  Int`，跨唯一输入维度缩小的结果 `f_(N-1)` 相当于以下递归程序：

```
f_0 = f(I_V, I_K, V_0, K_0)
f_1 = f(f_0.first, f_0.second, V_1, K_1)
...
f_(N-1) = f(f_(N-2).first, f_(N-2).second, V_(N-1), K_(N-1))
```

将此缩减应用于值数组和顺序索引数组（即 iota），将在数组上进行共迭代，并返回包含最大值和匹配索引的元组。

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

|参数       | 类型                   | 语义                   |
|------------ | ----------------------- | --------------------|
|`operand`    | `XlaOp` | 类型为 T 的数组|
|`dimensions` | `int64` vector          | 维度折叠的顺序|
|`new_sizes`  | `int64` vector          | 新维度大小的矢量|

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

|参数          | 类型                    | 语义                  |
|------------ | ----------------------- | ---------------------|
|`operand`    | `XlaOp`                 | 类型为 T 的数组 |
|`dimensions` | `ArraySlice<int64>`     | 待反转的维度|

反转操作是将 `operand` 数组沿指定的维度 `dimensions` 对元素的顺序反转，产生一个形状相同的数组。operand 数组的每个元素被存储在输出数组的变换后的位置上。元素的原索引位置在每个待倒置维度上都被反转了，得到其在输出数组中的索引位置（即，如果一个大小为 N 的维度是待倒置的，则索引 i 被变换为 N-i-i）。

`Rev` 操作的一个用途是在神经网络的梯度计算时沿两个窗口维度对卷积权重值进行倒置。

## RngNormal

另请参阅 [`XlaBuilder::RngNormal`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

构造给定形状的输出，按 $$N(\mu，\sigma)$$ 正态分布生成随机数。参数 $$\mu$$ 和 $$\sigma$$，以及输出形状必须具有浮点元素类型。此外，参数还必须是标量值。

<b>`RngNormal(mu, sigma, shape)`</b>

| 参数 | 类型    |语义                                           |
| --------- | ------- | --------------------------------------------------- |
| `mu`      | `XlaOp` | T 类型标量，指定生成数的平均值。 |
| `sigma`   | `XlaOp` | T 类型标量，指定生成数的标准差 |
| `shape`   | `Shape` | 输出类型形状                             |

## RngUniform

另请参阅 [`XlaBuilder::RngUniform`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

构造一个给定形状的输出，在区间 $$[a,b)$$ 上均匀分布后生成随机数。参数和输出元素类型必须是布尔类型、整型或浮点类型，而且类型必须一致。CPU 和 GPU 后端当前仅支持 F64、F32、F16、BF16、S64、U64、S32 和 U32。此外，还需要对参数进行标量赋值。如果 $$b <= a$$，则结果是由实现过程定义的。

<b>`RngUniform(a, b, shape)`</b>

| 参数 | 类型                    | 语义                         |
| --------- | ----------------------- | --------------------------------- |
| `a`       | `XlaOp`                 | T 类型标量，指定生成数的下限。 |
| `b`       | `XlaOp`                 | T 类型标量，指定生成数的上限。 |
| `shape`   | `Shape`                 | 输出类型形状             |

## Scatter

XLA Scatter操作生成一个结果，它是输入张量 `operand` 的值，有几个切片(按 `scatter_indices` 指定的索引值)使用 `update_computation` 更新为 `updates` 中的值。

另请参阅 [`XlaBuilder::Scatter`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b> `scatter(operand, scatter_indices, updates, update_computation, index_vector_dim, update_window_dims, inserted_window_dims, scatter_dims_to_operand_dims)` </b>

|参数              | 类型                   | 语义                        |
|------------------|------------------------|----------------------------------|
|`operand`         | `XlaOp`                | 将被分散到的张量。    |
|`scatter_indices` | `XlaOp`                | 包含必须分散到的切片的起始索引的张量。 |
|`updates`         | `XlaOp`                | 包含散射必须使用的值的张量。        |
|`update_computation`| `XlaComputation`     | 用于将输入张量中的现有值与散射期间的更新组合在一起的计算。此计算应为类型 `T, T -> T`。 |
|`index_vector_dim`| `int64`                | `scatter_indices` 中包含起始索引的维度。`scatter_indices` |     
|`update_window_dims`| `ArraySlice<int64>`  | `updates`  形状中的一组维度，它们是 _window dimensions_ |
|`inserted_window_dims`| `ArraySlice<int64>`| 必须插入到 `updates` 形状中的 _window dimensions_|
|`scatter_dims_to_operand_dims`| `ArraySlice<int64>`  | 维数从分散指数映射到操作数索引空间。此数组被解释为将 `i` 映射到 `scatter_dims_to_operand_dims[i]`。它必须是一对一且完全的。 |

如果 `index_vector_dim` 等于 `scatter_indices.rank` ，我们会默认 `scatter_indices` 有一个尾随的 `1` 维。

我们将  `ArraySlice<int64>` 类型的 `update_scatter_dims` 当作以升序排列的在 `updates` 而不在 `update_window_dims` 的维度元组。

scatter 的参数应遵循以下限制条件：

  - `updates` 张量秩必须为 `update_window_dims.size + scatter_indices.rank - 1`。

  - `updates` 中的维度 `i` 的边界必须符合以下条件：
      - 如果 `i` 出现在 `update_window_dims` 中（即对于某些 `k` 其等于 `update_window_dims`[`k`]）,则  `updates` 中维度 `i` 的范围在计算  `inserted_window_dims` 后必须不能超过 `oprand` 的相应界限（即 `adjusted_window_bounds`[`k`]，其中 `adjusted_window_bounds` 包含 `oprand` 的界限，而索引 `inserted_window_dims` 处的界限已被删除）。
      - If `i` is present in 如果 `i` 出现在 `update_scatter_dims` 中（即对于某些 `k` 其等于 `update_scatter_dims`[`k`]），则 `update` 中的维度 `i` 的界限必须等于 `scatter_indices` 中的相应界限，跳过 `index_vector_dim`（即 `scatter_indices.shape.dims`[`k`]，如果 `k` < `index_vector_dim`，否则d `scatter_indices.shape.dims`[`k+1`]）。
      
  - `update_window_dims` 必须按升序排列，没有任何重复的标注编号，并且必须在 `[0, updates.rank)` 范围内。

  - `inserted_window_dims` 必须按升序排列，没有任何重复的标注编号，并且必须在 `[0, operand.rank)` 范围内。

  - `scatter_dims_to_operand_dims.size` 必须等于 `scatter_indices`[`index_vector_dim`]， 且它的值在 `[0, operand.rank)` 范围内。

对于 `updates` 张量中的给定索引 `U`，必须对其应用此更新的 `oprand` 张量中的相应索引 `I` 计算如下：

  1. 使 `G` = { `U`[`k`] for `k` in `update_scatter_dims` }。用 `G` 在 `scatter_indices` 中查找索引向量比如  `S`[`i`] = `scatter_indices`[Combine(`G`, `i`)]，其中 Combine(A, b) 表示将 b 插入 A 中的 `index_vector_dim` 位置。
  2.  使用 `S` 在 `operand` 创建索引 `S`<sub>`in`</sub>，其使用 `scatter_dims_to_operand_dims` 映射分散 `S`。更准确地说：
       1. `S`<sub>`in`</sub>[`scatter_dims_to_operand_dims`[`k`]] = `S`[`k`] 如果 `k` < `scatter_dims_to_operand_dims.size`。
       2. 否则 `S`<sub>`in`</sub>[`_`] = `0`。
  3. 在 `oprand` 中创建一个索引 `W`<sub>`in`</sub>，其按 `inserted_window_dims` 将 `update_window_dims` 分散到 `U` 中。
     更准确地说：
       1. `W`<sub>`in`</sub>[`window_dims_to_operand_dims`(`k`)] = `U`[`k`] 如果 `k` < `update_window_dims.size`, 且 `window_dims_to_operand_dims` 在定义域 [`0`, `update_window_dims.size`) 和值域 [`0`, `operand.rank`) \\ `inserted_window_dims` 中为单调函数。（例如，如果 `update_window_dims.size` 是 `4`，`operand.rank` 为 `6` 且 `inserted_window_dims` 为 {`0`, `2`} 那么 `window_dims_to_operand_dims` 为 {`0`→`1`, `1`→`3`, `2`→`4`, `3`→`5`})。
       2. 否则 `W`<sub>`in`</sub>[`_`] = `0`
  4. `I` 为 `W`<sub>`in`</sub> + `S`<sub>`in`</sub> 这里 + 为元素对应相加。

总而言之，scatter 操作可以定义如下。

   - 根据  `operand` 初始化 `output`，即对于 `operand` 张量中的所有索引 `O`：\
       `output`[`O`] = `operand`[`O`]
   - 对于 `updates` 张量中的每个索引 `U` 和 `oprand` 张量中的相应索引 `O`：
       `output`[`O`] = `update_computation`(`output`[`O`], `updates`[`U`])

应用更新的顺序是不确定的。因此，当 `updates` 中的多个索引引用 `operand` 中的同一索引时，`output` 中的相应值将是不确定的。

请注意，传递到 `update_computation` 中的第一个参数将始终是 `output` 张量的当前值，而第二个参数将始终是 `updates`张量的值。这一点对于`update_computation` **不可交换性**是很重要的。

通俗来说，散布OP可以被看作是 gather 操作的一个**逆向**，即 scatter 操作更新输入中由相应的 gather 操作提取的元素。

有关详细的非正式描述和示例，请参阅 `Gather` 下的 “非正式描述” 一节。

## Select

也可参见
[`XlaBuilder::Select`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

Constructs an output array from elements of two input arrays, based on the values of a predicate array.
从两个输入数组的元素构造输出数组，此操作基于 pred 数组的值。

<b> `Select(pred, on_true, on_false)` </b>

| 参数      | 类型     | 语义              |
|---------- | ------- | ------------------ |
|`pred`     | `XlaOp` | 类型 PRED 的数组 |
|`on_true`  | `XlaOp` | 类型 T 的数组 |
|`on_false` | `XlaOp` | 类型 T 的数组 |

数组 `on_true` 和 `on_false` 必须具有相同的形状。这也是输出数组的形状。数组 `pred` 必须与 `on_true` 和 `on_false` 具有相同的维数，其元素类型为 `PRED`。

对于 `pred` 的每个元素 `P`，如果 `P` 的值是 `true`，则输出数组的相应元素取自 `on_true`；如果 `P` 的值是 `false`，则取自 `on_false`。作为受限形式的 [broadcasting](broadcasting.md)，`pred` 可以是 `PRED` 类型的标量。在本例中，如果 `pred` 是 `true`，则输出数组完全取自`on_true`；如果`pred` 是 `false`，则取自 `on_false`。

非标量 `pred` 的样例：

```
let pred: PRED[4] = {true, false, false, true};
let v1: s32[4] = {1, 2, 3, 4};
let v2: s32[4] = {100, 200, 300, 400};
==>
Select(pred, v1, v2) = s32[4]{1, 200, 300, 4};
```
标量 `pred` 的样例：

```
let pred: PRED = true;
let v1: s32[4] = {1, 2, 3, 4};
let v2: s32[4] = {100, 200, 300, 400};
==>
Select(pred, v1, v2) = s32[4]{1, 2, 3, 4};
```

支持元组之间的选择。为此，元组被视为标量类型。如果 `on_true` 和 `on_false` 是元组(必须具有相同的形状！)然后，`pred` 必须是 `PRED` 类型的标量。

## SelectAndScatter

也可参见
[`XlaBuilder::SelectAndScatter`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

此操作可被视为一种复合操作，该操作首先计算  `operand` 数组上的 `ReduceWindow`，以从每个窗口中选择一个元素，然后将 `source` 数组分散到所选元素的索引中，以构造与操作数组形状相同的输出数组。`select` 二元函数用于通过将元素应用于每个窗口从每个窗口中选择一个元素，并且调用该函数时具有这样的特性，即第一个参数的索引向量在词典上小于第二个参数的索引向量。如果选择了第一个参数，则 `select` 函数返回 `true`，如果选择了第二个参数，则返回 `false`，并且函数必须具保持传递性(即，如果 `select(a, b)` 和 `select(b, c)` 为 `true`，则 `select(a，c)`也为 `true`)，因此所选元素不依赖于为给定窗口遍历的元素的顺序。

在输出数组中的每个选定索引处应用函数 `scatter`。它需要两个标量参数：

1.  输出数组中选定索引处的当前值。
2.  应用于所选索引的  `source` 中的分散量。

它组合了这两个参数并返回一个标量值，该值用于更新输出数组中所选索引处的值。最初，输出数组的所有索引都设置为 `init_value`。

输出数组的形状与 `operand` 数组相同，而 `source` 数组的形状必须与对 `operand` 数组应用 `ReduceWindow` 操作的结果相同。`SelectAndScatter`可用于反向传播神经网络中的池化层的梯度值。

<b>`SelectAndScatter(operand, select, window_dimensions, window_strides,
padding, source, init_value, scatter)`</b>

| Arguments           | Type                | Semantics                        |
| ------------------- | ------------------- | -------------------------------- |
| `operand`           | `XlaOp`             | T 类型数组，窗口在其中滑动 |
| `select`            | `XlaComputation`    | `T, T -> PRED` 类型的二元计算，应用于每个窗口中的所有元素中；如果选择了第一个参数，则返回 `true`，如果选择第二个参数，则返回 `false`。 |
| `window_dimensions` | `ArraySlice<int64>` | 窗口维度值的整数数组   |
| `window_strides`    | `ArraySlice<int64>` | 窗口步距值的整数数组   |
| `padding`           | `Padding`           | 窗口的填充类型（Padding\:\:kSame or Padding\:\:kValid） |
| `source`            | `XlaOp`             | 具有要散布的值的 T 类型数组  |
| `init_value`        | `XlaOp`             | 输出数组初始值的类型为 T 的标量值  |
| `scatter`           | `XlaComputation`    | `T, T -> T` 类型的二元计算，以将源元素与其目标元素绑定执行每个分散操作 |

下图显示了使用 `SelectAndScatter` 的示例，其中 `select` 函数计算其参数中的最大值。请注意，当窗口重叠时，如下图 (2) 所示， `operand` 数组的索引可以由不同的窗口多次选择。在图中，值为 9 的元素由上面的两个窗口(蓝色和红色)选择，二进制相加 `scatter` 函数生成值 8(2+6) 的输出元素。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%"
    src="https://www.tensorflow.org/images/ops_scatter_to_selected_window_element.png">
</div>

`scatter` 函数的评估顺序是任意的，可能是不确定的。因此，`scatter` 功能不应对重新关联过于敏感。有关更多详细信息，请参见 [`Reduce`](#reduce) 上下文中关于关联性的讨论。

## Send

也可参见
[`XlaBuilder::Send`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b> `Send(operand, channel_handle)` </b>

参数              | 类型               | 语义
---------------- | --------------- | -----------------------------------------
`operand`        | `XlaOp`         | 发送的数据
`channel_handle` | `ChannelHandle` | 每个发送/接收配对的唯一标识符

将给定的操作数据发送到共享相同通道句柄的另一次计算中的 `Recv` 指令下。不返回任何数据。

与 `Recv` 操作类似，`Send` 操作的客户端 API 也是同步通信，并在内部分解为两个 HLO 指令(`Send` 和 `SendDone`)，以支持异步数据传输。另见[`HloInstruction：CreateSend` 和 `HloInstructions：CreateSendDone`](https://www.tensorflow.org/code/tensorflow/compiler/xla/service/hlo_instruction.h)。

<b>`Send(HloInstruction operand, int64 channel_id)`</b>

，下面的“SendDone”指令使用该上下文等待数据传输完成。上下文是{操作数(形)、请求标识符(U32)}的元组，只能由“SendDone”指令使用。
要开始一次将操作数发送到具有相同 id 的 `Recv` 指令下的异步操作。返回一个环境设置，它会在数据传输完成后被一个随后的 `SendDone` 指令所使用。这个环境配置是一个 {operand (shape), request identifier (U32)} 元组，只能由 `SendDone` 使用。

<b> `SendDone(HloInstruction context)` </b>

发送一条由 `Send` 指令创建的环境配置，等待数据传输完成。指令不返回任何数据。

<b> Scheduling of channel instructions </b>

每个通道的 4 条指令(`Recv`、`RecvDone`、`Send`、 `SendDone`)的执行顺序如下所示。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:70%" src="../../images/send_recv_order.png">
</div>

* `Recv` 在 `Send` 之前发生
* `Send` 在 `RecvDone` 之前发生
* `Recv` 在 `RecvDone` 之前发生
* `Send` 在 `SendDone` 之前发生

当后端编译器为通过通道指令进行通信的每个计算生成线性调度时，计算之间不能有数据循环。例如，以下计划会导致死锁。

<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/send_recv_schedule.png">
</div>

## Slice

也可参见
[`XlaBuilder::Slice`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

Slice 是从输入数组中提取子数组。子数组与输入数组保持相同的秩，并且包含输入数组内的边界框内的值，其中边界框的尺寸和索引作为切片操作的参数提供。

<b> `Slice(operand, start_indices, limit_indices)` </b>

| Arguments       | Type                | Semantics                            |
| --------------- | ------------------- | ------------------------------------ |
| `operand`       | `XlaOp`             | T 类型的 N 维数组         |
| `start_indices` | `ArraySlice<int64>` | 包含每个维度切片起始索引的N个整数的列表。值必须大于或等于零。    |
| `limit_indices` | `ArraySlice<int64>` | 包含每个维度切片的结尾索引(独占)的 N 个整数的列表。每个值必须大于或等于维度的相应`start_indices`，并且小于或等于维度的大小。   |

1-维样例：

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
Slice(a, {2}, {4}) produces:
  {2.0, 3.0}
```

2-维样例：

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

也可查看 [`XlaBuilder::Sort`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

有两种不同类型的 Sort 指令：单操作数和双操作数。

<b>`Sort(operand)`</b>

参数   | 类型    | 语义
----------- | ------- | --------------------
`operand`   | `XlaOp` | 需要排序的操作数。
`dimension` | `int64` | 排序所依据的维度。

按照提供的维度按升序对操作数中的元素进行排序。例如，对于一个秩为 2 的操作数(矩阵)，`dimension` 值如为 0 将对每列进行独立排序，而`dimension` 值如为 1 则将对每行进行独立排序。如果操作数的元素具有浮点类型，并且操作数包含 NaN 元素，则输出中元素的顺序由实现过程定义。

<b>`Sort(key, value)`</b>

对键和值操作数进行排序。键按单操作数版本进行排序。这些值根据其相应键的顺序进行排序。例如，如果输入是 `keys = [3, 1]` 和 `values = [42, 50]`，则排序的输出是元组  `{[1, 3], [50, 42]}`。

排序不能保证是稳定的，也就是说，如果键数组包含重复项，则可能不会保留它们对应的值的顺序。

参数   | 类型    | 语义
----------- | ------- | -------------------
`keys`      | `XlaOp` | 用于排序的键。
`values`    | `XlaOp` | 需要排序的值。
`dimension` | `int64` | 需要排序的维度。

`keys` 和 `values` 必须具有相同的维度，但可能具有不同的元素类型。

## Transpose

也可参见 `tf.reshape` 操作.

<b>`Transpose(operand)`</b>

| 参数          | 类型                | 语义                   |
|------------- | ------------------- | ------------------------------|
|`operand`     | `XlaOp`             | 需要转置的操作数。      |
|`permutation` | `ArraySlice<int64>` | 如何变更维度。         |

使用给定的置换来变更操作数维数，因此：
`∀ i . 0 ≤ i < rank ⇒ input_dimensions[permutation[i]] = output_dimensions[i]`.

这如同 Reshape(operand, permutation,
              Permute(permutation, operand.shape.dimensions)) 作用一样。

## Tuple

请参见 [`XlaBuilder::Tuple`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

包含可变数量的数据句柄的元组，每个数据句柄都有自己的形状。

这类似于 C++ 中的 `std：tuple`。概念上如同：

```
let v: f32[10] = f32[10]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
let s: s32 = 5;
let t: (f32[10], s32) = tuple(v, s);
```

可以通过 [`GetTupleElement`](#gettupleelement) 操作解构(访问)元组。

## While

请参见 [`XlaBuilder::While`](https://www.tensorflow.org/code/tensorflow/compiler/xla/client/xla_builder.h)。

<b> `While(condition, body, init)` </b>

| 参数        | 类型              | 语义                                |
| ----------- | ---------------- | ---------------------------------------- |
| `condition` | `XlaComputation` | `T -> PRED` 类型的 XlaComputation，其定义循环的终止条件 |
| `body`      | `XlaComputation` | XlaComputation of type `T -> T` 类型的XlaComputation，其定义了循环的执行内容 |
| `init`      | `T`              | 参数 `condition` 和 `body` 的初始值 |

顺序执行 `body`，直到 `condition` 失败。除了下面列出的差异和限制之外，这与许多其他语言中的 While 循环类似。

*   `While` 节点返回类型为 `T` 的值，该值是 `body` 的最后一次执行的结果。
*   类型 `T` 的形状是静态确定的，并且在所有迭代中必须是相同的。

计算的 T 参数在第一次迭代中用 `init` 值初始化，并在每次后续迭代中从 `body` 自动更新为新结果。

 `While` 节点的一个主要用例是在神经网络中实现重复执行训练。简化的伪代码如下所示，其中有一个表示计算的图。代码可在 [`while_test.cc`](https://www.tensorflow.org/code/tensorflow/compiler/xla/tests/while_test.cc)中找到。本例中的类型 `T` 是由一个用于迭代计数的 `int32` 和一个用于累加器的 `vector[10]` 组成的 `Tuple`。对于 1000 次迭代，循环不断向累加器添加一个常量向量。

```
// 计算过程的伪代码。
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
