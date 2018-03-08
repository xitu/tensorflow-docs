
# 迁移至 TensorFlow 1.0


TensorFlow 1.0 中 API 的改动不再完全向后兼容。因此，运行于 TensorFlow 0.n 版本的 TensorFlow 应用可能不能在 TensorFlow 1.0 版本中正常运行。在此版本中，我们对 API 进行了一些修改，确保了其内部一致性；在接下来的整个 1.N 版本周期中都不会进行任何断代式变更。

本指南将引导您了解新版 API 的主要变更，以及如何将您的程序自动升级至 TensorFlow 1.0。除了帮助您完成程序的修改之外，本指南也解释了我们为何要做出这些变更。 

## 如何升级

如果您希望自动将代码迁移至 1.0 版本，可以尝试使用我们的 `tf_upgrade.py` 脚本。此脚本能处理大多数情况，但有时还是需要您进行手动修改。
  您可以在我们的 [GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/compatibility) 中获取此脚本。

如要将单个的 0.n 版 TensorFlow 源文件转换为 1.0 版本，请输入如下格式的命令：

<pre>
$ <b>python tf_upgrade.py --infile</b> <i>InputFile</i> <b>--outfile</b> <i>OutputFile</i>
</pre>

例如，以下命令会将一个名为 `test.py` 的 0.n 版 TensorFlow 程序转换为名为 `test_1.0.py` 的 1.0 版 TensorFlow 程序：

<pre>
$ <b>python tf_upgrade.py --infile test.py --outfile test_1.0.py</b>
</pre>

`tf_upgrade.py` 脚本还会生成一个名为 `report.txt` 的文件，记录了其在升级过程中做的所有修改，并给出了一些可能需要您手动修改的建议。

如要将整个目录的 0.n 版 TensorFlow 程序升级为 1.0 版本，请输入如下格式的命令：

<pre>
$ <b>python tf_upgrade.py --intree</b> <i>InputDir</i> <b>--outtree</b> <i>OutputDir</i>
</pre>

例如，以下命令会将 `/home/user/cool` 中的所有 0.n 版 TensorFlow 程序转换为 1.0 版并放入新建的 `/home/user/cool_1.0` 目录中：

<pre>
$ <b>python tf_upgrade.py --intree /home/user/cool --outtree /home/user/cool_1.0</b>
</pre>

### 限制

在使用脚本进行升级时，有几点注意事项。尤其是：

 * 你需要手动修复所有 `tf.reverse()` 实例。
   `tf_upgrade.py` 脚本也会在屏幕输出以及 `report.txt` 文件中警告您关于 `tf.reverse()` 的信息。
 * 如果遇上一些需要重排序的参数，`tf_upgrade.py` 将会试着最小化地格式化您的代码，但不能自动地改变实际的参数顺序。因此 `tf_upgrade.py` 将使用关键字参数，让函数参数与顺序无关。
 * `tf.get_variable_scope().reuse_variables()` 之类的构造器将失效。我们建议删除它们用以下方法代替：

   <pre class="prettyprint">
   with tf.variable_scope(tf.get_variable_scope(), reuse=True):
     ...
   </pre>

 * 与 `tf.pack` 和  `tf.unpack` 类似，我们将 `TensorArray.pack` 以及 `TensorArray.unpack` 重命名为 `TensorArray.stack` 和 `TensorArray.unstack`。但是，`TensorArray.pack` 和 `TensorArray.unpack` 并不直接关联 `tf` 命名空间，因而无法通过词法直接检测出来，例如 `foo = tf.TensorArray(); foo.unpack()`。因此需要手动修改它们。

## 手动升级您的代码

您也可以不使用 `tf_upgrade.py`，手动升级代码。本文档剩余部分提供了完整的 TensorFlow 1.0 非向后兼容变更列表。


### 变量（Variables）

现在 Variable 函数更具一致性，减少了误解。

* `tf.VARIABLES`
    * 需要重命名为 `tf.GLOBAL_VARIABLES`
* `tf.all_variables`
    * 需要重命名为 `tf.global_variables`
* `tf.initialize_all_variables`
    * 需要重命名为 `tf.global_variables_initializer`
* `tf.initialize_local_variables`
    * 需要重命名为 `tf.local_variables_initializer`
* `tf.initialize_variables`
    * 需要重命名为 `tf.variables_initializer`

### 聚合函数

现在所有的聚合函数（Summary function）都被统一放置于 `tf.summary` 命名空间中。

* `tf.audio_summary`
    * 需要重命名为 `tf.summary.audio`
* `tf.contrib.deprecated.histogram_summary`
    * 需要重命名为 `tf.summary.histogram`
* `tf.contrib.deprecated.scalar_summary`
    * 需要重命名为 `tf.summary.scalar`
* `tf.histogram_summary`
    * 需要重命名为 `tf.summary.histogram`
* `tf.image_summary`
    * 需要重命名为 `tf.summary.image`
* `tf.merge_all_summaries`
    * 需要重命名为 `tf.summary.merge_all`
* `tf.merge_summary`
    * 需要重命名为 `tf.summary.merge`
* `tf.scalar_summary`
    * 需要重命名为 `tf.summary.scalar`
* `tf.train.SummaryWriter`
    * 需要重命名为 `tf.summary.FileWriter`

### 数值差异


整数除法以及 `tf.floordiv` 将使用向下取整（floor）语义。这样就能使 `np.divide` 和 `np.mod` 的结果与 `tf.divide` 和 `tf.mod` 的结果保持一致。另外，我们修改了 `tf.round` 使用的取整算法，使其与 NumPy 保持一致。


* `tf.div`

    * 除法 `tf.divide` 的语义现在已经修改与 Python 语义保持一致，即 Python 3 中的 `/` 符号以及 Python 2 future 模块的 division 将始终得到浮点数、`//` 将进行求整除法。此外，`tf.div` 将只进行求整除法。如需使用 C 语言强制截断风格的除法运算，可以使用 `tf.truncatediv`。

    * 请尽量将你的代码 `tf.div` 改为 `tf.divide`，它将遵循 Python 的语义。

* `tf.mod`

    * 求余 `tf.mod` 的语义现在已经修改与 Python 语义保持一致。另外，对于整数的运算将使用向下取整（floor）语义。如需使用 C 语言强制截断风格的求余运算，可以使用 `tf.truncatemod`。


新版和旧版的除法操作对比总结如下表所示：

| 表达式                | TF 0.11 (py2) | TF 0.11 (py3) | TF 1.0 (py2) | TF 1.0 (py3) |
|---------------------|---------------|---------------|--------------|--------------|
| tf.div(3,4)         | 0             | 0             | 0            | 0            |
| tf.div(-3,4)        | 0             | 0             | -1           | -1           |
| tf.mod(-3,4)        | -3            | -3            | 1            | 1            |
| -3/4                | 0             | -0.75         | -1           | -0.75        |
| -3/4tf.divide(-3,4) | N/A           | N/A           | -0.75        | -1           |

新版和旧版的取整操作对比总结如下表所示：

| 输入 | Python | NumPy | C++ round() | TensorFlow 0.11(floor(x+.5)) | TensorFlow 1.0 |
|-------|--------|-------|-------------|------------------------------|----------------|
| -3.5  | -4     | -4    | -4          | -3                           | -4             |
| -2.5  | -2     | -2    | -3          | -2                           | -2             |
| -1.5  | -2     | -2    | -2          | -1                           | -2             |
| -0.5  | 0      | 0     | -1          | 0                            | 0              |
| 0.5   | 0      | 0     | 1           | 1                            | 0              |
| 1.5   | 2      | 2     | 2           | 2                            | 2              |
| 2.5   | 2      | 2     | 3           | 3                            | 2              |
| 3.5   | 4      | 4     | 4           | 4                            | 4              |



### 匹配 NumPy 命名


新版本对许多函数进行了重命名以匹配 NumPy。这么做旨在使得 NumPy 与 TensorFlow 之间的转换尽量简便。虽然我们已经排除了一些常见的不一致情况，但现在还有一些函数未能完全匹配。

* `tf.inv`
    * 需要重命名为 `tf.reciprocal`
    * 这么做是为了防止其与 NumPy 的矩阵求逆函数 `np.inv` 混淆
* `tf.list_diff`
    * 需要重命名为 `tf.setdiff1d`
* `tf.listdiff`
    * 需要重命名为 `tf.setdiff1d`
* `tf.mul`
    * 需要重命名为 `tf.multiply`
* `tf.neg`
    * 需要重命名为 `tf.negative`
* `tf.select`
    * 需要重命名为 `tf.where`
    * `tf.where` 现在与 `np.where` 一样，需要传入 3 个或 1 个参数
* `tf.sub`
    * 需要重命名为 `tf.subtract`

### 匹配 NumPy 参数

一些 TensorFlow 1.0 方法的参数现在与 NumPy 的方法相匹配了。为了实现这一点，TensorFlow 1.0 对一些关键字参数进行了修改，并对一些参数进行了重排序。需要注意的是，TensorFlow 1.0 现在不再使用 `dimension` 而转为使用 `axis`。TensorFlow 1.0 在修改张量的操作中将保持张量参数始终在第一位。（参见 `tf.concat` 的改动）。


* `tf.argmax`
    * 关键字参数 `dimension` 需要重命名为 `axis`
* `tf.argmin`
    * 关键字参数 `dimension` 需要重命名为 `axis`
* `tf.concat`
    * 关键字参数 `concat_dim` 需要重命名为 `axis`
    * 输入参数重排序为 `tf.concat(values, axis, name='concat')`.
* `tf.count_nonzero`
    * 关键字参数 `reduction_indices` 需要重命名为 `axis`
* `tf.expand_dims`
    * 关键字参数 `dim` 需要重命名为 `axis`
* `tf.reduce_all`
    * 关键字参数 `reduction_indices` 需要重命名为 `axis`
* `tf.reduce_any`
    * 关键字参数 `reduction_indices` 需要重命名为 `axis`
* `tf.reduce_join`
    * 关键字参数 `reduction_indices` 需要重命名为 `axis`
* `tf.reduce_logsumexp`
    * 关键字参数 `reduction_indices` 需要重命名为 `axis`
* `tf.reduce_max`
    * 关键字参数 `reduction_indices` 需要重命名为 `axis`
* `tf.reduce_mean`
    * 关键字参数 `reduction_indices` 需要重命名为 `axis`
* `tf.reduce_min`
    * 关键字参数 `reduction_indices` 需要重命名为 `axis`
* `tf.reduce_prod`
    * 关键字参数 `reduction_indices` 需要重命名为 `axis`
* `tf.reduce_sum`
    * 关键字参数 `reduction_indices` 需要重命名为 `axis`
* `tf.reverse`
    * `tf.reverse` 之前需要传入 1 维 `bool` 型张量用以控制维度的顺序调换，现在使用一组轴的索引进行控制。
    * 例如 `tf.reverse(a, [True, False, True])` 现在需改为 `tf.reverse(a, [0, 2])`
* `tf.reverse_sequence`
    * 关键字参数 `batch_dim` 需要重命名为 `batch_axis`
    * 关键字参数 `seq_dim` 需要重命名为 `seq_axis`
* `tf.sparse_concat`
    * 关键字参数 `concat_dim` 需要重命名为 `axis`
* `tf.sparse_reduce_sum`
    * 关键字参数 `reduction_axes` 需要重命名为 `axis`
* `tf.sparse_reduce_sum_sparse`
    * 关键字参数 `reduction_axes` 需要重命名为 `axis`
* `tf.sparse_split`
    * 关键字参数 `split_dim` 需要重命名为 `axis`
    * 输入参数重排序为 `tf.sparse_split(keyword_required=KeywordRequired(), sp_input=None, num_split=None, axis=None, name=None, split_dim=None)`.
* `tf.split`
    * 关键字参数 `split_dim` 需要重命名为 `axis`
    * 关键字参数 `num_split` 需要重命名为 `num_or_size_splits`
    * 输入参数重排序为 `tf.split(value, num_or_size_splits, axis=0, num=None, name='split')`.
* `tf.squeeze`
    * 关键字参数 `squeeze_dims` 需要重命名为 `axis`
* `tf.svd`
    * 输入参数重排序为 `tf.svd(tensor, full_matrices=False, compute_uv=True, name=None)`.

### 简化数学变换

批量版数学运算操作已被移除。现在非批量版的函数已经包含了批量运算的功能。例如，`tf.complex_abs` 的功能已迁移至 `tf.abs`

* `tf.batch_band_part`
    * 需要重命名为 `tf.band_part`
* `tf.batch_cholesky`
    * 需要重命名为 `tf.cholesky`
* `tf.batch_cholesky_solve`
    * 需要重命名为 `tf.cholesky_solve`
* `tf.batch_fft`
    * 需要重命名为 `tf.fft`
* `tf.batch_fft3d`
    * 需要重命名为 `tf.fft3d`
* `tf.batch_ifft`
    * 需要重命名为 `tf.ifft`
* `tf.batch_ifft2d`
    * 需要重命名为 `tf.ifft2d`
* `tf.batch_ifft3d`
    * 需要重命名为 `tf.ifft3d`
* `tf.batch_matmul`
    * 需要重命名为 `tf.matmul`
* `tf.batch_matrix_determinant`
    * 需要重命名为 `tf.matrix_determinant`
* `tf.batch_matrix_diag`
    * 需要重命名为 `tf.matrix_diag`
* `tf.batch_matrix_inverse`
    * 需要重命名为 `tf.matrix_inverse`
* `tf.batch_matrix_solve`
    * 需要重命名为 `tf.matrix_solve`
* `tf.batch_matrix_solve_ls`
    * 需要重命名为 `tf.matrix_solve_ls`
* `tf.batch_matrix_transpose`
    * 需要重命名为 `tf.matrix_transpose`
* `tf.batch_matrix_triangular_solve`
    * 需要重命名为 `tf.matrix_triangular_solve`
* `tf.batch_self_adjoint_eig`
    * 需要重命名为 `tf.self_adjoint_eig`
* `tf.batch_self_adjoint_eigvals`
    * 需要重命名为 `tf.self_adjoint_eigvals`
* `tf.batch_set_diag`
    * 需要重命名为 `tf.set_diag`
* `tf.batch_svd`
    * 需要重命名为 `tf.svd`
* `tf.complex_abs`
    * 需要重命名为 `tf.abs`

### 其它改动

除上文所述的改动外，还有以下一些变化：

* `tf.image.per_image_whitening`
    * 需要重命名为 `tf.image.per_image_standardization`
* `tf.nn.sigmoid_cross_entropy_with_logits`
    * 输入参数重排序为 `tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)`.
* `tf.nn.softmax_cross_entropy_with_logits`
    * 输入参数重排序为 `tf.nn.softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, dim=-1, name=None)`.
* `tf.nn.sparse_softmax_cross_entropy_with_logits`
    * 输入参数重排序为 `tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)`.
* `tf.ones_initializer`
    * 需要修改成函数调用，例如  `tf.ones_initializer()`
* `tf.pack`
    * 需要重命名为 `tf.stack`
* `tf.round`
    * `tf.round` 的语义现在与银行家舍入法（Banker's rounding）相同。
* `tf.unpack`
    * 需要重命名为 `tf.unstack`
* `tf.zeros_initializer`
    * 需要修改成函数调用，例如 `tf.zeros_initializer()`
