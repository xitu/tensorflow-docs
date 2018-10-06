# Estimator

这篇文档介绍的 `tf.estimator` 是一个高级的 TensorFlow API，它极大的简化了机器学习编程工作。Estimator 封装了如下行为：

* 训练
* 评估
* 预测
* 作为服务导出

你可以使用我们预定义好的 Estimator，也可以自己写一个 Estimator。但无论预定义的还是自定义的 Estimator 都继承自 `tf.estimator.Estimator` 类。

想要快速示例（quick example），请看 [Estimator 教程]](../tutorials/estimators/linear)。想深入查看每个子主题，请查看 [Estimator 指南](premade_estimators)。

注意：TensorFlow 还包含了一个已经被弃用的 `Estimator` 类 `tf.contrib.learn.Estimator`，我们不要去使用它。

## 使用 Estimator 的优势

Estimator 可以为我们带来以下几点好处：

* 基于 Estimator 的模型可以运行在单机上，也可以运行在分布式的多台服务器环境上并且不需要做任何修改。更棒的是，你还可以将基于 Estimator 的模型运行在 CPUs，GPUs 或者 TPUs 上。
* Estimator 简化了模型开发人员内部共享的实现。
* 你可以用高级的直观的代码编写某种状态的艺术模型。简而言之，使用 Estimator 通常会比使用 TensorFlow 的低级 API 更加简捷。
* Estimator 是建立在 `tf.keras.layers` 上的，简化了自定义的内容。
* Estimator 已经为你构建了图表。
* Estimator 提供了一个安全的分布式训练的循环，能够控制以下操作的运行时间和运行方式：
  * 创建图表
  * 初始化变量
  * 加载数据
  * 处理异常
  * 创建校验文件和错误恢复
  * 储存给 TensorBoard 展示的数据

当你用 Estimator写应用时，你必须将数据输入管道和模型分开。这种分离简化了不同数据集的实验。

## 预定义的 Estimator

比起 TensorFlow 的低级 API，预定义的 Estimator 可以让你在更高抽象的层面上工作。你不再需要操心的创建计算图和会话，因为 Estimator 已经帮你把这一切都`串通`好了。也就是说，预定义的 Estimator 已经帮你创建和管理 `tf.Graph` 和 `tf.Session` 对象。甚至，预定义的 Estimator 可以让你修改最少的代码来试验不同的模型架构。譬如 `tf.estimator.DNNClassifier` 就是一个预定义的 Estimator，它可以训练密集的前向传递神经网络分类模型。

### 预定义 Estimator 的程序结构

基于预定义 Estimator 的程序一般包含下面四步：

1. **编写一个或多个数据集的导入函数。**举个例子，你可能会创建两个函数，一个用于导入训练数据，另一个用于导入测试数据。每一个数据集的导入函数都会返回下面两个对象：
    
    * 一个字典，它的 key 是特征名，而 values 是对应的张量（或者是稀疏张量），张量里面包含了对应的特征数据。
    * 一个张量，它包含了一个或多个标签。
    
    举个例子，下面的代码是一个输入函数的基本框架：
    
    ```python
    def input_fn(dataset):
    	... # 操作数据集，提取特征字典和标签
    	return feature_dict, label
    ```
    
    更多的细节，请看[导入数据](../guide/datasets.md)。

2. **定义特征列。**每一个 `tf.feature_column` 定义了特征的名字、类型或者各种输入预处理函数。举个例子，下面的代码片段创建了三个特征列，它们的类型是整形或者浮点型。前面两个特征列简单的标识了它们的名称和类型。第三个特征则定义了一个 lambda 表达式来对原始数据做转换：
	
   ```python
   # 定义三个数值类型的特征列
   population = tf.feature_column.numeric_column('population')
   crime_rate = tf.feature_column.numeric_column('crime_rate')
   median_education = tf.feature_column.numeric_column('median_education',
                      normalizer_fn=lambda x: x - global_education_mean)
   ```
	
3. **实例化相关的预定义 Estimator。**举个例子，下面有一个 `LinearClassifier` Estimator 的实例化的代码：
	
   ```python
   estimator = tf.estimator.LinearClassifier(
     feature_columns=[population, crime_rate, median_education],
   )
   ```

4. **调用训练，评估和推断的方法。**
	譬如说，所有的 Estimator 都提供了 `train` 方法，它可以用来训练模型。
	
   ```python
   # my_training_set 是在第一步中创建的函数
   estimator.train(input_fn=my_training_set, steps=2000)
   ```

### 预定义 Estimator 的好处

预定义 Estimator 是编码的最佳实践，它有下面两点好处：

* 单机或者集群上运行时，计算图的哪部分应该在哪里运行和其实现策略的最佳实践。
* 事件记录和通用内容摘要的最佳实践。
    
如果你不使用预定义的 Estimator，那么你需要自己实现上面所说到的功能。

## 自定义 Estimator

预定义和自定义 Estimator 的核心是**模型函数**， 它可以用来构建训练、评价和预测的图表。当你使用预定义 Estimator 时，里面已经实现了模型函数了。但是当你要使用自定义 Estimator 时，你就要自己编写模型函数。[Companion document](../guide/custom_estimators.md) 描述了编写模型函数的方法。

## 推荐的工作流

我们推荐的工作流如下：

1. 假设存在一个合适的 Estimator，使用它来构建你的第一个模型，并以这个模型的结果作为基准。
2. 使用当前的预定义 Estimator 构建、测试你所有的管道，包括数据的完整性和可靠性。
3. 如果存在可选的预定义 Estimator，可以对这几个 Estimator 做实验，从中选择一个能够产生最好结果的 Estimator。
4. 或许，可以通过构建你自己的 Estimator 来进一步提升模型的效果。

## 从 Keras 模块中创建 Estimator

你可以将 Keras 模型转换成 Estimator。这样 Keras 模型就可以利用到 Estimator 的优点了，譬如分布式训练。可以如下例所示调用 `tf.keras.estimator.model_to_estimator`：

```python
# 实例化一个 kera inception v3 模型。
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# 定义好用来训练模型使用的优化器，损失和评价指标，然后再编译它
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# 从已编译的 Keras 模型中创建一个 Estimator，注意，keras 模型的初始状态会被保存在这个 Estimator 中。
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
# 像处理其他 Estimator 一样处理派生 Estimator。
# 首先，恢复 Keras 模型的输入名称，这样就可以将它们当做 Estimator 数去函数的特征列名：
keras_inception_v3.input_names  # print out: ['input_1']
# 一旦有了输入名称，就可以创建输入函数，例如，对于 numpy ndarray 格式的输入：
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)
# 需要进行训练时，调用 Estimator 的训练函数：
est_inception_v3.train(input_fn=train_input_fn, steps=2000)
```

注意，Keras Estimator 的特征列和标签的名称来自相应的已编译 Keras 模型。例如上面的 `train_input_fn` 的输入键值可以从 `keras_inception_v3.input_names` 获取，类似地，预测的输出名称可以从 `keras_inception_v3.output_names` 获得。

想要了解更多的细节，请查阅 `tf.keras.estimator.model_to_estimator`。
