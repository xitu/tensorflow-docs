# TensorFlow 白皮书

本文档定义有关TensorFlow的白皮书。

## 异构分布式系统上的大规模机器学习

[访问此白皮书。](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

**摘要:** TensorFlow 是表达机械学习算法的接口，也是执行这些算法的具体实现。使用 TensorFlow 表达的计算可以在各种各样的异构系统执行，从移动设备（如手机和平板）到数百台大型分布式系统以及数千个 GPU 卡等计算设备。该系统具有很强的灵活性，可用于表达各种算法，包括深度神经网络模型的训练和推理算法，已经被用于跨越数十个计算科学的领域在生产环境中进行部署并且开展研究，其中包括语音识别，计算机视觉，机器人学，信息检索，自然语言处理，地理信息提取和计算药物学。本文描述了 TensorFlow 接口以及我们在Google中构建的该接口的实现。 TensorFlow API和参考实现在2015年11月发布为Apache 2.0许可下的开源软件包，可在www.tensorflow.org上获取。

### 以 BibTeX 格式

如果您在研究中使用 TensorFlow 并希望引用 TensorFlow 系统，我们建议您引用本白皮书。

<pre>

@misc{tensorflow2015-whitepaper,

title={ {TensorFlow}: Large-Scale Machine Learning on Heterogeneous Systems},

url={https://www.tensorflow.org/},

note={Software available from tensorflow.org},

author={

    Mart\'{\i}n~Abadi and

    Ashish~Agarwal and

    Paul~Barham and

    Eugene~Brevdo and

    Zhifeng~Chen and

    Craig~Citro and

    Greg~S.~Corrado and

    Andy~Davis and

    Jeffrey~Dean and

    Matthieu~Devin and

    Sanjay~Ghemawat and

    Ian~Goodfellow and

    Andrew~Harp and

    Geoffrey~Irving and

    Michael~Isard and

    Yangqing Jia and

    Rafal~Jozefowicz and

    Lukasz~Kaiser and

    Manjunath~Kudlur and

    Josh~Levenberg and

    Dandelion~Man\'{e} and

    Rajat~Monga and

    Sherry~Moore and

    Derek~Murray and

    Chris~Olah and

    Mike~Schuster and

    Jonathon~Shlens and

    Benoit~Steiner and

    Ilya~Sutskever and

    Kunal~Talwar and

    Paul~Tucker and

    Vincent~Vanhoucke and

    Vijay~Vasudevan and

    Fernanda~Vi\'{e}gas and

    Oriol~Vinyals and

    Pete~Warden and

    Martin~Wattenberg and

    Martin~Wicke and

    Yuan~Yu and

    Xiaoqiang~Zheng},

  year={2015},

}

</pre>

或者以文本的形式:

<pre>

Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo,

Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis,

Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow,

Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia,

Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster,

Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens,

Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker,

Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas,

Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke,

Yuan Yu, and Xiaoqiang Zheng.

TensorFlow: Large-scale machine learning on heterogeneous systems,

2015. Software available from tensorflow.org.
</pre>



## TensorFlow: 大规模机器学习系统

[访问此白皮书](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

**概述:** TensorFlow 用来运行于大规模系统以及异构环境的机器学习系统。TensorFlow 用数据流图来表示计算过程，共享其状态以及改变改状态的操作。它将数据流图的节点映射到集群中的许多机器上，并跨越多个计算设备（包括多核CPU，通用GPU和定制设计的称为Tensor Processing Units（TPU）的ASIC）。 这种架构给应用开发人员充分的灵活性: 在此之前的“参数服务器”设计中，共享状态管理内置于系统中， TensorFlow 允许开发者尝试新颖的优化和训练算法。 TensorFlow 支持各种应用程序，重点是深度神经网络的训练和演算。 Google 的许多服务已经在生产环境中使用了 TensorFlow ，我们已经将其作为开源项目发布，并且已经被用于广泛的用于机器学习研究。在本文中，我们描述了 TensorFlow 数据流模型，并展示了 TensorFlow 多个实际应用中实现的令人惊叹的性能。
