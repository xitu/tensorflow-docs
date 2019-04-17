<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

> [TensorFlow Docs](https://github.com/xitu/tensorflow-docs) 是由[掘金翻译计划](https://github.com/xitu/gold-miner)实时维护的 TensorFlow 官方文档中文版，维护者为全球各大公司开发人员和各著名高校研究者及学生。欢迎大家加入维护团队，欢迎提 Issue 和 PR，参与之前请阅读[文档维护说明](https://github.com/xitu/tensorflow-docs/wiki#%E6%96%87%E6%A1%A3%E7%BB%B4%E6%8A%A4%E8%A7%84%E8%8C%83)。
>
> - 阅读文档请到 👉 https://tensorflow.juejin.im
> - 推荐学习顺序等更多内容详见：[TensorFlow Docs WIKI](https://github.com/xitu/tensorflow-docs/wiki)
> - 相关术语表：[TensorFlow 术语表](https://github.com/xitu/tensorflow-docs/wiki/TensorFlow-%E6%9C%AF%E8%AF%AD%E8%A1%A8)，[人工智能术语表](https://github.com/xitu/tensorflow-docs/wiki#%E6%9C%AF%E8%AF%AD%E8%A1%A8)
> - 掘金翻译计划欢迎大家的加入，详见 👉 [加入我们](https://github.com/xitu/gold-miner)

---

| **`Documentation`** |
|-----------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) |

**TensorFlow** 是一个使用数据流图进行数值计算开源软件库。图的节点表示数学运算，节点之间的边表示流动的多维数据数组（张量）。这种灵活的架构使你能在无需重写代码的情况下，将计算在桌面端、服务端或移动端部署到一个或多个 CPU 和 GPU 中。TensorFlow 还包含 [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard)，它是一个数据可视化工具包。

TensorFlow 最初由 Google 机器智能研究机构内的 Google Brain 团队的研究人员和工程师开发，用于进行机器学习和深度神经网络研究。此系统一般足以适用于各种其他领域。

TensorFlow 提供了稳定的 Python API 和 C 语言 API，以及没有向后兼容性保证的如 C++、Go、Java、JavaScript 和 Swift 等 API。

你可以通过订阅 [announce@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/announce) 来及时获得 TensorFlow 最新的公告及更新等信息。

## 安装

**在 [安装 TensorFlow](https://www.tensorflow.org/install) 页面中查看关于稳定二进制版的安装或从源码安装的安装步骤。**

喜欢挑战的人也可以尝试我们的开发版：

**开发版 pip 包**
* 我们非常高兴发布 TensorFlow 的开发版，现在 pypi 提供开发版的 pip 包 [tf-nightly](https://pypi.python.org/pypi/tf-nightly) 和
  [tf-nightly-gpu](https://pypi.python.org/pypi/tf-nightly-gpu) 项目。在干净的环境中简单运行 `pip install tf-nightly` 或 `pip install tf-nightly-gpu` 即可安装 TensorFlow 开发版。 我们为 Linux、Mac 和 Windows 提供  CPU 和 GPU 支持。

#### 开启你的第一个 TensorFlow 程序

```shell
$ python
```
```python
>>> import tensorflow as tf
>>> tf.enable_eager_execution()
>>> tf.add(1, 2)
3
>>> hello = tf.constant('Hello, TensorFlow!')
>>> hello.numpy()
'Hello, TensorFlow!'
```

在 [tensorflow.org 的教程页面](https://www.tensorflow.org/tutorials/)中了解更多有关如何在 TensorFlow 中执行特定任务的示例吧。

## 贡献指南

**如果你想参与贡献 TensorFlow，请先查看我们的 [贡献指南](CONTRIBUTING.md)。此项目遵循 TensorFlow
[项目规范](CODE_OF_CONDUCT.md)。我们期望你能遵循此规范。**

**我们还使用 [GitHub issues](https://github.com/tensorflow/tensorflow/issues) 来跟进 requests 和 bugs。对于一般性问题和讨论请查看 
[TensorFlow 讨论](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss)，或直接在 [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow) 提问。**

TensorFlow 项目致力于遵守开源软件开发中普遍接受的最佳实践：

[![CII 最佳实践](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)

## 持续构建状态

### 官方构建

| Build Type      | Status | Artifacts |
| ---             | ---    | ---       |
| **Linux CPU**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-cc.html) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Linux GPU**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-gpu-py3.html) | [pypi](https://pypi.org/project/tf-nightly-gpu/) |
| **Linux XLA**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/ubuntu-xla.html) | TBA |
| **MacOS**       | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/macos-py2-cc.html) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Windows CPU** | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-cpu.html) | [pypi](https://pypi.org/project/tf-nightly/) |
| **Windows GPU** | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/windows-gpu.html) | [pypi](https://pypi.org/project/tf-nightly-gpu/) |
| **Android**     | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/android.html) | [![Download](https://api.bintray.com/packages/google/tensorflow/tensorflow/images/download.svg)](https://bintray.com/google/tensorflow/tensorflow/_latestVersion) |
| **Raspberry Pi 0 and 1** | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py2.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py2.html) [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi01-py3.html) | [Py2](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp27-none-linux_armv6l.whl) [Py3](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv6l.whl) |
| **Raspberry Pi 2 and 3** | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py2.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py2.html) [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/rpi23-py3.html) | [Py2](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp27-none-linux_armv7l.whl) [Py3](https://storage.googleapis.com/tensorflow-nightly/tensorflow-1.10.0-cp34-none-linux_armv7l.whl) |

### 社区支持下的构建

| Build Type      | Status | Artifacts |
| ---             | ---    | ---       |
| **IBM s390x**       | [![Build Status](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/badge/icon)](http://ibmz-ci.osuosl.org/job/TensorFlow_IBMZ_CI/) | TBA |
| **IBM ppc64le CPU** | [![Build Status](http://powerci.osuosl.org/job/TensorFlow_Ubuntu_16.04_CPU/badge/icon)](http://powerci.osuosl.org/job/TensorFlow_Ubuntu_16.04_CPU/) | TBA |
| **IBM ppc64le GPU** | [![Build Status](http://powerci.osuosl.org/job/TensorFlow_Ubuntu_16.04_PPC64LE_GPU/badge/icon)](http://powerci.osuosl.org/job/TensorFlow_Ubuntu_16.04_PPC64LE_GPU/) | TBA |
| **Linux CPU with Intel® MKL-DNN** Nightly | [![Build Status](https://tensorflow-ci.intel.com/job/tensorflow-mkl-linux-cpu/badge/icon)](https://tensorflow-ci.intel.com/job/tensorflow-mkl-linux-cpu/) | [Nightly](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-whl-nightly/) |
| **Linux CPU with Intel® MKL-DNN** Python 2.7<br> **Linux CPU with Intel® MKL-DNN** Python 3.5<br>  **Linux CPU with Intel® MKL-DNN** Python 3.6 | [![Build Status](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-release-whl/badge/icon)](https://tensorflow-ci.intel.com/job/tensorflow-mkl-build-release-whl/lastStableBuild)|[1.10.0 py2.7](https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-1.10.0-cp27-cp27mu-linux_x86_64.whl)<br>[1.10.0 py3.5](https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-1.10.0-cp35-cp35m-linux_x86_64.whl)<br>[1.10.0 py3.6](https://storage.googleapis.com/intel-optimized-tensorflow/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl) |

## 更多信息

* [TensorFlow 网站](https://www.tensorflow.org)
* [TensorFlow 教程](https://www.tensorflow.org/tutorials/)
* [TensorFlow 模型](https://github.com/tensorflow/models)
* [TensorFlow Twitter](https://twitter.com/tensorflow)
* [TensorFlow Blog](https://medium.com/tensorflow)
* [TensorFlow Course at Stanford](https://web.stanford.edu/class/cs20si)
* [TensorFlow Roadmap](https://www.tensorflow.org/community/roadmap)
* [TensorFlow White Papers](https://www.tensorflow.org/about/bib)
* [TensorFlow YouTube Channel](https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ)

你可以在 [tensorflow.org 社区页](https://www.tensorflow.org/community) 了解更多关于参与 TensorFlow 社区的方法。

## 文档管理团队

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
| [<img src="https://avatars0.githubusercontent.com/u/26959437?s=460&v=4" width="100px;"/><br /><sub>LeviDing</sub>](https://github.com/leviding)<br />[👀](#review-leviding "Reviewed Pull Requests") [🌍](#translation-leviding "Translation") [📋](#eventOrganizing-leviding "Event Organizing") | [<img src="https://avatars0.githubusercontent.com/u/4813445?s=460&v=4" width="100px;"/><br /><sub>pkuwwt</sub>](https://github.com/pkuwwt)<br />[👀](#review-pkuwwt "Reviewed Pull Requests") [🌍](#translation-pkuwwt "Translation") | [<img src="https://avatars1.githubusercontent.com/u/6165782?s=460&v=4" width="100px;"/><br /><sub>John Jiang</sub>](https://github.com/JohnJiangLA)<br />[👀](#review-JohnJiangLA "Reviewed Pull Requests") [🌍](#translation-JohnJiangLA "Translation") | [<img src="https://avatars2.githubusercontent.com/u/5164225?s=460&v=4" width="100px;"/><br /><sub>lsvih</sub>](https://github.com/lsvih)<br />[👀](#review-lsvih "Reviewed Pull Requests") [🌍](#translation-lsvih "Translation") | [<img src="https://avatars3.githubusercontent.com/u/9419075?s=460&v=4" width="100px;"/><br /><sub>foxxnuaa</sub>](https://github.com/foxxnuaa)<br />[👀](#review-foxxnuaa "Reviewed Pull Requests") [🌍](#translation-foxxnuaa "Translation") | [<img src="https://avatars0.githubusercontent.com/u/5498964?s=460&v=4" width="100px;"/><br /><sub>changkun</sub>](https://github.com/changkun)<br />[👀](#review-changkun "Reviewed Pull Requests") [🌍](#translation-changkun "Translation") |
|:-:|:-:|:-:|:-:|:-:|:-:|
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Co-Translators

[所有译者详细信息](https://github.com/xitu/tensorflow-docs/graphs/contributors)

## 文档维护支持

[<img src="https://user-images.githubusercontent.com/26959437/37653530-37bd3cde-2c7a-11e8-98d0-749a59194c22.png" width="200px;"/>](https://juejin.im)

<!--

<a href="https://coding.net/?utm_source=javascript-tutorial-zh&utm_medium=banner&utm_campaign=march2019" target="_blank"><img src="https://user-images.githubusercontent.com/26959437/56273145-c56aa000-612e-11e9-9137-a1388ef18cf2.png" width="200px;" target="_blank"/></a>

[<img src="https://user-images.githubusercontent.com/26959437/37953025-3fa103ae-31d4-11e8-9e55-136b05d7cb96.jpg" width="200px;"/>](https://jizhi.im/index)<br />了解更多人工智能知识请前往[景略集智](https://jizhi.im/index)，想从零开始学习人工智能请前往[景略集智 AI 课堂](https://h5.youzan.com/v2/showcase/homepage?alias=U5eAeeuRD2)。

每月 27 号为合作周期截止日，七月开始未提供物资。

-->

## 许可

文档正在完善中，未经允许禁止任何形式的转载。

[Apache 许可 2.0](LICENSE)
