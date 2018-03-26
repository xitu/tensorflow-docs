<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

> [TensorFlow Docs](https://github.com/xitu/tensorflow-docs) 是由[掘金翻译计划](https://github.com/xitu/gold-miner)实时维护的 TensorFlow 官方文档中文版，维护者为全球各大公司开发人员和各著名高校研究者及学生。欢迎大家加入维护团队，欢迎提 Issue 和 PR，参与之前请阅读[文档维护说明](https://github.com/xitu/tensorflow-docs/wiki#%E6%96%87%E6%A1%A3%E7%BB%B4%E6%8A%A4%E8%A7%84%E8%8C%83)。
>
> - 译者团队正在向 [TensorFlow V1.7 官方中文文档](https://github.com/xitu/tensorflow-docs)更新
> - 推荐学习顺序等更多内容详见：[TensorFlow Docs WIKI](https://github.com/xitu/tensorflow-docs/wiki)
> - 相关术语表：[TensorFlow 术语表](https://github.com/xitu/tensorflow-docs/wiki/TensorFlow-%E6%9C%AF%E8%AF%AD%E8%A1%A8)，[人工智能术语表](https://github.com/xitu/tensorflow-docs/wiki#%E6%9C%AF%E8%AF%AD%E8%A1%A8)
> - 掘金翻译计划欢迎大家的加入，详见 👉 [加入我们](https://github.com/xitu/gold-miner)

---

| **`Documentation`** | **`Linux CPU`** | **`Linux GPU`** | **`Mac OS CPU`** | **`Windows CPU`** | **`Android`** |
|-----------------|---------------------|------------------|-------------------|---------------|---------------|
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/api_docs/) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-cpu)](https://ci.tensorflow.org/job/tensorflow-master-cpu) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-linux-gpu)](https://ci.tensorflow.org/job/tensorflow-master-linux-gpu) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-mac)](https://ci.tensorflow.org/job/tensorflow-master-mac) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-win-cmake-py)](https://ci.tensorflow.org/job/tensorflow-master-win-cmake-py) | [![Build Status](https://ci.tensorflow.org/buildStatus/icon?job=tensorflow-master-android)](https://ci.tensorflow.org/job/tensorflow-master-android) [ ![Download](https://api.bintray.com/packages/google/tensorflow/tensorflow/images/download.svg) ](https://bintray.com/google/tensorflow/tensorflow/_latestVersion)

**TensorFlow** 是一个使用数据流图进行数值计算开源软件库。
图的节点表示数学运算，节点之间的边表示流动的多维数据数组（张量）。
这种灵活的架构允许你在无需重写代码的情况下，将计算在桌面端、服务端或移动端部署到一个或多个 CPU 和 GPU 中。
TensorFlow 还包含 TensorBoard，它是一个数据可视化工具包。

TensorFlow 最初由 Google 机器智能研究机构内的 
Google Brain 团队的研究人员和工程师开发，用于进行机器学习和深度神经网络研究。
此系统一般足以适用于各种其他领域。

## 安装

**在 [安装 TensorFlow](https://www.tensorflow.org/get_started/os_setup.html) 页面中查看关于稳定二进制版的安装或从源码安装的安装步骤。**

喜欢挑战的人也可以尝试我们的开发版：

**开发版 pip 包**
* 我们非常高兴发布 TensorFlow 的开发版，现在 pypi 提供开发版的 pip 包 [tf-nightly](https://pypi.python.org/pypi/tf-nightly) 和
  [tf-nightly-gpu](https://pypi.python.org/pypi/tf-nightly-gpu) 项目。在干净的环境中简单运行 `pip install tf-nightly` 或 `pip install tf-nightly-gpu` 即可安装 TensorFlow 开发版。 我们为 Linux、Mac 和 Windows 提供  CPU 和 GPU 支持。


**独立的 whl 文件**
* Linux CPU-only: [Python 2](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp27-none-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/)) / [Python 3.4](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp34-cp34m-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/)) / [Python 3.5](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp35-cp35m-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=cpu-slave/)) / [Python 3.6](http://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.6,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp36-cp36m-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.6,label=cpu-slave/))
* Linux GPU: [Python 2](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-linux/42/artifact/pip_test/whl/tf_nightly_gpu-1.head-cp27-none-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=gpu-linux/)) / [Python 3.4](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly_gpu-1.head-cp34-cp34m-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=gpu-linux/)) / [Python 3.5](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly_gpu-1.head-cp35-cp35m-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=gpu-linux/)) / [Python 3.6](http://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.6,label=gpu-linux/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly_gpu-1.head-cp36-cp36m-linux_x86_64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.6,label=gpu-linux/))
* Mac CPU-only: [Python 2](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-py2-none-any.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=mac-slave/)) / [Python 3](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-py3-none-any.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-mac/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=mac-slave/))
* Windows CPU-only: [Python 3.5 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=35/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly-1.head-cp35-cp35m-win_amd64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=35/)) / [Python 3.6 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=36/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly-1.head-cp36-cp36m-win_amd64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows,PY=36/))
* Windows GPU: [Python 3.5 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=35/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly_gpu-1.head-cp35-cp35m-win_amd64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=35/)) / [Python 3.6 64-bit](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=36/lastSuccessfulBuild/artifact/cmake_build/tf_python/dist/tf_nightly_gpu-1.head-cp36-cp36m-win_amd64.whl) ([构建历史](https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-windows/M=windows-gpu,PY=36/))
* Android: [demo APK](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/tensorflow_demo.apk), [native libs](https://ci.tensorflow.org/view/Nightly/job/nightly-android/lastSuccessfulBuild/artifact/out/native/)
  ([构建历史](https://ci.tensorflow.org/view/Nightly/job/nightly-android/))

#### 开启你的第一个 TensorFlow 程序

```shell
$ python
```
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
'Hello, TensorFlow!'
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a + b)
42
>>> sess.close()
```

## 贡献指南

**如果你想参与贡献 TensorFlow，请先查看我们的 [贡献指南](CONTRIBUTING.md)。此项目遵循 TensorFlow
[项目规范](CODE_OF_CONDUCT.md)。我们期望你能遵循此规范。**

**我们还使用 [GitHub issues](https://github.com/tensorflow/tensorflow/issues) 来跟进 requests 和 bugs。对于一般性问题和讨论请查看 
[TensorFlow 讨论](https://groups.google.com/a/tensorflow.org/forum/#!forum/discuss)，或直接在 [Stack Overflow](https://stackoverflow.com/questions/tagged/tensorflow) 提问。**

TensorFlow 项目致力于遵守开源软件开发中普遍接受的最佳实践：

[![CII 最佳实践](https://bestpractices.coreinfrastructure.org/projects/1486/badge)](https://bestpractices.coreinfrastructure.org/projects/1486)

## 更多信息

* [TensorFlow 网站](https://www.tensorflow.org)
* [TensorFlow 白皮书](https://www.tensorflow.org/about/bib)
* [TensorFlow 模型](https://github.com/tensorflow/models)
* [TensorFlow MOOC 教程](https://www.udacity.com/course/deep-learning--ud730)
* [TensorFlow Stanford 教程](https://web.stanford.edu/class/cs20si)

你可以在 [tensorflow.org 社区页](https://www.tensorflow.org/community) 了解更多关于参与 TensorFlow 社区的方法。

## 文档管理团队

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
| [<img src="https://avatars0.githubusercontent.com/u/26959437?s=460&v=4" width="100px;"/><br /><sub>LeviDing</sub>](https://github.com/leviding)<br />[👀](#review-leviding "Reviewed Pull Requests") [🌍](#translation-leviding "Translation") [📋](#eventOrganizing-leviding "Event Organizing") | [<img src="https://avatars0.githubusercontent.com/u/4813445?s=460&v=4" width="100px;"/><br /><sub>pkuwwt</sub>](https://github.com/pkuwwt)<br />[👀](#review-pkuwwt "Reviewed Pull Requests") [🌍](#translation-pkuwwt "Translation") | [<img src="https://avatars1.githubusercontent.com/u/6165782?s=460&v=4" width="100px;"/><br /><sub>John Jiang</sub>](https://github.com/JohnJiangLA)<br />[👀](#review-JohnJiangLA "Reviewed Pull Requests") [🌍](#translation-JohnJiangLA "Translation") | [<img src="https://avatars2.githubusercontent.com/u/5164225?s=460&v=4" width="100px;"/><br /><sub>lsvih</sub>](https://github.com/lsvih)<br />[👀](#review-lsvih "Reviewed Pull Requests") [🌍](#translation-lsvih "Translation") | [<img src="https://avatars3.githubusercontent.com/u/9419075?s=460&v=4" width="100px;"/><br /><sub>foxxnuaa</sub>](https://github.com/foxxnuaa)<br />[👀](#review-foxxnuaa "Reviewed Pull Requests") [🌍](#translation-foxxnuaa "Translation") | [<img src="https://avatars0.githubusercontent.com/u/5498964?s=460&v=4" width="100px;"/><br /><sub>changkun</sub>](https://github.com/changkun)<br />[👀](#review-changkun "Reviewed Pull Requests") [🌍](#translation-changkun "Translation") |
|:-:|:-:|:-:|:-:|:-:|:-:|
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Co-Translators

[所有译者详细信息](https://github.com/xitu/tensorflow-docs/graphs/contributors)

## 文档维护支持方

[<img src="https://user-images.githubusercontent.com/26959437/37653530-37bd3cde-2c7a-11e8-98d0-749a59194c22.png" width="200px;"/>](https://juejin.im)

本文档维护由[掘金](https://juejin.im)提供支持，阅读更多文章或与更多开发者交流请到[掘金](https://juejin.im)，关注[感兴趣的标签](https://juejin.im/subscribe/all)，订阅相关领域实时动态。

## 许可

[Apache 许可 2.0](LICENSE)
