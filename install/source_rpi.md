# 为树莓派从源码构建安装包

本指南为运行 [Raspbian 9.0](https://www.raspberrypi.org/downloads/raspbian/){:.external} 的[树莓派](https://www.raspberrypi.org/){:.external} 设备构建了 TensorFlow 安装包。同时该指南可能也适用于其他版本的树莓派，但是只对这个配置的版本进行了测试和适配。

我们推荐使用**交叉编译** TensorFlow 的 Raspbian 的安装包。交叉编译是使用不同平台去构建包而不去部署。不使用树莓派有限的内存和相当慢的处理器，使用运行着 Linux、macOS 或者 Windows 系统，性能更强的主机构建 TensorFlow 会更容易。

注意：我们已经为 Raspbian 系统提供了测试好，预编译的 [TensorFlow 包](./pip.md)。

## 设置主机

### 安装 Docker

为了简化依赖管理，构建脚本使用 [Docker](https://docs.docker.com/install/){:.external} 为编译创建一个虚拟的 Linux 开发环境。通过执行以下命令验证你的 Dokcer 是否安装成功：`docker run --rm hello-world`

### 下载 TensorFlow 源码

使用 [Git](https://git-scm.com/){:.external} 克隆 [TensorFlow 仓库](https://github.com/tensorflow/tensorflow){:.external}：

<pre class="devsite-click-to-copy">
<code class="devsite-terminal">git clone https://github.com/tensorflow/tensorflow.git</code>
<code class="devsite-terminal">cd tensorflow</code>
</pre>

仓库默认在 `master` 开发分支. 你同样可以切换到一个 [release 分支](https://github.com/tensorflow/tensorflow/releases){:.external} 去构建：

<pre class="devsite-terminal prettyprint lang-bsh">
git checkout <em>branch_name</em>  # r1.9, r1.10, etc.
</pre>

要点：如果构建最近的开发分支时期间遇到了问题，尝试一个你确定能工作的 release 分支。

## 从源码构建

使用 ARMv7 [NEON 指南](https://developer.arm.com/technologies/neon){:.external}交叉编译 TensorFlow 源码去构建一个 Python *pip* 包，其可以运行在 树莓派 2 和 3 设备上。构建脚本启动一个 Docker 容器进行编译。 为编译出来的包在 Python 3 和 Python 2.7 之间进行选择：

<div class="ds-selector-tabs">
  <section>
    <h3>Python 3</h3>
<pre class="devsite-terminal prettyprint lang-bsh">
CI_DOCKER_EXTRA_PARAMS="-e CI_BUILD_PYTHON=python3 -e CROSSTOOL_PYTHON_INCLUDE_PATH=/usr/include/python3.4" \\
    tensorflow/tools/ci_build/ci_build.sh PI-PYTHON3 \\
    tensorflow/tools/ci_build/pi/build_raspberry_pi.sh
</pre>
  </section>
<section>
    <h3>Python 2.7</h3>
<pre class="devsite-terminal prettyprint lang-bsh">
tensorflow/tools/ci_build/ci_build.sh PI \\
    tensorflow/tools/ci_build/pi/build_raspberry_pi.sh
</pre>
  </section>
</div><!--/ds-selector-tabs-->

如果需要构建一个适用于全部树莓派设备的包 — 包括 Pi 1 和 Zero，通过 `PI_ONE` 参数，举个例子：

<pre class="devsite-terminal prettyprint lang-bsh">
tensorflow/tools/ci_build/ci_build.sh PI \
    tensorflow/tools/ci_build/pi/build_raspberry_pi.sh PI_ONE
</pre>

当构建完成后（大约 30 分钟），一个 `.whl` 包文件会被创建在主机源树的output-artifacts目录中。将 wheel 文件复制给树莓派并使用 `pip` 进行安装：

<pre class="devsite-terminal devsite-click-to-copy">
pip install tensorflow-<var>version</var>-cp34-none-linux_armv7l.whl
</pre>

成功：TensorFlow 现在已经安装在了 Raspbian 上。
