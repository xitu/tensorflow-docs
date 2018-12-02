# Docker

[Docker](https://docs.docker.com/install/){:.external} 使用**容器**创建了一个与外部系统隔离开的 TensorFlow 运行环境。TensorFlow 可**运行在**这个虚拟环境中并使用宿主机的资源（比如目录访问，使用 GPU，网络连接等等）。[TensorFlow Docker 镜像](https://hub.docker.com/r/tensorflow/tensorflow/){:.external} 已在各个 release 版本上测试可用。

因为只需在宿主机上安装 [NVIDIA® GPU 驱动](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver){:.external}（**NVIDIA® CUDA® Toolkit** 不需安装），所以在 Linux 上使用 Docker 是启用 TensorFlow [GPU 支持](./gpu.md)最简单的方法。

## TensorFlow Docker 安装要求

1. 在本地宿主机[安装 Docker](https://docs.docker.com/install/){:.external}。
2. 在 Linux 环境下获取 GPU 支持，[安装 nvidia-docker](https://github.com/NVIDIA/nvidia-docker){:.external}。

注意：想要不使用 `sudo` 运行 `docker` 命令，需要创建 `docker` 用户组并将你的用户加入此用户组。详细步骤请查看 [Linux 安装后配置](https://docs.docker.com/install/linux/linux-postinstall/){:.external}。

## 下载 TensorFlow Docker 镜像

官方 TensorFlow Docker 镜像位于 Docker Hub 中的 [tensorflow/tensorflow](https://hub.docker.com/r/tensorflow/tensorflow/){:.external}。镜像中的使用以下格式 [Tag](https://hub.docker.com/r/tensorflow/tensorflow/tags/){:.external} 后发布。

<table>
  <tr><th>Tag</th><th>描述</th></tr>
  <tr><td><code>latest</code></td><td>最新发布的已编译 TensorFlow CPU 镜像。默认。</td></tr>
  <tr><td><code>nightly</code></td><td>最新构建 TensorFlow 镜像。（不稳定）</td></tr>
  <tr><td><code><em>version</em></code></td><td>特定<em>版本</em>的已编译 TensorFlow 镜像，例如：<em>1.11</em></td></tr>
  <tr class="alt"><td colspan="2">Tag 后缀</td></tr>
  <tr><td><code><em>tag</em>-devel<code></td><td>特定 <em>tag</em> 的发布版本和源码。</td></tr>
  <tr><td><code><em>tag</em>-gpu<code></td><td>特定 <em>tag</em> 的带有 GPU 支持的发布版本。 (<a href="#gpu_support">查看</a>)</td></tr>
  <tr><td><code><em>tag</em>-py3<code></td><td>特定 <em>tag</em> 的支持 Python 3 的发布版本。</td></tr>
  <tr><td><code><em>tag</em>-gpu-py3<code></td><td>特定 <em>tag</em> 的带有 GPU-支持并支持 Python 3 的发布版本。</td></tr>
  <tr><td><code><em>tag</em>-devel-py3<code></td><td>特定 <em>tag</em> 的支持 Python 3 的发布版本和源码。</td></tr>
  <tr><td><code><em>tag</em>-devel-gpu<code></td><td>特定 <em>tag</em> 的带有 GPU-支持 的发布版本和源码。</td></tr>
  <tr><td><code><em>tag</em>-devel-gpu-py3<code></td><td>特定 <em>tag</em> 的带有 GPU 支持并支持 Python 3 的发布版本以及源码。td></tr>
</table>

For example, the following downloads TensorFlow release images to your machine:
例如，使用如下命令下载 TensorFlow 发布版本镜像到本地机器

<pre class="devsite-click-to-copy prettyprint lang-bsh">
<code class="devsite-terminal">docker pull tensorflow/tensorflow                    # 最新稳定发布版本</code>
<code class="devsite-terminal">docker pull tensorflow/tensorflow:nightly-devel-gpu  # 最新开发版本 w/ GPU 支持</code>
</pre>

## 启动 TensorFlow Docker 容器

使用以下命令启动一个已配置的 TensorFlow 容器：

<pre class="devsite-terminal devsite-click-to-copy">
docker run [-it] [--rm] [-p <em>hostPort</em>:<em>containerPort</em>] tensorflow/tensorflow[:<em>tag</em>] [<em>command</em>]
</pre>

详细内容，请查看 [docker 启动参考](https://docs.docker.com/engine/reference/run/){:.external}。

### 使用 Cpu-only 镜像的样例

我们使用 tag 为 `latest` 的镜像确认 TensorFlow 安装过程。Docker 在第一次运行的时候会下载一个新的 TensorFlow 镜像：

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-bsh">
docker run -it --rm tensorflow/tensorflow \
    python -c "import tensorflow as tf; print(tf.__version__)"
</pre>

成功：TensorFlow 现在已安装完成。查看[教程](../tutorials)开始你的学习之旅吧。

我们来看下更多 TensorFlow Docker 的用法。在 TensorFlow 已配置容器中启动一个 `bash` 命令行会话：

<pre class="devsite-terminal devsite-click-to-copy">
docker run -it tensorflow/tensorflow bash
</pre>

在这个容器中，你可以启动一个 `python` 会话并导入 TensorFlow。
To run a TensorFlow program developed on the *host* machine within a container, mount the host directory and change the container's working directory
为了在容器中运行**宿主机**中开发的 TensorFlow 程序，需要将宿主机的主目录挂载到容器中并设定容器的工作路径
(`-v hostDir:containerDir -w workDir`)：

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-bsh">
docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./script.py
</pre>

在容器中创建文件时会出现权限问题并会暴露给宿主机。所以最好在宿主机中进行文件操作。

启动一个由使用 Python 3 构建的最新版 TensorFlow 服务支持的 [Jupyter Notebook](https://jupyter.org/){:.external}。

<pre class="devsite-terminal devsite-click-to-copy">
docker run -it -p 8888:8888 tensorflow/tensorflow:nightly-py3
</pre>

按照说明在浏览器中打开 URL：`http://127.0.0.1:8888/?token=...`。

## GPU 支持

因为只需在宿主机上安装 [NVIDIA® GPU 驱动](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver){:.external}（**NVIDIA® CUDA® Toolkit** 不需安装），所以在 Linux 上使用 Docker 是启用 TensorFlow [GPU 支持](./gpu.md)最简单的方法。

安装 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker){:.external} 获得支持 NVIDIA® GPU 支持的 Docker 容器。`nvidia-docker` 仅供 Linux 环境下使用，详细内容请查看[平台支持 FAQ](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#platform-support){:.external}。

查看 GPU 是否可用：

<pre class="devsite-terminal devsite-click-to-copy">
lspci | grep -i nvidia
</pre>

确认 `nvidia-docker` 安装过程

<pre class="devsite-terminal devsite-click-to-copy">
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
</pre>

注意：`nvidia-docker` v1 使用别名 `nvidia-docker`，v2 则需使用 `docker --runtime=nvidia`。

### 使用已启用 GPU 镜像的样例

下载并启动已启用 GPU 的 TensorFlow 镜像（可能需要花费几分钟）：

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-bsh">
docker run --runtime=nvidia -it --rm tensorflow/tensorflow:latest-gpu \
    python -c "import tensorflow as tf; print(tf.contrib.eager.num_gpus())"
</pre>

配置已启用 GPU 的镜像可能会花费很长时间。如果需要重复运行基于 GPU 的脚本，你可以使用 `docker exec` 来重用容器。

使用最新的 TensorFlow GPU 镜像并在容器中运行一个 `bash` 命令行会话：

<pre class="devsite-terminal devsite-click-to-copy">
docker run --runtime=nvidia -it tensorflow/tensorflow:latest-gpu bash
</pre>

成功：TensorFlow 现在已安装完成。查看[教程](../tutorials)开始你的学习之旅吧。
