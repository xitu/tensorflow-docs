# Docker

 [Docker](https://docs.docker.com/install/){:.external} 使用**容器**创建了一个与外部系统隔离开的 TensorFlow 安装环境。TensorFlow 可**运行在**这个虚拟环境中并使用宿主机的资源（比如目录访问，使用 GPU，网络连接等等）。[TensorFlow Docker 镜像](https://hub.docker.com/r/tensorflow/tensorflow/){:.external}已在各个 release 版本上测试可用。

因为只需在宿主机上安装 [NVIDIA® GPU 驱动](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver){:.external}（**NVIDIA® CUDA® Toolkit** 不需安装），所以在 Docker 上使用是 Linux 环境下启用 TensorFlow 的 [GPU 支持](./gpu.md)最简单的方法。   

## TensorFlow Docker 安装要求

1. 在本地宿主机[安装 Docker](https://docs.docker.com/install/){:.external}。
2. 在 Linux 环境下获取 GPU 支持， [安装 nvidia-docker](https://github.com/NVIDIA/nvidia-docker){:.external}。

Note: To run the `docker` command without `sudo`, create the `docker` group and add your user. For details, see the [post-installation steps for Linux](https://docs.docker.com/install/linux/linux-postinstall/){:.external}.

注意：想要不使用 `sudo` 运行 `docker` 命令，需要创建 `docker` 用户组并将你的用户加入此用户组。详细步骤请查看 [Linux 安装后配置](https://docs.docker.com/install/linux/linux-postinstall/){:.external}。

## 下载 TensorFlow Docker 镜像

The official TensorFlow Docker images are located in the [tensorflow/tensorflow](https://hub.docker.com/r/tensorflow/tensorflow/){:.external} Docker Hub repository. Image releases [are tagged](https://hub.docker.com/r/tensorflow/tensorflow/tags/){:.external} using the following format:
官方 TensorFlow Docker 镜像位于 Docker Hub 中的 [tensorflow/tensorflow](https://hub.docker.com/r/tensorflow/tensorflow/){:.external}。镜像中的使用以下格式 [Tag](https://hub.docker.com/r/tensorflow/tensorflow/tags/){:.external} 后发布。

<table>
  <tr><th>Tag</th><th>描述</th></tr>
  <tr><td><code>latest</code></td><td>最新发布的 TensorFlow CPU 已编译镜像. 默认.</td></tr>
  <tr><td><code>nightly</code></td><td>最新构建 TensorFlow 镜像. (不稳定)</td></tr>
  <tr><td><code><em>version</em></code></td><td>特定<em>版本</em>的 TensorFlow 已编译镜像，例如：<em>1.11</em></td></tr>
  <tr class="alt"><td colspan="2">Tag 后缀</td></tr>
  <tr><td><code><em>tag</em>-devel<code></td><td>特定 <em>tag</em> 的发布版本和源码。</td></tr>
  <tr><td><code><em>tag</em>-gpu<code></td><td>特定 <em>tag</em> 的带有 GPU-支持的发布版本。 (<a href="#gpu_support">查看</a>)</td></tr>
  <tr><td><code><em>tag</em>-py3<code></td><td>特定 <em>tag</em> 的支持 Python 3 的发布版本。</td></tr>
  <tr><td><code><em>tag</em>-gpu-py3<code></td><td>特定 <em>tag</em> 的带有 GPU-支持并支持 Python 3 的发布版本。</td></tr>
  <tr><td><code><em>tag</em>-devel-py3<code></td><td>特定 <em>tag</em> 的支持 Python 3 的发布版本和源码。</td></tr>
  <tr><td><code><em>tag</em>-devel-gpu<code></td><td>特定 <em>tag</em> 的带有 GPU-支持 的发布版本和源码。</td></tr>
  <tr><td><code><em>tag</em>-devel-gpu-py3<code></td><td>特定 <em>tag</em> 的带有 GPU-支持并支持 Python 3 的发布版本以及源码。td></tr>
</table>

For example, the following downloads TensorFlow release images to your machine:
例如，使用如下命令下载 TensorFlow 发布版本镜像到本地机器

<pre class="devsite-click-to-copy prettyprint lang-bsh">
<code class="devsite-terminal">docker pull tensorflow/tensorflow                    # 最新稳定发布版本</code>
<code class="devsite-terminal">docker pull tensorflow/tensorflow:nightly-devel-gpu  # 最新开发版本 w/ GPU-支持</code>
</pre>


## Start a TensorFlow Docker container

To start a TensorFlow-configured container, use the following command form:

<pre class="devsite-terminal devsite-click-to-copy">
docker run [-it] [--rm] [-p <em>hostPort</em>:<em>containerPort</em>] tensorflow/tensorflow[:<em>tag</em>] [<em>command</em>]
</pre>

For details, see the [docker run reference](https://docs.docker.com/engine/reference/run/){:.external}.

### Examples using CPU-only images

Let's verify the TensorFlow installation using the `latest` tagged image. Docker downloads a new TensorFlow image the first time it is run:

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-bsh">
docker run -it --rm tensorflow/tensorflow \
    python -c "import tensorflow as tf; print(tf.__version__)"
</pre>

Success: TensorFlow is now installed. Read the [tutorials](../tutorials) to get started.

Let's demonstrate some more TensorFlow Docker recipes. Start a `bash` shell session within a TensorFlow-configured container:

<pre class="devsite-terminal devsite-click-to-copy">
docker run -it tensorflow/tensorflow bash
</pre>

Within the container, you can start a `python` session and import TensorFlow.

To run a TensorFlow program developed on the *host* machine within a container, mount the host directory and change the container's working directory
(`-v hostDir:containerDir -w workDir`):

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-bsh">
docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./script.py
</pre>

Permission issues can arise when files created within a container are exposed to the host. It's usually best to edit files on the host system.

Start a [Jupyter Notebook](https://jupyter.org/){:.external} server using TensorFlow's nightly build with Python 3 support:

<pre class="devsite-terminal devsite-click-to-copy">
docker run -it -p 8888:8888 tensorflow/tensorflow:nightly-py3
</pre>

Follow the instructions and open the URL in your host web browser: `http://127.0.0.1:8888/?token=...`


## GPU support

Docker is the easiest way to run TensorFlow on a GPU since the *host* machine only requires the [NVIDIA® driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver){:.external} (the *NVIDIA® CUDA® Toolkit* is not required).

Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker){:.external} to launch a Docker container with NVIDIA® GPU support. `nvidia-docker` is only available for Linux, see their [platform support FAQ](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#platform-support){:.external} for details.

Check if a GPU is available:

<pre class="devsite-terminal devsite-click-to-copy">
lspci | grep -i nvidia
</pre>

Verify your `nvidia-docker` installation:

<pre class="devsite-terminal devsite-click-to-copy">
docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
</pre>

Note: `nvidia-docker` v1 uses the `nvidia-docker` alias, where v2 uses `docker --runtime=nvidia`.

### Examples using GPU-enabled images

Download and run a GPU-enabled TensorFlow image (may take a few minutes):

<pre class="devsite-terminal devsite-click-to-copy prettyprint lang-bsh">
docker run --runtime=nvidia -it --rm tensorflow/tensorflow:latest-gpu \
    python -c "import tensorflow as tf; print(tf.contrib.eager.num_gpus())"
</pre>

It can take a while to set up the GPU-enabled image. If repeatably running GPU-based scripts, you can use `docker exec` to reuse a container.

Use the latest TensorFlow GPU image to start a `bash` shell session in the container:

<pre class="devsite-terminal devsite-click-to-copy">
docker run --runtime=nvidia -it tensorflow/tensorflow:latest-gpu bash
</pre>

Success: TensorFlow is now installed. Read the [tutorials](../tutorials) to get started.
