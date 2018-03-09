Skip to content
This repository
Search
Pull requests
Issues
Marketplace
Explore
 @leviding
 Sign out
14
32 1 xitu/tensorflow-docs
 Code  Issues 1  Pull requests 0  Projects 0  Wiki  Insights  Settings
tensorflow-docs/programmers_guide/ 
using_tpu.md
  or cancel
    
 
1
# Using TPUs
2
​
3
This document walks through the principal TensorFlow APIs necessary to make
4
effective use of a [Cloud TPU](https://cloud.google.com/tpu/), and highlights
5
the differences between regular TensorFlow usage, and usage on a TPU.
6
​
7
This doc is aimed at users who:
8
​
9
* Are familiar with TensorFlow's `Estimator` and `Dataset` APIs
10
* Have maybe [tried out a Cloud TPU](https://cloud.google.com/tpu/docs/quickstart)
11
  using an existing model.
12
* Have, perhaps, skimmed the code of an example TPU model
13
  [[1]](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_tpu.py)
14
  [[2]](https://github.com/tensorflow/tpu-demos/tree/master/cloud_tpu/models).
15
* Are interested in porting an existing `Estimator` model to
16
  run on Cloud TPUs
17
​
18
## TPUEstimator
19
​
20
@{tf.estimator.Estimator$Estimators} are TensorFlow's model-level abstraction.
21
Standard `Estimators` can drive models on CPU and GPUs. You must use
22
@{tf.contrib.tpu.TPUEstimator} to drive a model on TPUs.
23
​
24
Refer to TensorFlow's Getting Started section for an introduction to the basics
25
of using a @{$get_started/premade_estimators$pre-made `Estimator`}, and
26
@{$get_started/custom_estimators$custom `Estimator`s}.
27
​
28
The `TPUEstimator` class differs somewhat from the `Estimator` class.
29
​
30
The simplest way to maintain a model that can be run both on CPU/GPU or on a
31
Cloud TPU is to define the model's inference phase (from inputs to predictions)
32
outside of the `model_fn`. Then maintain separate implementations of the
33
`Estimator` setup and `model_fn`, both wrapping this inference step. For an
34
example of this pattern compare the `mnist.py` and `mnist_tpu.py` implementation in
35
[tensorflow/models](https://github.com/tensorflow/models/tree/master/official/mnist).
36
​
37
### Running a `TPUEstimator` locally
38
​
39
To create a standard `Estimator` you call the constructor, and pass it a
40
`model_fn`, for example:
41
​
42
```
43
my_estimator = tf.estimator.Estimator(
44
  model_fn=my_model_fn)
45
```
@leviding
Commit changes

Update using_tpu.md

Add an optional extended description…
  Commit directly to the v1.6 branch.
  Create a new branch for this commit and start a pull request. Learn more about pull requests.
 
© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
API
Training
Shop
Blog
About
