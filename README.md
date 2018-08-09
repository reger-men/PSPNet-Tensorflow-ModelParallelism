# PSPNet-Tensorflow-ModelParallelism
Tensorflow Model Parallelism approach of PSPNet 

### Introduction
This is an implementation of PSPNet in TensorFlow for semantic segmentation on the cityscapes dataset. 
To be able to train this network with the full input size (1024x2048), the memory capacity should be big enough or you must use the model parallelism approach.

### GPUs
![](https://github.com/reger-men/PSPNet-Tensorflow-ModelParallelism/blob/master/screenshots/gpus.png)

### The problem
It looks like the memory is being allocated several times. I run the same model on single GPU and on multiple GPUs. The following screenshots show the memory usage on different GPUs:

