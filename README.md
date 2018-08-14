# PSPNet-Tensorflow-ModelParallelism
Tensorflow Model Parallelism approach of PSPNet 

### Introduction
This is an implementation of PSPNet in TensorFlow for semantic segmentation on the cityscapes dataset. 
To be able to train this network with the full input size (1024x2048), the memory capacity should be big enough or you must use the model parallelism approach.

### run
Download the cityscapes dataset and run this command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --data-dir=/Dataset_Path/cityscapes/ --input-size="height,width" --batch-size=n>
```

### GPUs
![](https://github.com/reger-men/PSPNet-Tensorflow-ModelParallelism/blob/master/screenshots/gpus.png)

### The problem
It looks like the memory is being allocated several times. I run the same model on single GPU and on multiple GPUs. The following screenshots show the memory usage on different GPUs:

On GPU:0                |  On GPU:0 and 1
:-------------------------:|:-------------------------:
![](https://github.com/reger-men/PSPNet-Tensorflow-ModelParallelism/blob/master/screenshots/gpu0.png)  |  ![](https://github.com/reger-men/PSPNet-Tensorflow-ModelParallelism/blob/master/screenshots/gpu01.png)

On GPU:0, 1 and 2                |  On GPU:0, 1, 2 and 3
:-------------------------:|:-------------------------:
![](https://github.com/reger-men/PSPNet-Tensorflow-ModelParallelism/blob/master/screenshots/gpu012.png)  |  ![](https://github.com/reger-men/PSPNet-Tensorflow-ModelParallelism/blob/master/screenshots/gpu0123.png)

### Issue URL
https://github.com/tensorflow/tensorflow/issues/21522
