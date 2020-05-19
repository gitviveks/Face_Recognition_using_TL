# Face Recognition using Transfer Learning on VGG16
## Transfer Learning
In the data science world, we face a lot of situation which require a lot of computational power or resources like RAM, CPU or GPU, etc. to train our models. To cope up with this situation one of the best ways is to use the pre-trained model as in Transfer Learning. This will also help to achieve more accuracy with less amount of data.

## VGG16
VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”. The model achieves 92.7% top-5 test accuracy in ImageNet, which is a dataset of over 14 million images belonging to 1000 classes. It was one of the famous models submitted to ILSVRC-2014. It makes the improvement over AlexNet by replacing large kernel-sized filters (11 and 5 in the first and second convolutional layer, respectively) with multiple 3×3 kernel-sized filters one after another. VGG16 was trained for weeks and was using NVIDIA Titan Black GPUs.

## Prerequisites
Install following libraires:
```
conda install tensorflow keras opencv-python pillow numpy
```
