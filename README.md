# SRCNN-Tensorflow
Tensorflow implementation of Convolutional Neural Networks for super-resolution. The original Matlab and Caffe from official website can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).

## Prerequisites
 * Tensorflow
 * Scipy version > 0.18 ('mode' option from scipy.misc.imread function)
 * h5py
 * matplotlib

This code requires Tensorflow. Also scipy is used instead of Matlab or OpenCV. Especially, installing OpenCV at Linux is sort of complicated. So, with reproducing this paper, I used scipy instead. For more imformation about scipy, click [here](https://www.scipy.org/). CuDNN/CUDA are required for gpu support. I believe the correct versions are Cuda 8 and CuDNN 6. Note a Nvidia account is needed for CuDNN

## Usage
For training, `python main.py`
<br>
For testing, `python main.py --is_train False --stride 21`

## Result
After training 10,000 epochs, we got gota good super-resolved image compared to bicubic. Training took around 8 hrs on a 980ti Result images from tegg89 are shown below.<br><br>
Original butterfly image:
![orig](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/orig.png)<br>
Bicubic interpolated image:
![bicubic](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/bicubic.png)<br>
Super-resolved image:
![srcnn](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/srcnn.png)

## References
* [tegg89](https://github.com/tegg89/SRCNN-Tensorflow)
  * base code that we made usabilty improvements to.
