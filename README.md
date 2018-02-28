# StarGAN-Tensorflow
Implementation of StarGAN in Tensorflow

[StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)
[Official Pytorch Code](https://github.com/yunjey/StarGAN)

This code is mainly revised from goldkim92's [code](https://github.com/goldkim92/StarGAN-tensorflow) base on official pytorch code. 
* Modifying the code to be more consistent to the official implementation.
* Fixing the bug in lost calculation.
* More testing function added.
* Adding Residual Block (base on [this code](https://github.com/xhujoy/CycleGAN-tensorflow/blob/master/module.py))



## Prerequisites
* Python 3.5
* Tensorflow 1.3.0
* Scipy
* tqdm

## Usage
Only CelebA part is implemented.

First, download dataset with:
```
$ python download.py
```
To train a model:
```
$ python main.py --phase=train --image_size=64 --batch_size=16
```
The default classification method is using sigmoid. If the attributes you chose satisfy sigle attribute classification (ex: hair color only. Or if you can access to RAFD), you could also try softmax.

```
$ python main.py --phase=train --image_size=64 --batch_size=16 --c_method=Softmax
```
The default adversarial training method is improved WGAN. You could also try different method such as LSGAN or GAN. But personally I've only tried the improved WGAN.
```
$ python main.py --phase=train --image_size=64 --batch_size=16 --adv_type=LSGAN
```
The output format of the sample image during training:

|                   | Orignial | Target | Reconstruct |
|-------------------|----------|--------|-------------|
| Target=Black Hair |          |        |             |
| Target=Blond Hair |          |        |             |
| Target=Brown Hair |          |        |             |
| ...               |          |        |             |


To test a model by given a specific attribute:
```
$ python main.py --phase=test --image_size=64 --binary_attrs=100000
```
The output format of the image is like:

|     | Orignial | Target | Reconstruct |
|-----|----------|--------|-------------|
| img |          |        |             |

Bianry attributes are now set up with the following sequence:
```
'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young', 'Pale_Skin'
```
You could modify the attributes in the main.py

Sample 100 images from testing data and test each image with each attribute:
```
$ python main.py --phase=test_all --image_size=64
```
The output format of the image is like:

|     | Orignial | Black Hair | Blond Hair | Brown Hair | Male | Young | Pale Skin |
|-----|----------|------------|------------|------------|------|-------|-----------|
| img |          |            |            |            |      |       |           |

To test the classifier of a model:
```
$ python main.py --phase=aux_test --image_size=64
```

## Result


