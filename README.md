# StarGAN-Tensorflow
Implementation of StarGAN in Tensorflow

[StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)
[Official Pytorch Code](https://github.com/yunjey/StarGAN)

This code is mainly revised from goldkim92's [code](https://github.com/goldkim92/StarGAN-tensorflow) base on official pytorch code. More testing function added.


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

To test a model by given a specific attribute:
```
$ python main.py --phase=test --image_size=64 --binary_attrs=1001110
```
Bianry attributes are now set up with the following sequence:
```
'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young', 'Pale_Skin'
```

Sample 100 images from testing data and test different attributes:
```
$ python main.py --phase=test_all --image_size=64
```

To test the classifier of a model:
```
$ python main.py --phase=aux_test --image_size=64
```

## Result


