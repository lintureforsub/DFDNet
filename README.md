# DFDNet: Disentangling and Filtering Dynamics for Enhanced Video Prediction

This repository contains the implementation code for the paper:

__DFDNet: Disentangling and Filtering Dynamics for Enhanced Video Prediction__

## Introduction

DFDNet presents an innovative temporal module that disentangle and filter dynamics for enhanced video prediction. 

![DFDNet](/img/figure1.png "The overall framework of DFDNet")
__Left:__ The overall structure of DFDNet. It contains three parts: spatial encoder, temporal module, and spatial decoder. Each layer of the temporal module consists of $k_{i}$ DFDBs in series; __Right:__  The structure of DFDB. DFDB internally contains LayerNorm, STAU (blue dashed box), LayerNorm, and the DFDU (red dashed box). DFDU consists of three pivotal modules: Feature Decomposition, Learnable Threshold Filter, and MLP prediction layer.

## Overview

* `API/` contains dataloaders.
* `TemporalModule/` contains the implement of Temporal Module.
* `DFDNet.py` contains the DFDNet model.
* `main.py` is the executable python file with possible arguments.
* `core.py` is the core file for model training, validating, and testing. 
* `WFL.py` contains the implement of weighted frequency loss.

### 1. Environment install
We provide the environment requirements file for easy reproduction:
```
  conda create -n DFDNet python=3.7
  conda activate DFDNet

  pip install -r requirements.txt
```
### 2. Dataset download

Our model has been experimented on the following four datasets:
* [Moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/)
* [KTH](https://www.csc.kth.se/cvap/actions/)
* [Human3.6M](http://vision.imar.ro/human3.6m/description.php) 
* [SJTU4K](https://medialab.sjtu.edu.cn/post/sjtu-4k-video-sequences/)

We provide a download script for the Moving MNIST dataset:

```
  cd ./data/moving_mnist
  bash download_mmnist.sh 
```

### 3. Model traning

This example provide the detail implementation on Moving MNIST, you can easily reproduce our work using the following command:

```
conda activate DFDNet
python main.py             
```
Please note that __the model training must strictly adhere to the hyperparameter settings provided in our paper__; otherwise, reproducibility may not be guaranteed.

