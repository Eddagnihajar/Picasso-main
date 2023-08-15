<p align="center">
# Geometric Feature Learning for 3D Meshes
</p>

![alt text](https://github.com/EnyaHermite/Picasso/blob/main/image/teaser.png)

### Introduction
[This journal work](https://arxiv.org/abs/2112.01801) is a sigificnat extension of [our original work](https://arxiv.org/abs/2103.15076) presented in CVPR 2021. We have improved the point cloud modules of *SPH3D-GCN* from homogeneous to heterogeneous representations as well, and included the point cloud modules into the picasso library. In this realease, Picasso is made available in both Pytorch and Tensorflow. 

Geometric feature learning for 3D meshes is central to computer graphics and highly important for numerous vision applications. However, deep learning currently lags in hierarchical modeling of heterogeneous 3D meshes due to the lack of required operations and/or their efficient implementations. In this paper, we propose a series of modular operations for effective geometric deep learning over heterogeneous 3D meshes. These operations include mesh convolutions, (un)pooling and efficient mesh decimation. We provide open source implementation of these operations, collectively termed Picasso. The mesh decimation module of Picasso is GPU-accelerated, which can process a batch of meshes on-the-fly for deep learning. Our (un)pooling operations compute features for newly-created neurons across network layers of varying resolution. Our mesh convolutions include facet2vertex, vertex2facet, and facet2facet convolutions that exploit vMF mixture and Barycentric interpolation to incorporate fuzzy modelling. Leveraging the modular operations of Picasso, we contribute a novel hierarchical neural network, PicassoNet-II, to learn highly discriminative features from 3D meshes. PicassoNet-II accepts primitive geometrics and fine textures of mesh facets as input features, while processing full scene meshes. Our network achieves highly competitive performance for shape analysis and scene parsing on a variety of benchmarks. We release Picasso and PicassoNet-II from Github.

### Citation
If you find our work useful in your research, please consider citing:

```
@article{lei2021geometric,
  title={Geometric Feature Learning for 3D Meshes},
  author={Lei, Huan and Akhtar, Naveed and Shah, Mubarak and Mian, Ajmal},
  journal={arXiv preprint arXiv:2112.01801},
  year={2021}
}
```
```
@inproceedings{lei2021picasso,
  title={Picasso: A CUDA-based Library for Deep Learning over 3D Meshes},
  author={Lei, Huan and Akhtar, Naveed and Mian, Ajmal},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13854--13864},
  year={2021}
} 
```

## Tensorflow 
1. #### Installation
- For tensorflow users, please use the provided package in *tensorflow* folder. The code was tested with Python 3.8, Tensorflow 2.4, Cuda 11.0 and Cudnn 8.0 on Ubuntu 18.04. **Note: We assumed that the GPU supports a block of 1024 threads.** 
  
- Please compile the cuda-based operations for tensorflow using the following commands:
```
$ cd picasso/mesh/modules/source
$ ./compile.sh
$ cd picasso/point/modules/source
$ ./compile.sh
```

2. Data Preparation
- We provide files in the `io' folder to process raw meshes to tfrecord format for Tensorflow usage. The raw meshes for HUMAN Body segmentation, ShapeNetCore, and S3DIS datasets can be downloaded from [this link](https://drive.google.com/drive/folders/1wyOpd5YnE9nUKe2m6oWQz9DgOLXCKwTg?usp=sharing).
```
$ cd io
$ python make_tfrecord_shapenetcore.py 
$ python make_tfrecord_human.py 
$ python make_tfrecord_scannet.py  
$ python make_tfrecord_s3dis.py    
```
3. Training and Testing
- **ShapeNetCore**
  * To train a model to classify the 55 object classes:
    ```
    $ ./train_shapenetcore.sh  
    ```
  * To test the classification results with augmentations:
    ```
    $ python evaluate_shapenetcore.py --ckpt_epoch=#checkpoint -num_augment=20 --data_dir=../TFdata/ShapeNetCore
    ```

- **HUMAN BODY**   
  * To train a model to segment different parts of the human body:
    ```
    $ ./train_human.sh
    ```

- **ScanNet V2**  
  * train 
    ```  
    $ ./train_scannet.sh
    ```
  * test
    ```
    $ python evaluate_scannet_plain.py --ckpt_epoch=#checkpoint --num_augment=20 --gpu=0 --data_dir=../TFdata/ScanNet
    ```

- **S3DIS**    
  * train  
    ```   
    $ ./train_s3dis.sh
    ```
  * test   
    ```
    $ python evaluate_s3dis_plain.py --ckpt_epoch=#checkpoint --num_augment=20 --gpu=0 --data_dir=../TFdata/S3DIS_Aligned_3cm_Mesh
    $ python s3dis_voxel2dense.py --dense_dir=../RAWdata/S3DIS_Aligned_dense_PointCloud	
    ```

## Pytorch Installation
Install [Pytorch](https://pytorch.org/get-started/locally/). The code was tested with Python 3.8, Pytorch 10, Cuda 11.3 and Cudnn 8.2 on Ubuntu 18.04. The used GPU is NVIDIA GeForce RTX 3090.   
**Note: We assumed that the GPU supports a block of 1024 threads. 
  
Please compile the cuda-based operations for meshes using the command
```
$ cd picasso/mesh/modules/source
$ python setup.py install
$ cd picasso/point/modules/source
$ python setup.py install
```
