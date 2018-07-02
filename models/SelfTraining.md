
# Training on your own dataset

很感谢原作者写了这部分的内容，但，还是没有说清楚，有很多坑仍未解决，简直被虐的不轻。
步骤总结为以下四个：

 - 1、下载`pre-trained ResNet weights`
 - 2、准备`dataset.mat`
 - 3、修改`pose_cfg.yaml`
 - 4、训练


----------


### 1、下载`pre-trained ResNet weights`
Before training the models you need to download the ImageNet pre-trained ResNet weights.

```
$ cd models/pretrained
$ ./download.sh
```
正常下载即可，会在./models/pretrained下生成：`resnet_v1_50.ckpt`以及`resnet_v1_101.ckpt`两个文件。


----------


### 2、准备`dataset.mat`
#### 2.1 元数据格式介绍
原作者给出的元数据为[dataset_example.mat](https://github.com/eldar/pose-tensorflow/blob/master/models/dataset_example.mat)，所需的三个元素分别为：

 - `image` - path to the image
 - `size` - 1x3 array containing **[num_channels, image_height, image_width]**. Set num_channels=3 for an RGB image.
 - `joints` - a cell array of nx3 joint annotations, for example:

```
joints = {[ ...
  0,  175,  261; ... % 0-indexed joint ID, X coordinate, Y coordinate
  1,  173,  178; ...
  2,  144,  122; ...
  3,  193,  124; ...
]};
```

#### 2.2 元数据生成
原作者没有给出生成方式，而且还是.mat格式的，matlab又用得不熟练，于是乎转战使用`import scipy.io as sio`.
```
array([[(array(['/dir/dataset/im00005_1.png'], dtype='<U26'), array([[  3, 391, 295]], dtype=uint16), array([[array([[  0, 175, 261],
       [  1, 173, 178],
       [  2, 144, 122],
       [  3, 193, 124],
       [  4, 203, 146],
       [  5, 199, 153],
       [  6, 166, 144],
       [  7, 131, 107],
       [  8, 163, 111],
       [  9, 223, 122],
       [ 10, 224, 159],
       [ 11, 220, 207],
       [ 12, 187, 126],
       [ 13, 226,  72]], dtype=int32)]], dtype=object)),
        (array(['/dir/dataset/im00006_3.png'], dtype='<U26'), array([[  3, 252, 170]], dtype=uint8), array([[array([[  2,  80, 187],
       [  3,  26, 187],
       [  6,  77, 130],
       [  7, 104, 159],
       [  8,  81,  88],
       [  9,   9,  88],
       [ 10,   9, 148],
       [ 12,  43,  76],
       [ 13,  31,  11]], dtype=int32)]], dtype=object))]],
      dtype=[('image', 'O'), ('size', 'O'), ('joints', 'O')])
```
 - 图片路径，为array，U26格式；
 - 图片尺寸，为array,uint8格式；
 - 图片关键点，这里非常特殊，笔者从np.array一直生成不到一模一样的格式，于是不得不草船借箭，从案例数据（`data
   templates`）那边把格式借过来。

笔者写了一个简单实现案例，可以借鉴：[dataset_generator.py](https://github.com/mattzheng/pose-tensorflow/blob/master/models/dataset_generator.py)


----------


### 3、修改`pose_cfg.yaml`
这里原作者有个坑，一些参数没有写在.yaml文件之中。

```
# path to the dataset description file
dataset: models/train_dataset.mat  # matt,careful!!

# all locations within this distance threshold are considered
# positive training samples for detector
pos_dist_thresh: 17 

# all images in the dataset will be rescaled by the following
# scaling factor to be processed by the CNN. You can select the
# optimal scale by cross-validation
global_scale: 0.80

# During training an image will be randomly scaled within the
# range [scale_jitter_lo; scale_jitter_up] to augment training data,
# We found +/- 15% scale jitter works quite well.
scale_jitter_lo: 0.85
scale_jitter_up: 1.15

# Randomly flips an image horizontally to augment training data
mirror: true

# list of pairs of symmetric joint IDs, for example in this case
# 0 and 5 are IDs for the symmetric parts, and 12 or 13 do not have
# symmetric parts. This is used to do flip training data correctly. 
# 这里区别最大
all_joints: [[10,13],[9,12],[8,11],[4,7],[3,6],[2,5],[16,17],[14,15],[0],[1]] # matt,careful!!
num_joints : 18   # matt,careful!!
snapshot_prefix : "./snapshot"  # matt,careful!!

# Type of the CNN to use, currently resnet_101 and resnet_50
# are supported
net_type: resnet_101
init_weights: ../../pretrained/resnet_v1_101.ckpt

# Location refinement parameters (check https://arxiv.org/abs/1511.06645)
location_refinement: true
locref_huber_loss: true
locref_loss_weight: 0.05
locref_stdev: 7.2801

# Enabling this adds additional loss layer in the middle of the ConvNet,
# which helps accuracy.
intermediate_supervision: true
intermediate_supervision_layer: 12

# all images larger with size
# width * height > max_input_size*max_input_size are not used in training.
# Prevents training from crashing with out of memory exception for very
# large images.
max_input_size: 850

# Learning rate schedule for the SGD optimiser. 
multi_step:
- [0.005, 10000]
- [0.02, 430000]
- [0.002, 730000]
- [0.001, 1030000]

# How often display loss
display_iters: 20

# How often to save training snapshot
save_iters: 6000
```

**主要区别在以下几个内容的设置：**

```
# 这里区别最大
all_joints: [[10,13],[9,12],[8,11],[4,7],[3,6],[2,5],[16,17],[14,15],[0],[1]] # matt
num_joints : 18   # matt
snapshot_prefix : "./snapshot"  # matt
```

 - all_joints，关键点对应关系，[10,13]代表第10号点与第13号点为对应，[0]代表为独立，没有与其对应的关键点（coco比赛具体关键点分布可见下图）
 - num_joints ，关键点的个数，此时为18，若你自己与其不一样的话，就得修改过来。
 - snapshot_prefix
   为模型保存的名称+保存路径，我定义了保存当前目录下，最后生成的名字为：`snapshot-100.meta`

![这里写图片描述](http://img.blog.csdn.net/20170908144238610?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaGFwcHlob3Jpemlvbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------


### 4、训练模型

```
$ cd models/coco/train/
$ TF_CUDNN_USE_AUTOTUNE=0 CUDA_VISIBLE_DEVICES=0 python3 ../../../train.py
```
主要是运行`train.py`，该.py文件在`models/coco/train/`文件下，找到`pose_cfg.yaml`文件（名字不可以改变，code只认这个名字）并载入。


----------


### 注意点一则
You don't have to crop images such that they all have the same size,
as training is done with `batch_size=1` (batch size larger than 1 is
currently not supported anyway).

