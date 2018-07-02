Before training the models you need to download the ImageNet pre-trained ResNet weights

```
$ cd models/pretrained
$ ./download.sh
```

Training parameters are specified in the `pose_cfg.yaml` file.

Here are the dataset specific instructions. 

## Training a model with MPII Pose Dataset (Single Person)


1. Download the dataset from [this page](http://human-pose.mpi-inf.mpg.de/),
both images and annotations. Unpack it to the path `<path_to_dataset>`
to have the following directory structure:

```
<path_to_dataset>/images/*.jpg
<path_to_dataset>/mpii_human_pose_v1_u12_1.mat
```

2. Preprocess dataset (crop and rescale)

```
$ cd matlab/mpii
$ matlab -nodisplay -nosplash

# in matlab execute the following function, be sure to specify the *absolute* path
preprocess_single('<path_to_dataset>')
```

3. Edit the training definition file
`models/mpii/train/pose_cfg.yaml` such that:

```
dataset: `<path_to_dataset>/cropped/dataset.mat`
```

4. Train the model

```
$ cd models/mpii/train/
$ TF_CUDNN_USE_AUTOTUNE=0 CUDA_VISIBLE_DEVICES=0 python3 ../../../train.py
```

## Training a model with MS COCO dataset (Multi-Person)

1. Download [MS COCO](http://mscoco.org/dataset/#download)
train2014 set with keypoint and object instances annotations.

2. Download pairwise statistics:
```
$ cd models/coco
$ ./download_models.sh
```

3. Edit the training definition file
`models/coco/train/pose_cfg.yaml` such that:

```
dataset: `<path_to_mscoco>`
```

4. Train the model:

```
$ cd models/coco/train/
$ TF_CUDNN_USE_AUTOTUNE=0 CUDA_VISIBLE_DEVICES=0 python3 ../../../train.py
```
