# Human Pose Estimation with TensorFlow

![](images/teaser.png)

## 2018-3-19更新日志

 - 主要更新：自己训练数据模块更新：[SelfTraining.md](https://github.com/mattzheng/pose-tensorflow-detailed/blob/master/models/SelfTraining.md)
 - 其中：更新一版训练数据集生成demo函数一则（[dataset_generator.py](https://github.com/mattzheng/pose-tensorflow-detailed/blob/master/models/dataset_generator.py)）
 - 类似项目一则解读 [galaxy-fangfang/AI-challenger-pose-estimation](https://github.com/galaxy-fangfang/AI-challenger-pose-estimation)

> 这大兄弟在玩 [AI challenger](https://challenger.ai/)人体骨骼关节点赛题的时候，同样自己训练并开源出来。
> 但是，该比赛的关键点只有：14个（[参考：赛题与数据](https://challenger.ai/competition/keypoint/subject)），该作者在生成时候（[ai2coco_art_neckhead_json.py](https://github.com/galaxy-fangfang/AI-challenger-pose-estimation/blob/master/ai2coco_art_neckhead_json.py)），拼凑成17个点，与coco一致，然后就可以完全使用coco框架训练（多人模式），同时共享pairwise  stat。
> 该作者在比赛数据上当时迭代了60W次，最终的得分为:0.36，而原来的coco数据集，多人关键点定位需要180W次。

```
# right_eye
cocokey[6:9] = [0, 0, 0]
# left_ear
cocokey[9:12] = [0, 0, 0]
# right_ear
cocokey[12:15] = [0, 0, 0]
```


----------


Here you can find the implementation of the Human Body Pose Estimation algorithm,
presented in the [ArtTrack](http://arxiv.org/abs/1612.01465) and [DeeperCut](http://arxiv.org/abs/1605.03170) papers:

**Eldar Insafutdinov, Leonid Pishchulin, Bjoern Andres, Mykhaylo Andriluka and Bernt Schiele
DeeperCut:  A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model.
In _European Conference on Computer Vision (ECCV)_, 2016**

**Eldar Insafutdinov, Mykhaylo Andriluka, Leonid Pishchulin, Siyu Tang, Evgeny Levinkov, Bjoern Andres and Bernt Schiele
ArtTrack: Articulated Multi-person Tracking in the Wild.
In _Conference on Computer Vision and Pattern Recognition (CVPR)_, 2017**

For more information visit http://pose.mpi-inf.mpg.de

Python 3 is required to run this code.
First of all, you should install TensorFlow as described in the
[official documentation](https://www.tensorflow.org/install/).
We recommended to use `virtualenv`.

You will also need to install the following Python packages:

```
$ pip3 install scipy scikit-image matplotlib pyyaml easydict cython munkres
```

When running training or prediction scripts, please make sure to set the environment variable
`TF_CUDNN_USE_AUTOTUNE` to 0 (see [this ticket](https://github.com/tensorflow/tensorflow/issues/5048)
for explanation).

If your machine has multiple GPUs, you can select which GPU you want to run on
by setting the environment variable, eg. `CUDA_VISIBLE_DEVICES=0`.


## Training models

 - coco（多人场景）/MPII（单人场景）数据集训练教程：[README.md](https://github.com/mattzheng/pose-tensorflow-detailed/blob/master/models/README.md)
 - 自己数据集训练教程：
   [SelfTraining.md](https://github.com/mattzheng/pose-tensorflow-detailed/blob/master/models/SelfTraining.md)



## Demo code

Single-Person (if there is only one person in the image)

```
# Download pre-trained model files
$ cd models/mpii
$ ./download_models.sh
$ cd -

# Run demo of single person pose estimation
$ TF_CUDNN_USE_AUTOTUNE=0 python3 demo/singleperson.py
```

Multiple People

```
# Compile dependencies
$ ./compile.sh

# Download pre-trained model files
$ cd models/coco
$ ./download_models.sh
$ cd -

# Run demo of multi person pose estimation
$ TF_CUDNN_USE_AUTOTUNE=0 python3 demo/demo_multiperson.py
```


## Citation
Please cite ArtTrack and DeeperCut in your publications if it helps your research:

    @inproceedings{insafutdinov2017cvpr,
	    title = {ArtTrack: Articulated Multi-person Tracking in the Wild},
	    booktitle = {CVPR'17},
	    url = {http://arxiv.org/abs/1612.01465},
	    author = {Eldar Insafutdinov and Mykhaylo Andriluka and Leonid Pishchulin and Siyu Tang and Evgeny Levinkov and Bjoern Andres and Bernt Schiele}
    }

    @article{insafutdinov2016eccv,
        title = {DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model},
	    booktitle = {ECCV'16},
        url = {http://arxiv.org/abs/1605.03170},
        author = {Eldar Insafutdinov and Leonid Pishchulin and Bjoern Andres and Mykhaylo Andriluka and Bernt Schiele}
    }

