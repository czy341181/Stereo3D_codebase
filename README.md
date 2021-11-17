# LIGA-Stereo

This is the unofficial implementation of the paper LIGA-Stereo for personal study.
This codebase is based on https://github.com/xinzhuma/monodle.

## Usage

### Installation
This repo is tested on our local environment (python=3.7, cuda=10.1, pytorch=1.6)

```bash
conda create -n LIGA python=3.7
```
Then, activate the environment:
```bash
conda activate LIGA
```

Install PyTorch:

```bash
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

and other  requirements:
```bash
pip install -r requirements.txt

python setup.py develop
```

Install modified mmdetection from [`[mmdetection_kitti]`](https://github.com/xy-guo/mmdetection_kitti)
```shell
git clone https://github.com/xy-guo/mmdetection_kitti

pip install mmcv-full==1.2.1 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

python setup.py develop
```


### Data Preparation
Please download [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize the data as follows:

```
#ROOT
  |data/
    |KITTI/
      |ImageSets/ [already provided in this repo]
      |object/			
        |training/
          |calib/
          |image_2/
          |image_3/
          |velodyne/
          |label/
        |testing/
          |calib/
          |image_2/
          |image_3/
```

### Training

Move to the workplace and train the network:

```sh
 cd #ROOT
 cd experiments/example
 CUDA_VISIBLE_DEVICES=0,1 bash ./dist_train.sh 2 kitti_example.yaml
```

### Plan
We want to build a codebase based on the image voxel method (DSGN(stereo), CaDDN(monocular), SECOND(LiDAR)).

The implementation of CaDDN is coming soon.

The implementation of Voxel-based code-based is coming soom.

