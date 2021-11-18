# Stereo-codebased

## Usage

### Installation
This repo is tested on our local environment (python=3.7, cuda=10.1, pytorch=1.4)

```bash
conda create -n Stereo python=3.7
```
Then, activate the environment:
```bash
conda activate Stereo
```

Install PyTorch:

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

and other  requirements:
```bash
pip install -r requirements.txt

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
CUDA_VISIBLE_DEVICES=0,1 python ../../tools/train_val.py --config kitti_example.yaml
```



