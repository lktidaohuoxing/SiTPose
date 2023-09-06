### Usage
---
#### 1. Install requirements
##### create  conda envirment
```bash
conda create -n SiTPose python=3.7
conda activate SiTPose
```
##### install requirements
```bash
conda install pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=11.1 -c pytorch -c nvidia
pip3 install scipy==1.7.3
pip3 install opencv-python==4.6.0.66
pip3 install git+https://github.com/princeton-vl/lietorch.git
```


#### 2.Prepare data
- 7Scenes dataset can be obtained from [https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
```
SiTPose/
└── data/
    └── 7Scenes
        └── Chess
        └── Fire
        └── Heads
        └── Office
        └── Pumpkin
        └── RedKitchen
        └── Stairs
        └── generate_data.py
        └── db_all_med_hard_train.txt
        └── db_all_med_hard_valid.txt
```
**generate data**
```bash
cd data/7Scenes
python3 generate_data.py
```

#### 3. Training and Eval
```bash
#SiTPose
#start training
python3 train.py 

#eval
python eval.py

#SiTPose-light, the light version of SiTPose while maintaining the same level of accuracy as the original version.
#start training
python3 train_light.py 

#eval
python eval_light.py

```

### 4.Reference
[8-point](https://github.com/crockwell/rel_pose)

[CCT](https://github.com/SHI-Labs/Compact-Transformers)

[HomographyNet](https://github.com/richard-guinto/homographynet)

