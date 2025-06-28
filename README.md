# LightBSR: Towards Lightweight Blind Super-Resolution via Discriminative Implicit Degradation Representation Learning

Created by [Jiang Yuan](https://github.com/Fieldhunter), [Ji Ma](https://github.com/MJ-NCEPU), [Bo Wang](https://github.com/wangbo2016), [Guanzhou Ke](https://github.com/Guanzhou-Ke), Weiming Hu

This repository contains PyTorch implementation for LightBSR: Towards Lightweight Blind Super-Resolution via Discriminative Implicit Degradation Representation Learning (Accepted by ICCV 2025).

## Environment
`pip install -r requirements.txt`

## Train
### 1. Prepare data 
#### 1.1 Training data
1.1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  dataset and the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.

1.1.2 Combine the HR images from these two datasets in `./datasets/DF2K/HR` to build the DF2K dataset. 

#### 1.2 Testing data
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in `./datasets/benchmark`.

### 2. Train
#### 2.1 Teacher model
2.1.1 Replace all code related to degradation settings in the specified files to ensure they match the training degradation settings (All locations requiring changes are marked with TODO) :
`./model/teacher.py` `./teacher_main.sh` `./teacher_trainer.py`

2.1.2 training: `bash teacher_main.sh`

#### 2.2 Student model
2.2.1 Replace all code related to degradation settings in the specified files to ensure they match the training degradation settings (All locations requiring changes are marked with TODO) :
`./student_main.sh` `./student_option.py` `./student_trainer.py`

2.2.2 training: `bash student_main.sh`

### 3. Test model
Select the corresponding degradation condition parameters and perform testing: `bash student_test.sh`

## Citation
```

```
