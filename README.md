# This repo comp


## Requirement
**Packages:**
* Python 3.8
* Pytorch 1.7
* numpy
* tqdm
* pandas
* yaml

## File Structure
```
ISTD_project
├─ train.py
├─ predict.py
├─ dataset
│  ├─ training
│  └─ val
├─ models
│  └─ ULite.py
└─ dataloader.py

```
## Dataset Preparation 
* MDvsFA-cGAN Dataset [**[Dataset]**](https://github.com/wanghuanphd/MDvsFA_cGAN)
[**[Paper]**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.pdf)

## Training
Experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with a single GeForce RTX 2080Ti GPU of 12 GB Memory.

Train from scratch
```
python train.py 
```

## Testing
Use pretrained model for testing
```
python predict.py
```

