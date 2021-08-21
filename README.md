## REN
Code for paper "Robust Ensembling Network for Unsupervised Domain Adaptation", by Han Sun, Lei Lin, Ningzhong Liu, and Huiyu Zhou.

#### Prerequisites
- PyTorch >= 0.4.0 (with suitable CUDA and CuDNN version)
- torchvision >= 0.2.1
- Python3
- Numpy
- argparse
- PIL
#### Dataset
##### Office-31 (https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)

##### Office-Home (http://hemanthdv.org/OfficeHome-Dataset/)

##### VisDA-2017 (https://github.com/VisionLearningGroup/taskcv-2017-public)

##### Image-clef (https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view)

#### Training

Office-31
```
sh bash/office31.sh
```
Office-Home

```
sh bash/office-home.sh
```
VisDA 2017
```
sh bash/visda.sh
```
Image-clef
```
sh bash/im-clef.sh
```
#### Testing
```
sh bash/test_gpu0.sh
```
