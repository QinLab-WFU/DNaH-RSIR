# Deep Neighbor-aware Hashing with Global-Local Representation for Multi-Label Remote Sensing Image Retrieval
This paper is accepted for publication with Signal Processing: Image Communication.
# Usage
### Install dependencies:
We use python to build our code, you need to install those package to run
- Python 3.9.7
- Pytorch 1.12.1
- torchvision 13.1
- CUDA 11.3

### Processing dataset
You should download three data sets including [UC Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html), 
[AID](https://captain-whu.github.io/AID/), and MLRSNet and put data set in the corresponding directory under `data`. 
If you want to construct your own training set and testing set, you should modify the path of images of training set, 
testing set and database respectively in the `data/.../train.txt` , `data/.../test.txt` and `data/.../database.txt` for corresponding data set.

## Training
All parameters are defined in the train.py file, and the test methods are integrated into the train.py file. Therefore, it is only necessary to run
python train.py

### Download pretrained model
Pretrained model is resnet50.pth.
