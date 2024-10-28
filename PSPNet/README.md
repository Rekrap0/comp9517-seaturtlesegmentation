# Sea Turtle Segmentation - PSPNet

## Setup Environment

Create and activate conda environment

```
conda create --name PSPNet python=3.9 
conda activate PSPNet
```

Install torch 
```
pip3 install install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Install openmim and use mim to install mmcv
```
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0
```

Install other depdencencies
```
pip install opencv-python pillow matplotlib seaborn tqdm pytorch-lightning 
pip install mmdet>=3.1.0
```
