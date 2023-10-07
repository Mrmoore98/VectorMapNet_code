# Set up Environment 

### Note

If you have trouble in `pip install`, try add `-i https://pypi.tuna.tsinghua.edu.cn/simple` to  the command.

### Create conda environment

```
conda create --name hdmap-opensource python==3.8
conda activate hdmap-opensource
```

### Install PyTorch

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### Install MMCV-series

We build our code on open-mmlab. So mmcv series all required.

```
# Install mmcv-series
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

### Install mmdetection3d

Currently we are using mmdetection3d of  version 0.17.x . To install mmdet3d, please first download the releases of 0.17.x from <https://github.com/open-mmlab/mmdetection3d/releases>, unzip the code and rename the folder to `mmdetection3d`. Then run

```
wget https://github.com/open-mmlab/mmdetection3d/archive/refs/tags/v0.17.3.zip
unzip v0.17.3.zip
cd  mmdetection3d-0.17.3
```

```
cd mmdetection3d
pip install -v -e .
```

to install mmdetection3d. Note that some installations above requires CUDA environment, make sure add `export CUDA_HOME=/usr/local/cuda` to your bash source file.

For more details about installation, please refer to open-mmlab <https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md>.

### Other requirements

Run

```
pip install -r requirements.txt
```

to install all requirements.
