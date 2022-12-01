# VectorMapNet_code
**VectorMapNet: End-to-end Vectorized HD Map Learning**

This is the official codebase of VectorMapNet


[Yicheng Liu](https://scholar.google.com/citations?user=vRmsgQUAAAAJ&hl=zh-CN), Yuantian Yuan, [Yue Wang](https://people.csail.mit.edu/yuewang/), [Yilun Wang](https://scholar.google.com.hk/citations?user=nUyTDosAAAAJ&hl=en/), [Hang Zhao](http://people.csail.mit.edu/hangzhao/)


**[[Paper](https://arxiv.org/pdf/2206.08920.pdf)] [[Project Page](https://tsinghua-mars-lab.github.io/vectormapnet/)]**

**Abstract:**
Autonomous driving systems require a good understanding of surrounding environments, including moving obstacles and static High-Definition (HD) semantic maps. Existing methods approach the semantic map problem by offline manual annotations, which suffer from serious scalability issues. More recent learning-based methods produce dense rasterized segmentation predictions which do not include instance information of individual map elements and require heuristic post-processing that involves many hand-designed components, to obtain vectorized maps. To that end, we introduce an end-to-end vectorized HD map learning pipeline, termed VectorMapNet. VectorMapNet takes onboard sensor observations and predicts a sparse set of polylines primitives in the bird's-eye view to model the geometry of HD maps. Based on this pipeline, our method can explicitly model the spatial relation between map elements and generate vectorized maps that are friendly for downstream autonomous driving tasks without the need for post-processing. In our experiments, VectorMapNet achieves strong HD map learning performance on nuScenes dataset, surpassing previous state-of-the-art methods by 14.2 mAP. Qualitatively, we also show that VectorMapNet is capable of generating comprehensive maps and capturing more fine-grained details of road geometry. To the best of our knowledge, VectorMapNet is the first work designed toward end-to-end vectorized HD map learning problems.

**Questions/Requests:** 
Please file an [issue](https://github.com/Tsinghua-MARS-Lab/vecmapnet/issues) or send an email to [Yicheng](moooooore66@gmail.com).


## Bibtex
If you found this paper or codebase useful, please cite our paper:
```
@article{liu2022vectormapnet,
    title={VectorMapNet: End-to-end Vectorized HD Map Learning},
    author={Liu, Yicheng and Wang, Yue and Wang, Yilun and Zhao, Hang},
    journal={arXiv preprint arXiv:2206.08920},
    year={2022}
    }
```


# Run VectorMapNet

## Note


## 0. Environment

Set up environment by following this [script](env.md)

## 1. Prepare your dataset

Store your data with following structure:

```
    root
        |--datasets
            |--nuScenes
            |--Argoverse2(optional)

```

### 1.1 Generate annotation files

#### Preprocess nuScenes

```
python tools/data_converter/nuscenes_converter.py --data-root your/dataset/nuScenes/
```

## 2. Evaluate VectorMapNet

### Download Checkpoint
| Method       | Modality    | Config | Checkpoint |
|--------------|-------------|--------|------------|
| VectorMapNet | Camera only | [config](configs/vectormapnet.py) | [model link](https://drive.google.com/file/d/1ccrlZ2HrFfpBB27kC9DkwCYWlTUpgmin/view?usp=sharing)      |


### Train VectorMapNet

In single GPU
```
python tools/train.py configs/vectormapnet.py
```

For multi GPUs
```
bash tools/dist_train.sh configs/vectormapnet.py $num_gpu
```


### Do Evaluation

In single GPU
```
python tools/test.py configs/vectormapnet.py /path/to/ckpt --eval name
```

For multi GPUs
```
bash tools/dist_test.sh configs/vectormapnet.py /path/to/ckpt $num_gpu --eval name
```


### Expected Results

| $AP_{ped}$   | $AP_{divider}$ | $AP_{boundary}$ | mAP   |
|--------------|----------------|-----------------|-------|
| 39.8 | 47.7    | 38.8          | 42.1 |


