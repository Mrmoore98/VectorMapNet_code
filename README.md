# VectorMapNet_code
**VectorMapNet: End-to-end Vectorized HD Map Learning**

This is the offical codebase of VectorMapNet


[Liu Yicheng](https://scholar.google.com/citations?user=vRmsgQUAAAAJ&hl=zh-CN), [Yue Wang](https://people.csail.mit.edu/yuewang/), [Yilun Wang](https://scholar.google.com.hk/citations?user=nUyTDosAAAAJ&hl=en/), [Hang Zhao](http://people.csail.mit.edu/hangzhao/)

**[[Paper](https://arxiv.org/submit/4361297/view)] [[Project Page](https://tsinghua-mars-lab.github.io/vecmapnet/)]**

**Abstract:**
Autonomous driving systems require a good understanding of surrounding environments, including moving obstacles and static High-Definition (HD) semantic maps. Existing methods approach the semantic map problem by offline manual annotations, which suffer from serious scalability issues. More recent learning-based methods produce dense rasterized segmentation predictions which do not include instance information of individual map elements and require heuristic post-processing that involves many hand-designed components, to obtain vectorized maps. To that end, we introduce an end-to-end vectorized HD map learning pipeline, termed VectorMapNet. VectorMapNet takes onboard sensor observations and predicts a sparse set of polylines primitives in the bird's-eye view to model the geometry of HD maps. Based on this pipeline, our method can explicitly model the spatial relation between map elements and generate vectorized maps that are friendly for downstream autonomous driving tasks without the need for post-processing. In our experiments, VectorMapNet achieves strong HD map learning performance on nuScenes dataset, surpassing previous state-of-the-art methods by 14.2 mAP. Qualitatively, we also show that VectorMapNet is capable of generating comprehensive maps and capturing more fine-grained details of road geometry. To the best of our knowledge, VectorMapNet is the first work designed toward end-to-end vectorized HD map learning problems.

**Questions/Requests:** 
Please file an [issue](https://github.com/Tsinghua-MARS-Lab/vecmapnet/issues) or send an email to [Yicheng](moooooore66@gmail.com).
