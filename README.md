# DGFusion: Depth-Guided Sensor Fusion for Robust Semantic Perception

**by [Tim Broedermann](https://people.ee.ethz.ch/~timbr/), [Christos Sakaridis](https://people.ee.ethz.ch/csakarid/), Luigi Piccinelli, Wim Abbeloos, and [Luc Van Gool](https://scholar.google.de/citations?user=TwMib_QAAAAJ&hl=en)**

## Overview

This repository contains the official code for **DGFusion**, a novel depth-guided multimodal fusion method for robust semantic perception. DGFusion enhances condition-aware fusion by integrating depth information, treating multimodal segmentation as a multi-task problem. It utilizes lidar measurements both as model inputs and as ground truth for learning depth, with an auxiliary depth head that learns depth-aware features. These features are encoded into spatially-varying local depth tokens, which, together with a global condition token, dynamically adapt sensor fusion to the spatially varying reliability of each sensor across the scene. Additionally, DGFusion introduces a robust loss for depth learning, addressing the challenges of sparse and noisy lidar inputs in adverse conditions. Our method achieves state-of-the-art panoptic and semantic segmentation performance on the challenging [MUSES](https://muses.vision.ee.ethz.ch/) and [DeLiVER](https://github.com/jamycheung/DELIVER) datasets.

[//]: # (![DGFusion Overview Figure]&#40;resources/dgfusion_teaser.png&#41;)


### Contents

1. [Installation](#installation)
2. [Prepare Datasets](#prepare-datasets)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Citation](#citation)
5. [Acknowledgments](#acknowledgments)


### Installation

- We use Python 3.9.18, PyTorch 2.3.1, and CUDA 11.8.
- We use Detectron2-v0.6.
- For complete installation instructions, please see [INSTALL.md](INSTALL.md).

### Prepare Datasets

- DGFusion support two datasets: [MUSES](https://muses.vision.ee.ethz.ch/download) and [DeLiVER](https://github.com/jamycheung/DELIVER). The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`. Under this directory, detectron2 will look for datasets in the structure described below, if needed.

```text
$DETECTRON2_DATASETS/
    muses/
    deliver/
```

- You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`. If left unset, the default is `./datasets` relative to your current working directory.
- For more details on how to prepare the datasets, please see [detectron2's documentation](https://detectron2.readthedocs.io/tutorials/datasets.html).

[MUSES](https://muses.vision.ee.ethz.ch/download) dataset structure:

You need to dowload the following packages from the MUSES dataset:
- RGB_Frame_Camera_trainvaltest
- Panoptic_Annotations_trainval
- Semantic_Annotations_trainval
- Event_Camera_trainvaltest
- Lidar_trainvaltest
- Radar_trainvaltest
- GNSS_trainvaltest

and place them in the following structure:

```text
$DETECTRON2_DATASETS/
    muses/
        calib.json
        gt_panoptic/
        frame_camera/
        lidar/
        radar/
        event_camera/
        gnss/
```

[DeLiVER](https://github.com/jamycheung/DELIVER) dataset structure:

You can download the DeLiVER dataset from the following [link](https://drive.google.com/file/d/1P-glCmr-iFSYrzCfNawgVI9qKWfP94pm/view?usp=share_link) and place it in the following structure:

```text
$DETECTRON2_DATASETS/
    deliver/
        semantic/
        img/
        lidar/
        event/
        hha/
        depth/
```

### Evaluation


- We provide [weights](https://drive.google.com/drive/folders/13jBDPWHS7KIhi6PkmVd61eMje1ak74r1?usp=drive_link) for DGFusion trained on MUSES and DeLiVER datasets.

- With the flag `MODEL.TEST.DEPTH_ON`, you can chose weather the depth is predicted during testing time or not.

- To evaluate a model's performance, use:


MUSES (on the validation set):

```bash

python train_net.py \

    --config-file configs/muses/swin/dgfusion_swin_tiny_bs8_180k_muses_clre.yaml \

    --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS <path-to-checkpoint> \

    DATASETS.TEST_PANOPTIC "('muses_panoptic_val',)" \

    MODEL.TEST.PANOPTIC_ON True MODEL.TEST.SEMANTIC_ON True \

    MODEL.TEST.DEPTH_ON False

```


Predict on the test set to upload to the [MUSES benchmark](https://muses.vision.ee.ethz.ch/benchmarks#panopticSegmentation) for both semantic and panoptic segmentation:

```bash

python train_net.py \

    --config-file configs/muses/swin/dgfusion_swin_tiny_bs8_180k_muses_clre.yaml \

    ----inference-only MODEL.IS_TRAIN False MODEL.WEIGHTS <path-to-checkpoint> \

    OUTPUT_DIR output/dgfusion_swin_tiny_bs8_200k_deliver_clde \

    DATASETS.TEST_PANOPTIC "('muses_panoptic_test',)" \

    MODEL.TEST.PANOPTIC_ON True MODEL.TEST.SEMANTIC_ON True \

    MODEL.TEST.DEPTH_ON False

```


This will create folders under `<OUTPUT_DIR>/inference` for the semantic and panoptic predictions (e.g. `output/dgfusion_swin_tiny_bs8_200k_deliver_clde/inference/...`).

- For the panoptic predictions, you can zip the `labelIds` folder under the `panoptic` folder and upload it to the MUSES benchmark.

- For the semantic predictions, you can zip the `labelTrainIds` folder under the `semantic` folder and upload it to the MUSES benchmark. 


For better visualization you can further set `MODEL.TEST.SAVE_PREDICTIONS.CITYSCAPES_COLORS True` to get additional folders with the predictions in the cityscapes colors.


DeLiVER on the test set:

```bash

python train_net.py \

    --config-file configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml \

    --eval-only MODEL.IS_TRAIN False MODEL.WEIGHTS <path-to-checkpoint> \

    DATASETS.TEST_SEMANTIC "('deliver_semantic_test',)" \

    MODEL.TEST.DEPTH_ON False

```


Replace `deliver_semantic_test` with `deliver_semantic_val` to evaluate on the validation set.


### Results


We provide the following results for the MUSES dataset, with the testing score from the official [MUSES Benchmark](https://muses.vision.ee.ethz.ch/benchmarks#panopticSegmentation)


| Method | Backbone | PQ-val | mIoU-val  | PQ-test | mIoU-test |                                   config                                    | Checkpoint |
| :---:  | :---:    |:------:|:---------:|:-------:|:---------:|:---------------------------------------------------------------------------:| :---: |
DGFusion | Swin-T | 58.88  |   79.72   |  61.03   |   79.49    |   [config](configs/muses/swin/dgfusion_swin_tiny_bs8_180k_muses_clre.yaml)   | [model](https://drive.google.com/file/d/1akqG1TSariaPzkIoqbG7Y7JA0I6lIcP6/view?usp=drive_link) |



We provide the following results for the DeLiVER dataset:


| Method | Modalities | Backbone | mIoU-val | mIoU-test |                                     config                                      | Checkpoint |
| :---:  | :---:    | :---:    |:--------:|:---------:|:-------------------------------------------------------------------------------:| :---: |
DGFusion | CLDE | Swin-T |  66.51  |   56.71   |   [config](configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_clde.yaml)   | [model](https://drive.google.com/file/d/1oWD9qfJyv-nH9VdoFi2bjdYt6eZ37Vhu/view?usp=drive_link) |
DGFusion | CLE | Swin-T |   56.64   |   51.55   | [config](configs/deliver/swin/dgfusion_swin_tiny_bs8_200k_deliver_cle.yaml) | [model](https://drive.google.com/file/d/1sdKIyjqwkObQ0M0mTVXn5wwEuHMXb_RW/view?usp=drive_link) |


### Training

- We followed the general setup of [CAFuser](https://github.com/timbroed/CAFuser) and trained DGFusion using 4 NVIDIA TITAN RTX GPUs with 24GB memory each. However, we do not include the training code in this project.


### Citation


If you find this project useful in your research, please consider citing:


```

@article{broedermann2025dgfusion,

  author={Br{\"o}dermann, Tim and Sakaridis, Christos and Piccinelli, Luigi and Van Gool, Luc},

  title={DGFusion: Depth-Guided Sensor Fusion for Robust Semantic Perception}, 

  journal   = {arXiv preprint arXiv:2505.xxxxx},

  year      = {2025}
  
  }

```

### License:

This work is licensed under the Creative Commons Attribution-Non Commercial ShareAlike 4.0 International License. To view a copy of this license, visit [Legal Code - Attribution-NonCommercial-ShareAlike 4.0 International - Creative Commons](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)

### Acknowledgments

This project is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [CAFuser](https://github.com/timbroed/CAFuser)
* [OneFormer](https://github.com/SHI-Labs/OneFormer)
* [Mask2Former](https://github.com/facebookresearch/Mask2Former)
* [GroupViT](https://github.com/NVlabs/GroupViT) 
* [Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)
* [detectron2](https://github.com/facebookresearch/detectron2)
* [MUSES SDK](https://github.com/timbroed/MUSES)
* [HRFuser](https://github.com/timbroed/HRFuser)

This work is funded by Toyota Motor Europe via the research project [TRACE-Zurich](https://trace.ethz.ch/) (Toyota Research on Automated Cars Europe).

