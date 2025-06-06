# Installation

## Requirements

We use an evironment with the following specifications, packages and dependencies:

- Python 3.9.21
- [PyTorch v2.3.1](https://pytorch.org/get-started/previous-versions/)
- [Torchvision v0.18.1](https://pytorch.org/get-started/previous-versions/)
- [Detectron2 v0.6](https://github.com/facebookresearch/detectron2/releases/tag/v0.6)
- [NATTEN v0.17.1](https://github.com/SHI-Labs/NATTEN/releases/tag/v0.17.1)
- [OneFormer Commit 4962ef6](https://github.com/SHI-Labs/OneFormer/tree/4962ef6a96ffb76a76771bfa3e8b3587f209752b)

## Setup Instructions

- Create a conda environment
  
  ```bash
  conda create --name dgfusion python=3.9 -y
  conda activate dgfusion
  ```

- Install packages and other dependencies.

  ```bash
  git clone https://github.com/timbroed/DGFusion.git
  cd dgfusion

  # Install Pytorch
  conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=11.8 -c pytorch -c nvidia

  # Install opencv
  pip3 install -U opencv-python

  # Install detectron2
  python tools/setup_detectron2.py
  pip install -e detectron2

  # Include OneFormer
  python tools/setup_oneformer.py

  # Include CAFuser
  python tools/setup_cafuser.py

  # Install other dependencies
  pip3 install git+https://github.com/timbroed/MUSES.git
  pip3 install git+https://github.com/cocodataset/panopticapi.git
  pip3 install git+https://github.com/mcordts/cityscapesScripts.git
  pip3 install -r requirements.txt
  ```

- Setup wandb.

  ```bash
  # Setup wand
  pip3 install wandb
  wandb login
  ```

- Setup CUDA Kernel for MSDeformAttn. `CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

  ```bash
  # Setup MSDeformAttn
  cd OneFormer/oneformer/modeling/pixel_decoder/ops
  sh make.sh
  cd ../../../../..
  ```
  
