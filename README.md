# CSF-Net: Context-Semantic Fusion Network for Large Mask Inpainting (WACV 2026)

[![WACV 2026]([https://img.shields.io/badge/WACV-2026-00008B)](https://wacv2026.thecvf.com/](https://openaccess.thecvf.com/content/WACV2026/papers/Heo_CSF-Net_Context-Semantic_Fusion_Network_for_Large_Mask_Inpainting_WACV_2026_paper.pdf))
[![arXiv](https://img.shields.io/badge/arXiv-2511.07987-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2511.07987)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)

> **CSF-Net: Context-Semantic Fusion Network for Large Mask Inpainting**


> Chae-Yeon Heo and Yeong-Jun Cho*


> Department of Artificial Intelligence Convergence, Chonnam National University, South Korea


## 📖 Abstract

In this paper, we propose a semantic-guided framework to address the challenging problem of large-mask image inpainting, where essential visual content is missing and contextual cues are limited. To compensate for the limited context, we leverage a pretrained Amodal Completion (AC) model to generate structure-aware candidates that serve as semantic priors for the missing regions. We introduce **Context-Semantic Fusion Network (CSF-Net)**, a transformer-based fusion framework that fuses these candidates with contextual features to produce a semantic guidance image for image inpainting.

## 🎯 Key Features

- **Semantic Guidance**: Leverages amodal completion to generate structure-aware candidates
- **Dual-Encoder Architecture**: Separately processes context and semantic information
- **Fusion Decoder**: Cross-attention based fusion for multi-scale feature integration
- **Hierarchical Pixel Selection**: SSN (Structure Score Network) + PSN (Perceptual Score Network)
- **Plug-and-Play**: Seamlessly integrates into existing inpainting models without architectural changes

## 🚀 Getting Started

### Prerequisites

```bash
# Python >= 3.8
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/chaeyeonheo/CSF-Net.git
cd CSF-Net

# Download pretrained VGG weights
mkdir -p pretrained_model
wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth -P pretrained_model/
```

### Dataset Preparation

1. Download Places365 dataset
2. Prepare amodal completion candidates using [Pix2Gestalt](https://github.com/cvlab-columbia/pix2gestalt)
3. Update data paths in `configs/aft_config.yaml`

### Training

```bash
# Single GPU
python train_aft.py --config_path ./configs/aft_config.yaml

# Multi-GPU (DDP)
torchrun --standalone --nnodes=1 --nproc_per_node=4 train_aft.py \
    --config_path ./configs/aft_config.yaml
```

### Inference

```bash
python inference_aft.py \
    --config_path ./configs/aft_config.yaml \
    --checkpoint_path /path/to/checkpoint.pth \
    --input_dir /path/to/test/images \
    --output_base_dir ./results
```



## 📝 Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Heo_2026_WACV,
    author    = {Heo, Chae-Yeon and Cho, Yeong-Jun},
    title     = {CSF-Net: Context-Semantic Fusion Network for Large Mask Inpainting},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {March},
    year      = {2026},
    pages     = {8292-8301}
}
```

## 📧 Contact

For questions and discussions, please contact:
- Chae-Yeon Heo: cyheo001@jnu.ac.kr
- Yeong-Jun Cho: yj.cho@jnu.ac.kr

---

**Note**: This is the official implementation of CSF-Net (WACV 2026).

