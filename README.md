# CSF-Net: Context-Semantic Fusion Network for Large Mask Inpainting

[![WACV 2026](https://img.shields.io/badge/WACV-2026-blue)](https://wacv2026.thecvf.com/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./WACV26_CSF_Net__Context_Semantic_Fusion_Network_for_Large_Mask_Inpainting_Camera_Ready.pdf)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)

> **CSF-Net: Context-Semantic Fusion Network for Large Mask Inpainting**
> Chae-Yeon Heo and Yeong-Jun Cho*
> Department of Artificial Intelligence Convergence, Chonnam National University, South Korea
> WACV 2026

## 📖 Abstract

In this paper, we propose a semantic-guided framework to address the challenging problem of large-mask image inpainting, where essential visual content is missing and contextual cues are limited. To compensate for the limited context, we leverage a pretrained Amodal Completion (AC) model to generate structure-aware candidates that serve as semantic priors for the missing regions. We introduce **Context-Semantic Fusion Network (CSF-Net)**, a transformer-based fusion framework that fuses these candidates with contextual features to produce a semantic guidance image for image inpainting.

## 🎯 Key Features

- **Semantic Guidance**: Leverages amodal completion to generate structure-aware candidates
- **Dual-Encoder Architecture**: Separately processes context and semantic information
- **Fusion Decoder**: Cross-attention based fusion for multi-scale feature integration
- **Hierarchical Pixel Selection**: SSN (Structure Score Network) + PSN (Perceptual Score Network)
- **Plug-and-Play**: Seamlessly integrates into existing inpainting models without architectural changes

## 🏗️ Architecture Overview

```
┌─────────────┐
│   I_mask    │ (Masked Image)
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       │              │              │
┌──────▼──────┐ ┌────▼─────┐  ┌─────▼──────┐
│  Context    │ │ Semantic │  │  Amodal    │
│  Encoder    │ │ Encoder  │  │ Completion │
└──────┬──────┘ └────┬─────┘  └─────┬──────┘
       │              │              │
       └──────┬───────┴──────────────┘
              │
       ┌──────▼──────┐
       │   Fusion    │
       │   Decoder   │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │ Hierarchical│
       │   Pixel     │
       │  Selection  │
       │ (SSN + PSN) │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │   I_guide   │ (Semantic Guidance)
       └─────────────┘
              │
       ┌──────▼──────┐
       │  Inpainting │
       │    Model    │
       └──────┬──────┘
              │
       ┌──────▼──────┐
       │ Final Result│
       └─────────────┘
```

## 📊 Main Results

### Quantitative Results on Places365

| Method | Center Box (80%) |  | Center Box (50%) |  | Random (50-80%) |  |
|--------|---------|--------|---------|--------|---------|--------|
|        | FID↓    | LPIPS↓ | FID↓    | LPIPS↓ | FID↓    | LPIPS↓ |
| ASUKA  | 10.10   | 0.377  | 4.408   | 0.143  | 5.835   | 0.332  |
| **CSF-Net + ASUKA** | **9.434** | **0.332** | **3.612** | **0.105** | **5.324** | **0.325** |

CSF-Net consistently improves performance across all baseline models (LaMa, MAT, DLID, ASUKA) and masking conditions.

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

## 📁 Project Structure

```
CSF-Net/
├── models/
│   ├── aft_network.py              # Main CSF-Net architecture
│   └── components/
│       ├── fusion_attention.py     # Fusion Decoder
│       ├── region_based_selector.py # Hierarchical Pixel Selection
│       └── swin_transformer_modules.py # Swin Transformer components
├── data_utils/
│   ├── aft_dataset.py              # Dataset loader
│   └── candidate_processing.py     # Candidate filtering & scoring
├── losses/
│   ├── aft_losses.py               # CSF-Net losses
│   └── vgg_perceptual_loss.py      # VGG perceptual loss
├── configs/
│   └── aft_config.yaml             # Configuration file
├── train_aft.py                    # Training script
└── inference_aft.py                # Inference script
```

## 🔧 Configuration

Key parameters in `configs/aft_config.yaml`:

```yaml
data:
  Kmax: 3  # Number of semantic candidates (P in paper)

model:
  fusion_method: "region_based"
  region_based_selector:
    region_scales: [32, 16, 8, 4, 2]  # Multi-scale hierarchical selection
    final_region_size: 1  # Pixel-level precision

loss_weights:
  l1_masked_weight: 1.0
  perceptual_weight: 0.1
  region_consistency_weight: 0.15
```

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{heo2026csf,
  title={CSF-Net: Context-Semantic Fusion Network for Large Mask Inpainting},
  author={Heo, Chae-Yeon and Cho, Yeong-Jun},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```

## 🙏 Acknowledgements

This work was supported in part by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2025-24683045).

We thank the authors of:
- [Pix2Gestalt](https://github.com/cvlab-columbia/pix2gestalt) for amodal completion
- [ASUKA](https://github.com/kodenii/ASUKA-pytorch) for baseline inpainting model
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) for backbone architecture

## 📧 Contact

For questions and discussions, please contact:
- Chae-Yeon Heo: cyheo001@jnu.ac.kr
- Yeong-Jun Cho: yj.cho@jnu.ac.kr

## 📜 License

This project is released under the MIT License.

---

**Note**: This is the official implementation of CSF-Net (WACV 2026).
