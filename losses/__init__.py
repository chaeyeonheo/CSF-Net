# losses/__init__.py

from .csf_losses import CSFLoss
from .vgg_perceptual_loss import VGGPerceptualLoss # 새로 추가될 파일

from .csf_losses import RegionBasedCSFLoss
from .csf_losses import CSFLoss

__all__ = [
    'CSFLoss',
    'VGGPerceptualLoss'
]