# losses/vgg_perceptual_loss.py (수정)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer_indices, use_input_norm=True, requires_grad=False, 
                 vgg_weights_path=None, logger=print): 
        super().__init__()
        self.logger = logger # 로거 저장

        vgg19 = models.vgg19(weights=None) 

        if vgg_weights_path and os.path.exists(vgg_weights_path):
            if hasattr(self.logger, 'info'): self.logger.info(f"VGGFeatureExtractor: Loading VGG19 weights from local path: {vgg_weights_path}")
            else: print(f"INFO: VGGFeatureExtractor - Loading VGG19 weights from local path: {vgg_weights_path}")
            try:
                vgg19.load_state_dict(torch.load(vgg_weights_path, map_location='cpu'))
            except Exception as e:
                log_msg = f"VGGFeatureExtractor: Failed to load VGG19 weights from {vgg_weights_path}: {e}. Perceptual loss may not work."
                if hasattr(self.logger, 'error'): self.logger.error(log_msg)
                else: print(f"ERROR: {log_msg}")
        else: 
            log_msg_init = f"VGGFeatureExtractor: Local VGG19 weights path ('{vgg_weights_path}') not provided or not found. Attempting to download pretrained weights."
            if hasattr(self.logger, 'info'): self.logger.info(log_msg_init)
            else: print(f"INFO: {log_msg_init}")
            try:
                vgg19_pretrained = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
                vgg19.load_state_dict(vgg19_pretrained.state_dict())
                log_msg_succ = f"VGGFeatureExtractor: Successfully loaded/downloaded VGG19 pretrained weights."
                if hasattr(self.logger, 'info'): self.logger.info(log_msg_succ)
                else: print(f"INFO: {log_msg_succ}")
            except Exception as e_download:
                log_msg_err = f"VGGFeatureExtractor: Failed to download VGG19 pretrained weights: {e_download}. Perceptual loss may not work if weights are not loaded."
                if hasattr(self.logger, 'error'): self.logger.error(log_msg_err)
                else: print(f"ERROR: {log_msg_err}")
        # ... (이하 VGGFeatureExtractor의 나머지 코드는 이전 답변과 동일)
        self.features = nn.Sequential(*list(vgg19.features.children())[:max(feature_layer_indices) + 1])
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        if not requires_grad:
            for param in self.parameters(): param.requires_grad = False
        self.feature_layer_indices = feature_layer_indices

    def forward(self, x): # ... (이전 답변과 동일)
        if self.use_input_norm and hasattr(self, 'mean') and hasattr(self, 'std'):
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        extracted_features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layer_indices: extracted_features.append(x)
        return extracted_features

class VGGPerceptualLoss(nn.Module): # __init__에서 logger를 VGGFeatureExtractor로 전달
    def __init__(self, 
                 feature_layer_indices=(2, 7, 16, 25, 34), 
                 style_layer_indices=None,
                 loss_type='l1',
                 use_input_norm=True,
                 requires_grad=False,
                 style_weight=0.0,
                 vgg_weights_path=None,
                 logger=print): # logger 인자 기본값 print
        super().__init__()
        self.feature_extractor = VGGFeatureExtractor(
            feature_layer_indices, use_input_norm, requires_grad, vgg_weights_path, logger=logger # logger 전달
        )
        self.loss_type = loss_type.lower()
        if self.loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='mean')
        elif self.loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='mean')
        else:
            log_msg_err_type = f"Unsupported VGGPerceptualLoss loss type: {loss_type}. Defaulting to L1."
            if hasattr(logger, 'error'): logger.error(log_msg_err_type)
            else: print(f"ERROR: {log_msg_err_type}")
            self.criterion = nn.L1Loss(reduction='mean')

        self.style_layer_indices = style_layer_indices
        self.style_weight = style_weight
        if self.style_weight > 0 and self.style_layer_indices is None:
            self.style_layer_indices = feature_layer_indices 
            log_msg_warn_style = "Warning: style_layer_indices not provided for Style Loss, using feature_layer_indices instead."
            if hasattr(logger, 'warning'): logger.warning(log_msg_warn_style)
            else: print(log_msg_warn_style)
            
    # _gram_matrix, forward 메소드는 이전 답변과 동일
    def _gram_matrix(self, x): # ... (이전과 동일)
        B, C, H, W = x.size(); features = x.view(B, C, H * W)
        G = torch.bmm(features, features.transpose(1, 2)); return G.div(C * H * W) 


    # vgg_perceptual_loss.py의 VGGPerceptualLoss.forward 메서드 (line 98 근처)
    # 전체 메서드를 다음으로 교체:

# losses/vgg_perceptual_loss.py의 VGGPerceptualLoss.forward 메서드 (line 98 근처)
# 전체 메서드를 다음으로 교체:

    def forward(self, generated_img, target_img, mask=None):
        # 입력 이미지들을 [0, 1] 범위로 안정화
        gen_normalized = torch.clamp(generated_img, 0.0, 1.0)
        
        if target_img.min() < -0.1 or target_img.max() > 1.1:
            mean = torch.tensor([0.485, 0.456, 0.406], device=target_img.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=target_img.device).view(1, 3, 1, 1)
            target_normalized = target_img * std + mean
        else:
            target_normalized = target_img
        target_normalized = torch.clamp(target_normalized, 0.0, 1.0)
        
        # Feature 추출
        gen_features = self.feature_extractor(gen_normalized)
        target_features = self.feature_extractor(target_normalized)
        
        perceptual_loss = 0.0
        for gen_f, target_f in zip(gen_features, target_features):
            if mask is not None:
                mask_resized = F.interpolate(mask, size=gen_f.shape[2:], mode='nearest')
                fill_mask = 1.0 - mask_resized # 채워야 할 영역 (값이 1)

                # ================= 🎯 Loss Explosion 해결을 위한 핵심 수정 🎯 =================
                # torch.where를 사용하여 마스크 바깥 영역(fill_mask=0)의 특징을 동일하게 만들어
                # 해당 영역의 손실 기여도를 0으로 만듭니다. 이는 수치적으로 매우 안정적입니다.
                gen_f_masked = torch.where(fill_mask.bool(), gen_f, target_f)

                # 이제 마스크된 생성 이미지 특징과 원본 타겟 이미지 특징 간의 손실을 계산합니다.
                # `reduction='mean'`은 전체 텐서에 대해 평균을 내지만, 마스크 바깥 영역의 차이가
                # 0이므로, 사실상 마스크 안쪽 영역의 평균 손실이 계산됩니다.
                perceptual_loss += self.criterion(gen_f_masked, target_f)
                # ========================================================================
            else:
                perceptual_loss += self.criterion(gen_f, target_f)
        
        if len(gen_features) > 0:
            perceptual_loss = perceptual_loss / len(gen_features)
        
        content_loss = perceptual_loss

        # Style loss (기존과 동일)
        style_loss = torch.tensor(0.0, device=generated_img.device)
        if self.style_weight > 0 and self.style_layer_indices:
            style_features_gen = [f for i, f in enumerate(gen_features) if i in self.style_layer_indices]
            style_features_target = [f for i, f in enumerate(target_features) if i in self.style_layer_indices]
            
            for gen_f, target_f in zip(style_features_gen, style_features_target):
                gen_gram = self._gram_matrix(gen_f)
                target_gram = self._gram_matrix(target_f)
                style_loss += self.criterion(gen_gram, target_gram)
            
            if len(style_features_gen) > 0:
                style_loss = (style_loss / len(style_features_gen)) * self.style_weight
        
        return perceptual_loss, content_loss, style_loss