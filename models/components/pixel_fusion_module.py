# models/components/pixel_fusion_module.py (개선된 버전)
# 픽셀 혼합 문제를 해결한 선택적 융합 방식

import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelLevelFusionModule(nn.Module):
    """
    후보 이미지들의 실제 픽셀을 직접 융합하는 모듈
    - 방식1: 하드 선택 (가장 좋은 후보 하나만 선택)
    - 방식2: 소프트 선택 (온도 조절된 선택적 혼합)
    - 방식3: 영역별 선택 (공간적으로 다른 전략 사용)
    """
    
    def __init__(self, num_candidates, feature_dim=256, hidden_dim=128, fusion_strategy="adaptive"):
        super().__init__()
        self.num_candidates = num_candidates
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.fusion_strategy = fusion_strategy  # "hard", "soft", "adaptive"
        
        # 1. 후보별 품질 점수 예측 네트워크
        input_channels = 3 + 1 + 3 * num_candidates + feature_dim
        
        self.candidate_scorer = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_candidates, 3, padding=1)
            # Softmax는 나중에 적용 (온도 조절 위해)
        )
        
        # 2. 전체적인 신뢰도 예측
        self.confidence_predictor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 3. 융합 전략 선택기 (adaptive 모드용)
        if fusion_strategy == "adaptive":
            self.strategy_selector = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 2, 3, padding=1),  # [hard_weight, soft_weight]
                nn.Softmax(dim=1)
            )
        
        # 4. 온도 파라미터 (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # 5. 특징 기반 보조 예측기
        self.feature_to_pixel_net = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
    def hard_selection(self, candidate_images, selection_logits):
        """하드 선택: 가장 좋은 후보 하나만 선택"""
        B, K, C, H, W = candidate_images.shape
        
        # 가장 높은 점수의 후보 선택
        best_candidate_indices = torch.argmax(selection_logits, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 원-핫 마스크 생성
        one_hot_mask = torch.zeros_like(selection_logits)  # [B, K, H, W]
        one_hot_mask.scatter_(1, best_candidate_indices, 1.0)
        
        # 선택된 후보만 사용
        selected_pixels = (one_hot_mask.unsqueeze(2) * candidate_images).sum(dim=1)  # [B, 3, H, W]
        
        return selected_pixels, one_hot_mask
    
    def soft_selection(self, candidate_images, selection_logits, temperature):
        """소프트 선택: 온도 조절된 가중 평균"""
        B, K, C, H, W = candidate_images.shape
        
        # 온도 조절된 소프트맥스 (온도가 낮을수록 하드 선택에 가까워짐)
        selection_weights = F.softmax(selection_logits / temperature, dim=1)  # [B, K, H, W]
        
        # 가중 평균으로 후보 결합
        weighted_pixels = (selection_weights.unsqueeze(2) * candidate_images).sum(dim=1)  # [B, 3, H, W]
        
        return weighted_pixels, selection_weights
    
    def adaptive_selection(self, candidate_images, selection_logits, strategy_weights):
        """적응적 선택: 영역별로 하드/소프트 전략을 다르게 적용"""
        B, K, C, H, W = candidate_images.shape
        
        hard_weight = strategy_weights[:, 0:1, :, :]  # [B, 1, H, W]
        soft_weight = strategy_weights[:, 1:2, :, :]  # [B, 1, H, W]
        
        # 하드 선택 결과
        hard_pixels, hard_weights = self.hard_selection(candidate_images, selection_logits)
        
        # 소프트 선택 결과 (adaptive 온도 사용)
        adaptive_temp = self.temperature * (0.1 + 0.9 * soft_weight.mean())  # 0.1~1.0 범위
        soft_pixels, soft_weights = self.soft_selection(candidate_images, selection_logits, adaptive_temp)
        
        # 두 전략의 가중 결합
        final_pixels = hard_weight * hard_pixels + soft_weight * soft_pixels
        final_weights = hard_weight.expand(-1, K, -1, -1) * hard_weights + \
                       soft_weight.expand(-1, K, -1, -1) * soft_weights
        
        return final_pixels, final_weights
        
    def forward(self, candidate_images, partial_image, mask, fused_features):
        """
        Args:
            candidate_images: [B, K, 3, H, W]
            partial_image: [B, 3, H, W] 
            mask: [B, 1, H, W]
            fused_features: [B, feature_dim, H, W]
            
        Returns:
            fused_pixels: [B, 3, H, W]
            pixel_confidence: [B, 1, H, W]
            candidate_weights: [B, K, H, W]
        """
        B, K, C, H, W = candidate_images.shape
        
        # 1. 입력 준비
        candidates_flat = candidate_images.view(B, K*C, H, W)
        combined_input = torch.cat([
            partial_image, mask, candidates_flat, fused_features
        ], dim=1)
        
        # 2. 후보별 품질 점수 계산
        selection_logits = self.candidate_scorer(combined_input)  # [B, K, H, W]
        
        # 3. 전체 신뢰도 계산
        pixel_confidence = self.confidence_predictor(combined_input)  # [B, 1, H, W]
        
        # 4. 특징 기반 보조 예측
        feature_based_pixels = self.feature_to_pixel_net(fused_features)
        
        # 5. 선택 전략에 따른 융합
        if self.fusion_strategy == "hard":
            candidate_fused_pixels, candidate_weights = self.hard_selection(
                candidate_images, selection_logits
            )
        elif self.fusion_strategy == "soft":
            candidate_fused_pixels, candidate_weights = self.soft_selection(
                candidate_images, selection_logits, self.temperature
            )
        elif self.fusion_strategy == "adaptive":
            strategy_weights = self.strategy_selector(combined_input)  # [B, 2, H, W]
            candidate_fused_pixels, candidate_weights = self.adaptive_selection(
                candidate_images, selection_logits, strategy_weights
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # 6. 후보 기반 결과와 특징 기반 결과를 신뢰도로 조합
        confidence_3ch = pixel_confidence.repeat(1, 3, 1, 1)
        final_fused_pixels = (confidence_3ch * candidate_fused_pixels + 
                             (1 - confidence_3ch) * feature_based_pixels)
        
        return final_fused_pixels, pixel_confidence, candidate_weights