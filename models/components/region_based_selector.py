# models/components/region_based_selector.py
# 🔧 디버깅 제어 변수 (한 줄로 모든 디버깅 끄기)
ENABLE_DEBUG = False  # False로 설정하면 모든 디버깅 비활성화
DEBUG_SCALE_EVALUATION = False  # 스케일 평가 디버깅
DEBUG_CONTEXT_MODULE = False    # 컨텍스트 모듈 디버깅
DEBUG_VGG_ISSUES = False       # VGG 관련 문제 디버깅
DEBUG_MEMORY = True            # 메모리 사용량만 모니터링

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
try:
    from losses.vgg_perceptual_loss import VGGFeatureExtractor
except ImportError:
    VGGFeatureExtractor = None

class RegionBasedSelector(nn.Module):
    """
    🎯 역할: 성능 유지하면서 메모리 최적화 + 디버깅 제어
    🔧 수정: 
    - 모든 디버깅 출력 제어 가능
    - 배치 크기 동적 조정으로 메모리 절약
    - CuDNN 안정성 향상
    """
    def __init__(self, num_candidates, feature_dim=256, hidden_dim=128, 
                region_scales=[32, 16, 8], selection_strategy="hierarchical",
                final_region_size=1):
        super().__init__()
        self.num_candidates = num_candidates
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.region_scales = region_scales
        self.selection_strategy = selection_strategy
        self.final_region_size = final_region_size
        self.apply_emergency_stabilization()
        
        if ENABLE_DEBUG:
            print(f"🎯 RegionBasedSelector (High Performance + Memory Opt) - scales={region_scales}")
            print(f"💎 Pixel-level precision: {'✅ Enabled' if final_region_size == 1 else '❌ Disabled'}")
        
        # 🔧 기존 고성능 스케일별 평가기 유지 (메모리만 최적화)
        self.scale_evaluators = nn.ModuleDict()
        for scale in region_scales:
            scale_name = f"scale_{scale}"
            self.scale_evaluators[scale_name] = ScaleEvaluator(
                num_candidates=num_candidates,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                region_size=scale
            )
        
        # 🔧 기존 계층적 제약 조정기 유지
        self.hierarchical_controller = HierarchicalController(
            num_candidates=num_candidates,
            num_scales=len(region_scales)
        )
        
        # 🔧 기존 최종 통합기 유지 (메모리만 최적화)
        self.final_integrator = FinalIntegrator(
            num_candidates=num_candidates,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            final_region_size=final_region_size
        )
        
    def forward(self, candidate_images, partial_image, mask, fused_features, candidate_scores=None):
        """
        🎯 역할: 성능 유지 + 메모리 최적화 + CuDNN 안정성 향상
        🔧 수정: 더 적극적인 배치 분할로 메모리 절약
        """
        B, K, C, H, W = candidate_images.shape
        device = candidate_images.device
        
        # 🔧 메모리 사용량 추정 및 배치 분할 (더 보수적으로)
        memory_gb = torch.cuda.memory_allocated() / 1e9

        # 🔧 Score 가중치 처리 (새로 추가)
        if candidate_scores is not None:
            # candidate_scores: [B, K] 형태
            # 정규화하여 확률 분포로 변환
            score_weights = torch.softmax(candidate_scores * 2.0, dim=1)  # temperature=2.0
            # 공간적 차원으로 확장: [B, K, H, W]
            score_weights_spatial = score_weights.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        else:
            # Score 정보가 없으면 균등 가중치
            score_weights_spatial = torch.ones(B, K, H, W, device=device) / K



        if DEBUG_MEMORY:
            # print(f"🔍 [MEMORY] Input memory: {memory_gb:.2f}GB, batch size: {B}, tensor size: {candidate_images.shape}")
            pass
        
        # 🔧 메모리 분할 조건 완화 (GPU 사용률 향상)
        # if memory_gb > 35.0 or B >= 24:  # 35GB 초과 또는 배치 24 이상 시
        if False:
            if DEBUG_MEMORY:
                print(f"🔧 [MEMORY] Large memory/batch detected, using split processing")
            return self._split_batch_forward(candidate_images, partial_image, mask, fused_features, score_weights_spatial)
        
        # 🔧 일반적인 경우: 기존 로직 그대로 (성능 유지)
        return self._normal_forward(candidate_images, partial_image, mask, fused_features, score_weights_spatial)
    
    def _normal_forward(self, candidate_images, partial_image, mask, fused_features, score_weights_spatial=None):
        """기존 고성능 로직 그대로 유지 (디버깅만 제어)"""
        B, K, C, H, W = candidate_images.shape
        device = candidate_images.device
        
        # 입력 안정화
        candidate_images = torch.clamp(candidate_images, -3.0, 3.0)
        partial_image = torch.clamp(partial_image, -3.0, 3.0)
        
        # 1. 모든 스케일에서 고품질 영역별 평가
        scale_results = {}
        
        for scale in self.region_scales:
            scale_name = f"scale_{scale}"
            
            # 고품질 리사이징
            scale_h, scale_w = max(8, H // scale), max(8, W // scale)
            
            candidates_scaled = self._resize_candidates_safely(candidate_images, scale_h, scale_w)
            partial_scaled = F.interpolate(partial_image, size=(scale_h, scale_w), 
                                         mode='bicubic', align_corners=False)
            mask_scaled = F.interpolate(mask, size=(scale_h, scale_w), mode='nearest')
            
            # 특징 맵 처리
            features_scaled = self._prepare_features_safely(fused_features, scale_h, scale_w)
            
            # 고성능 스케일별 평가
            scale_result = self.scale_evaluators[scale_name](
                candidates_scaled, partial_scaled, mask_scaled, features_scaled
            )

            # 🔧 Score 가중치 적용 (새로 추가)
            if score_weights_spatial is not None:
                model_scores = scale_result['quality_scores']
                # 현재 스케일 크기로 Score 가중치 리사이즈
                scale_h, scale_w = model_scores.shape[2:]
                score_weights_scaled = F.interpolate(
                    score_weights_spatial, size=(scale_h, scale_w), 
                    mode='bilinear', align_corners=False
                )
                
                # 모델 점수와 JSON Score 가중평균 (70% 모델 + 30% Score)
                combined_scores = model_scores * 0.7 + score_weights_scaled * 0.3
                scale_result['quality_scores'] = combined_scores

            # 고품질 업샘플링으로 원본 해상도 복원
            scale_result = self._upsample_scale_result(scale_result, H, W)
            scale_results[scale] = scale_result
        
        # 2. 고급 계층적 제약 적용
        constrained_results = self.hierarchical_controller(scale_results, mask)
        
        # 3. 최고 품질 최종 통합
        final_image, confidence_map, selection_info = self.final_integrator(
            candidate_images, partial_image, constrained_results, mask, fused_features
        )
        
        return final_image, confidence_map, selection_info
    
    def apply_emergency_stabilization(self):
        """모든 모듈에 gradient 안정화 적용"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Weight 초기화 재조정
                if hasattr(module, 'weight'):
                    with torch.no_grad():
                        module.weight.data = torch.clamp(module.weight.data, -0.1, 0.1)
                if hasattr(module, 'bias') and module.bias is not None:
                    with torch.no_grad():
                        module.bias.data = torch.clamp(module.bias.data, -0.01, 0.01)



    def _split_batch_forward(self, candidate_images, partial_image, mask, fused_features, score_weights_spatial=None):
        """
        🎯 역할: 더 작은 배치로 분할하여 메모리 절약 및 CuDNN 안정성 향상
        🔧 수정: 더 보수적인 배치 크기로 안정성 확보
        """
        B, K, C, H, W = candidate_images.shape
        device = candidate_images.device
        
        # 🔧 GPU 사용률 향상을 위한 더 큰 분할 크기
        memory_gb = torch.cuda.memory_allocated() / 1e9
        if memory_gb > 38.0:
            split_size = 8   # 35.0→38.0, 2→8로 증가
        elif memory_gb > 35.0:
            split_size = 12  # 30.0→35.0, 4→12로 증가
        elif memory_gb > 30.0:
            split_size = 16  # 25.0→30.0, 6→16으로 증가
        else:
            split_size = 20  # 8→20으로 증가
                
        if DEBUG_MEMORY:
            print(f"🔧 [MEMORY] Splitting batch {B} into chunks of {split_size}")
        
        # 결과 저장용
        final_images = []
        confidence_maps = []
        selection_infos = []
        
        # 배치 분할 처리
        for start_idx in range(0, B, split_size):
            end_idx = min(start_idx + split_size, B)
            
            # 현재 청크 추출
            chunk_candidates = candidate_images[start_idx:end_idx]
            chunk_partial = partial_image[start_idx:end_idx]
            chunk_mask = mask[start_idx:end_idx]
            chunk_features = fused_features[start_idx:end_idx]

            # 🔧 Score 가중치도 청크로 분할
            chunk_score_weights = score_weights_spatial[start_idx:end_idx] if score_weights_spatial is not None else None

            if DEBUG_MEMORY:
                print(f"🔧 [MEMORY] Processing chunk {start_idx}:{end_idx}")

            # 🔧 청크별로 기존 고성능 로직 적용 + Score 가중치 전달
            chunk_final, chunk_conf, chunk_info = self._normal_forward(
                chunk_candidates, chunk_partial, chunk_mask, chunk_features, chunk_score_weights
            )
                        
            final_images.append(chunk_final)
            confidence_maps.append(chunk_conf)
            selection_infos.append(chunk_info)
            
            # 🔧 적극적인 메모리 정리
            del chunk_candidates, chunk_partial, chunk_mask, chunk_features
            del chunk_final, chunk_conf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # CuDNN 안정성
        
        # 결과 결합
        final_image = torch.cat(final_images, dim=0)
        confidence_map = torch.cat(confidence_maps, dim=0)
        
        # selection_info 결합
        combined_selection_info = selection_infos[0]
        if 'selection_weights' in combined_selection_info:
            combined_weights = torch.cat([info['selection_weights'] for info in selection_infos], dim=0)
            combined_selection_info['selection_weights'] = combined_weights
        if 'final_selection_scores' in combined_selection_info:
            combined_scores = torch.cat([info['final_selection_scores'] for info in selection_infos], dim=0)
            combined_selection_info['final_selection_scores'] = combined_scores
        
        # 디버그 정보 업데이트
        combined_selection_info['debug_info']['batch_split'] = True
        combined_selection_info['debug_info']['split_size'] = split_size
        
        if DEBUG_MEMORY:
            print(f"✅ [MEMORY] Split processing completed, final memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        return final_image, confidence_map, combined_selection_info

    def _resize_candidates_safely(self, candidate_images, target_h, target_w):
        """고품질 후보 이미지 리사이징"""
        B, K, C, H, W = candidate_images.shape
        
        if H == target_h and W == target_w:
            return candidate_images
        
        # 각 후보를 개별적으로 고품질 리사이징
        resized_candidates = []
        for k in range(K):
            resized_cand = F.interpolate(
                candidate_images[:, k], 
                size=(target_h, target_w), 
                mode='bicubic',
                align_corners=False
            )
            resized_candidates.append(resized_cand.unsqueeze(1))
        
        return torch.cat(resized_candidates, dim=1)
    
    def _prepare_features_safely(self, features, target_h, target_w):
        """특징 맵 안전 처리"""
        try:
            B, C, H, W = features.shape
            
            # VGG 호환을 위해 채널 수 제한
            if C > 512:
                if not hasattr(self, 'channel_reducer'):
                    self.channel_reducer = nn.Conv2d(C, 512, 1, bias=False).to(features.device)
                    nn.init.kaiming_normal_(self.channel_reducer.weight)
                features = self.channel_reducer(features)
            
            # 고품질 크기 조정
            if H != target_h or W != target_w:
                features = F.interpolate(
                    features, 
                    size=(target_h, target_w), 
                    mode='bicubic',
                    align_corners=False
                )
            
            return features
            
        except Exception as e:
            if ENABLE_DEBUG:
                print(f"⚠️ Feature preparation failed: {e}")
            return torch.zeros(features.shape[0], 256, target_h, target_w, device=features.device)
    
    def _upsample_scale_result(self, scale_result, target_h, target_w):
        """최고 품질 스케일 결과 업샘플링"""
        upsampled = {}
        for key, value in scale_result.items():
            if isinstance(value, torch.Tensor) and value.dim() == 4:
                if value.shape[-2:] != (target_h, target_w):
                    if value.size(1) > 1:  # 다채널
                        upsampled[key] = F.interpolate(
                            value, size=(target_h, target_w), 
                            mode='bicubic', align_corners=False
                        )
                    else:  # 단일 채널
                        upsampled[key] = F.interpolate(
                            value, size=(target_h, target_w), 
                            mode='bilinear', align_corners=False
                        )
                else:
                    upsampled[key] = value
            else:
                upsampled[key] = value
        return upsampled


class ScaleEvaluator(nn.Module):
    """
    🎯 역할: 각 스케일에서 후보들의 품질을 평가
    🔧 수정: 디버깅 출력 제어 가능
    """
    
    def __init__(self, num_candidates, feature_dim, hidden_dim, region_size):
        super().__init__()
        self.num_candidates = num_candidates
        self.region_size = region_size
        
        # 입력 채널 계산 (VGG 호환)
        max_feature_channels = min(feature_dim, 512)
        input_channels = 3 + 1 + 3 * num_candidates + max_feature_channels
        
        # 간소화된 품질 분석기
        self.quality_analyzer = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_candidates, 1),
            nn.Softmax(dim=1)
        )
        
        # 향상된 컨텍스트 평가기
        self.context_evaluator = ImprovedContextModule(num_candidates, hidden_dim)

    def forward(self, candidates_scaled, partial_scaled, mask_scaled, features_scaled):
        """
        🎯 역할: 스케일별 평가 (디버깅 출력 제어)
        🔧 수정: 모든 디버깅 출력을 제어 변수로 관리
        """
        B, K, C, H, W = candidates_scaled.shape
        device = candidates_scaled.device
        
        if DEBUG_SCALE_EVALUATION:
            print(f"🔍 [SCALE_DEBUG] ScaleEvaluator input shapes:")
            print(f"  candidates_scaled: {candidates_scaled.shape}")
            print(f"  partial_scaled: {partial_scaled.shape}")
            print(f"  mask_scaled: {mask_scaled.shape}")
            print(f"  features_scaled: {features_scaled.shape}")
        
        try:
            # 입력 준비
            candidates_flat = candidates_scaled.view(B, K*C, H, W)
            if DEBUG_SCALE_EVALUATION:
                print(f"🔍 [SCALE_DEBUG] candidates_flat: {candidates_flat.shape}")
            
            # 특징 채널 수 제한 (VGG 호환)
            if features_scaled.shape[1] > 512:
                features_scaled = features_scaled[:, :512, :, :]
                if DEBUG_SCALE_EVALUATION:
                    print(f"🔍 [SCALE_DEBUG] features_scaled (limited): {features_scaled.shape}")
            
            combined_input = torch.cat([
                partial_scaled, mask_scaled, candidates_flat, features_scaled
            ], dim=1)
            if DEBUG_SCALE_EVALUATION:
                print(f"🔍 [SCALE_DEBUG] combined_input: {combined_input.shape}")

            # 기본 품질 점수
            if hasattr(self, 'quality_analyzer'):
                if DEBUG_SCALE_EVALUATION:
                    print(f"🔍 [SCALE_DEBUG] Using quality_analyzer")
                basic_scores = self.quality_analyzer(combined_input)
                if DEBUG_SCALE_EVALUATION:
                    print(f"🔍 [SCALE_DEBUG] basic_scores: {basic_scores.shape}")
            else:
                if DEBUG_SCALE_EVALUATION:
                    print(f"🔍 [SCALE_DEBUG] No analyzer found, using uniform scores")
                basic_scores = torch.ones(B, K, H, W, device=device) / K

            # 컨텍스트 점수
            try:
                if DEBUG_SCALE_EVALUATION:
                    print(f"🔍 [SCALE_DEBUG] Before context_evaluator:")
                    print(f"  candidates_scaled: {candidates_scaled.shape}")
                    print(f"  partial_scaled: {partial_scaled.shape}")
                    print(f"  mask_scaled: {mask_scaled.shape}")
                
                context_scores = self.context_evaluator(
                    candidates_scaled, partial_scaled, mask_scaled
                )
                if DEBUG_SCALE_EVALUATION:
                    print(f"🔍 [SCALE_DEBUG] context_scores: {context_scores.shape}")
                
            except Exception as context_e:
                if DEBUG_SCALE_EVALUATION:
                    print(f"❌ [SCALE_DEBUG] Context evaluation failed: {context_e}")
                    print(f"🔧 [SCALE_DEBUG] Using uniform context scores")
                context_scores = torch.ones(B, K, H, W, device=device) / K

            # 크기 일치 확인 및 조정
            if DEBUG_SCALE_EVALUATION:
                print(f"🔍 [SCALE_DEBUG] Before final combination:")
                print(f"  basic_scores: {basic_scores.shape}")
                print(f"  context_scores: {context_scores.shape}")
            
            # 크기가 다르면 조정
            if basic_scores.shape != context_scores.shape:
                if DEBUG_SCALE_EVALUATION:
                    print(f"🔧 [SCALE_DEBUG] Size mismatch detected! Adjusting...")
                target_shape = basic_scores.shape
                
                if context_scores.shape[-2:] != target_shape[-2:]:
                    if DEBUG_SCALE_EVALUATION:
                        print(f"🔧 [SCALE_DEBUG] Resizing context_scores from {context_scores.shape} to match {target_shape}")
                    context_scores = F.interpolate(
                        context_scores, 
                        size=target_shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )

            # 두 점수 결합
            try:
                final_scores = (basic_scores + context_scores) / 2.0
                if DEBUG_SCALE_EVALUATION:
                    print(f"✅ [SCALE_DEBUG] final_scores: {final_scores.shape}")
            except Exception as combine_e:
                if DEBUG_SCALE_EVALUATION:
                    print(f"❌ [SCALE_DEBUG] Score combination failed: {combine_e}")
                    print(f"🔧 [SCALE_DEBUG] Using basic_scores only")
                final_scores = basic_scores
            
            return {
                'quality_scores': final_scores,
                'basic_scores': basic_scores
            }
            
        except Exception as e:
            if DEBUG_SCALE_EVALUATION:
                print(f"❌ [SCALE_DEBUG] Scale evaluation completely failed: {e}")
                print(f"🔧 [SCALE_DEBUG] Using emergency fallback")
            return {
                'quality_scores': torch.ones(B, K, H, W, device=device) / K
            }


class HierarchicalController(nn.Module):
    """
    🎯 역할: 다중 스케일 간의 일관성을 보장하는 계층적 제약 적용
    🔧 수정: 디버깅 출력 제어
    """
    
    def __init__(self, num_candidates, num_scales):
        super().__init__()
        self.num_candidates = num_candidates
        self.num_scales = num_scales
        
        # 간소화된 제약 강도 예측기
        self.constraint_predictor = nn.Sequential(
            nn.Conv2d(num_candidates * 2, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, scale_results, mask):
        """계층적 제약 적용 (디버깅 출력 제어)"""
        if not scale_results:
            return scale_results

        scales = sorted(scale_results.keys(), reverse=True)
        constrained_results = {}

        # 첫 번째 스케일은 그대로 사용
        constrained_results[scales[0]] = scale_results[scales[0]]

        # 나머지 스케일들에 계층적 제약 적용
        for i in range(1, len(scales)):
            current_scale = scales[i]
            parent_scale = scales[i-1]
            
            try:
                current_scores = scale_results[current_scale]['quality_scores']
                parent_scores = constrained_results[parent_scale]['quality_scores']
                
                B, K, H, W = current_scores.shape
                
                # 부모 스케일을 현재 크기로 조정
                if parent_scores.shape[-2:] != (H, W):
                    parent_upsampled = F.interpolate(
                        parent_scores, size=(H, W), mode='bilinear', align_corners=False
                    )
                else:
                    parent_upsampled = parent_scores
                
                # 제약 강도 계산
                combined_input = torch.cat([current_scores, parent_upsampled], dim=1)
                constraint_strength = self.constraint_predictor(combined_input)
                
                # 부드러운 제약 적용
                alpha = constraint_strength * 0.2
                constrained_scores = current_scores * (1 - alpha) + parent_upsampled * alpha
                
                # 정규화 보장
                constrained_scores = F.softmax(constrained_scores, dim=1)
                
                # 결과 저장
                constrained_result = scale_results[current_scale].copy()
                constrained_result['quality_scores'] = constrained_scores
                constrained_result['constraint_strength'] = constraint_strength
                constrained_results[current_scale] = constrained_result
                
            except Exception as e:
                if ENABLE_DEBUG:
                    print(f"⚠️ Hierarchical constraint failed for scale {current_scale}: {e}")
                constrained_results[current_scale] = scale_results[current_scale]

        return constrained_results


class ImprovedContextModule(nn.Module):
    """
    🎯 역할: VGG 채널 불일치 및 메모리 문제 완전 해결
    🔧 수정: 디버깅 출력 제어 추가
    """
    
    def __init__(self, num_candidates, hidden_dim):
        super().__init__()
        self.num_candidates = num_candidates
        self.hidden_dim = hidden_dim
        
        if DEBUG_CONTEXT_MODULE:
            print(f"🔍 [CONTEXT] ImprovedContextModule init - num_candidates: {num_candidates}, hidden_dim: {hidden_dim}")
        
        # VGG 사용 시도
        self.use_vgg = False
        self.vgg_extractor = None
        self.feature_comparator = None
        
        try:
            if VGGFeatureExtractor is not None:
                self.vgg_extractor = VGGFeatureExtractor(
                    feature_layer_indices=[2, 7, 16],
                    use_input_norm=True,
                    requires_grad=False,
                    vgg_weights_path="./pretrained_model/vgg19-dcbb9e9d.pth"
                )
                
                # VGG 특징 비교기 (채널 수 정확히 수정)
                self.feature_comparator = nn.ModuleDict({
                    'level_0': nn.Conv2d(64 * 2, 32, 3, padding=1),
                    'level_1': nn.Conv2d(128 * 2, 32, 3, padding=1),
                    'level_2': nn.Conv2d(256 * 2, 32, 3, padding=1),
                })
                fusion_input_dim = 96  # 32 * 3 levels
                self.use_vgg = True
                if DEBUG_CONTEXT_MODULE:
                    print("✅ [CONTEXT] VGG context evaluation enabled with correct channels")
                
        except Exception as e:
            if DEBUG_CONTEXT_MODULE:
                print(f"⚠️ [CONTEXT] VGG initialization failed: {e}")
            self.use_vgg = False
        
        # 메모리 효율적인 CNN fallback
        self.simple_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        if not self.use_vgg:
            fusion_input_dim = 512  # CNN fallback 출력 채널
            if DEBUG_CONTEXT_MODULE:
                print("📱 [CONTEXT] Using optimized CNN fallback")
        
        # 고정 크기 harmony_predictor
        self.harmony_predictor = nn.Sequential(
            nn.Conv2d(fusion_input_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, num_candidates, 1),
            nn.Softmax(dim=1)
        )
        
        if DEBUG_CONTEXT_MODULE:
            print(f"✅ [CONTEXT] Context module initialized with {fusion_input_dim} input channels")
        
    def forward(self, candidate_images, partial_image, mask):
        """메모리 효율적이고 안정적인 컨텍스트 평가 (디버깅 제어)"""
        B, K, C, H, W = candidate_images.shape
        
        # 메모리 체크
        try:
            memory_gb = torch.cuda.memory_allocated() / 1e9
            if memory_gb > 35.0:
                if DEBUG_CONTEXT_MODULE:
                    print(f"⚠️ [CONTEXT] High memory usage ({memory_gb:.2f}GB), using simple evaluation")
                return self._simple_fallback_evaluation(candidate_images, partial_image, H, W)
        except:
            pass
        
        # 최소 크기 보장
        min_size = 32
        if H < min_size or W < min_size:
            if DEBUG_CONTEXT_MODULE:
                print(f"🔧 [CONTEXT] Upsampling from {H}x{W} to {min_size}x{min_size}")
            candidate_images = F.interpolate(
                candidate_images.view(-1, C, H, W), 
                size=(min_size, min_size), 
                mode='bilinear', align_corners=False
            ).view(B, K, C, min_size, min_size)
            
            partial_image = F.interpolate(
                partial_image, size=(min_size, min_size), 
                mode='bilinear', align_corners=False
            )
            H, W = min_size, min_size
        
        # VGG 시도
        if self.use_vgg:
            try:
                return self._fixed_vgg_evaluation(candidate_images, partial_image, H, W)
            except Exception as e:
                if DEBUG_VGG_ISSUES:
                    print(f"⚠️ [CONTEXT] VGG failed: {e}, using CNN")
                self.use_vgg = False  # 영구적으로 비활성화
        
        # CNN fallback
        return self._fixed_cnn_evaluation(candidate_images, partial_image, H, W)
    
    def _fixed_vgg_evaluation(self, candidate_images, partial_image, H, W):
        """수정된 VGG 평가 (디버깅 제어)"""
        B, K = candidate_images.shape[:2]
        
        try:
            # 부분 이미지 특징 추출
            with torch.no_grad():
                partial_features = self.vgg_extractor(partial_image)
            
            all_harmony_scores = []
            for k in range(K):
                candidate = candidate_images[:, k]
                
                # 후보 이미지 특징 추출
                with torch.no_grad():
                    candidate_features = self.vgg_extractor(candidate)
                
                level_similarities = []
                for level_idx, (partial_feat, candidate_feat) in enumerate(zip(partial_features, candidate_features)):
                    level_name = f'level_{level_idx}'
                    
                    if DEBUG_VGG_ISSUES:
                        print(f"🔧 [CONTEXT] Level {level_idx} - partial: {partial_feat.shape}, candidate: {candidate_feat.shape}")
                    
                    # 크기 맞춤
                    min_h = min(partial_feat.shape[2], candidate_feat.shape[2])
                    min_w = min(partial_feat.shape[3], candidate_feat.shape[3])
                    partial_feat = partial_feat[:, :, :min_h, :min_w]
                    candidate_feat = candidate_feat[:, :, :min_h, :min_w]
                    
                    # 특징 결합
                    combined_features = torch.cat([partial_feat, candidate_feat], dim=1)
                    if DEBUG_VGG_ISSUES:
                        print(f"🔧 [CONTEXT] Combined shape: {combined_features.shape}, expected by {level_name}: {self.feature_comparator[level_name].in_channels}")
                    
                    # 채널 수 검증
                    expected_channels = self.feature_comparator[level_name].in_channels
                    if combined_features.shape[1] != expected_channels:
                        if DEBUG_VGG_ISSUES:
                            print(f"⚠️ [CONTEXT] Channel mismatch at {level_name}: got {combined_features.shape[1]}, expected {expected_channels}")
                        # 채널 수 조정
                        if combined_features.shape[1] < expected_channels:
                            padding = expected_channels - combined_features.shape[1]
                            combined_features = F.pad(combined_features, (0, 0, 0, 0, 0, padding))
                        else:
                            combined_features = combined_features[:, :expected_channels, :, :]
                        if DEBUG_VGG_ISSUES:
                            print(f"🔧 [CONTEXT] Adjusted to: {combined_features.shape}")
                    
                    processed_similarity = self.feature_comparator[level_name](combined_features)
                    
                    # 원본 크기로 복원
                    processed_similarity = F.interpolate(
                        processed_similarity, size=(H, W), mode='bilinear', align_corners=False
                    )
                    level_similarities.append(processed_similarity)
                
                # 레벨별 특징 결합
                combined_features = torch.cat(level_similarities, dim=1)
                harmony_score = self.harmony_predictor(combined_features)[:, k:k+1]
                all_harmony_scores.append(harmony_score)
            
            result = torch.cat(all_harmony_scores, dim=1)
            if DEBUG_VGG_ISSUES:
                print(f"✅ [CONTEXT] VGG evaluation successful: {result.shape}")
            return result
            
        except Exception as e:
            if DEBUG_VGG_ISSUES:
                print(f"❌ [CONTEXT] VGG evaluation failed: {e}")
            raise e
    
    def _fixed_cnn_evaluation(self, candidate_images, partial_image, H, W):
        """메모리 효율적인 CNN 평가 (디버깅 제어)"""
        B, K = candidate_images.shape[:2]
        
        try:
            # 부분 이미지 특징 추출
            partial_feat = self.simple_feature_extractor(partial_image)
            
            all_harmony_scores = []
            for k in range(K):
                candidate = candidate_images[:, k]
                candidate_feat = self.simple_feature_extractor(candidate)
                
                # 특징 결합
                combined_features = torch.cat([partial_feat, candidate_feat], dim=1)
                
                # 원본 크기로 조정
                combined_features = F.interpolate(
                    combined_features, size=(H, W), mode='bilinear', align_corners=False
                )
                
                harmony_score = self.harmony_predictor(combined_features)[:, k:k+1]
                all_harmony_scores.append(harmony_score)
                
                # 메모리 정리
                del candidate_feat, combined_features, harmony_score
            
            result = torch.cat(all_harmony_scores, dim=1)
            
            # 메모리 정리
            del partial_feat, all_harmony_scores
            
            if DEBUG_CONTEXT_MODULE:
                print(f"✅ [CONTEXT] CNN evaluation successful: {result.shape}")
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                if DEBUG_CONTEXT_MODULE:
                    print(f"⚠️ [CONTEXT] CNN evaluation OOM, using simple fallback")
                torch.cuda.empty_cache()
                return self._simple_fallback_evaluation(candidate_images, partial_image, H, W)
            else:
                raise e
    
    def _simple_fallback_evaluation(self, candidate_images, partial_image, H, W):
        """최종 간단한 fallback (디버깅 제어)"""
        B, K = candidate_images.shape[:2]
        device = candidate_images.device
        
        if DEBUG_CONTEXT_MODULE:
            print(f"🔧 [CONTEXT] Using simple fallback evaluation")
        
        try:
            # 간단한 픽셀 레벨 비교
            partial_expanded = partial_image.unsqueeze(1).expand(-1, K, -1, -1, -1)
            
            # L2 거리 기반 유사도 (메모리 효율적)
            similarity_scores = []
            for k in range(K):
                candidate = candidate_images[:, k]
                diff = torch.abs(partial_expanded[:, k] - candidate).mean(dim=1, keepdim=True)
                similarity = torch.exp(-diff)  # 유사도로 변환
                similarity_scores.append(similarity)
            
            result = torch.cat(similarity_scores, dim=1)
            result = F.softmax(result, dim=1)  # 정규화
            
            if DEBUG_CONTEXT_MODULE:
                print(f"✅ [CONTEXT] Simple fallback successful: {result.shape}")
            return result
            
        except Exception as e:
            if DEBUG_CONTEXT_MODULE:
                print(f"❌ [CONTEXT] All evaluations failed: {e}")
            # 최종 균등 분포
            return torch.ones(B, self.num_candidates, H, W, device=device) / self.num_candidates


class FinalIntegrator(nn.Module):
    """
    🎯 역할: 다중 스케일 결과를 통합하여 최종 이미지와 신뢰도 생성
    🔧 수정: 순수 픽셀 복사로 노이즈 완전 제거 + 디버깅 제어
    """

    def __init__(self, num_candidates, feature_dim, hidden_dim, final_region_size=1):
        super().__init__()
        self.num_candidates = num_candidates
        self.final_region_size = final_region_size

        # 스케일 융합기
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(num_candidates * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_candidates, 3, padding=1),
            nn.Softmax(dim=1)
        )

        # 신뢰도 예측기
        confidence_input_channels = 3 + 1 + num_candidates + min(feature_dim, 256)
        self.confidence_predictor = nn.Sequential(
            nn.Conv2d(confidence_input_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # 순수 픽셀 복사 선택기
        self.region_selector = RegionSelector(num_candidates, region_size=self.final_region_size)

    def forward(self, candidate_images, partial_image, constrained_results, mask, fused_features):
        """최종 이미지 생성 (순수 픽셀 복사) - 디버깅 제어"""
        B, K, C, H, W = candidate_images.shape
        device = candidate_images.device

        # 1. 다중 스케일 점수 융합
        if len(constrained_results) >= 2:
            scales = sorted(constrained_results.keys())
            finest_scores = constrained_results[scales[-1]]['quality_scores']
            coarsest_scores = constrained_results[scales[0]]['quality_scores']
            
            # coarsest를 finest 크기로 맞춤
            if coarsest_scores.shape[-2:] != finest_scores.shape[-2:]:
                coarsest_scores = F.interpolate(
                    coarsest_scores, size=finest_scores.shape[-2:], 
                    mode='bilinear', align_corners=False
                )
            
            # 융합
            combined_scores = torch.cat([finest_scores, coarsest_scores], dim=1)
            final_selection_scores = self.scale_fusion(combined_scores)
        else:
            # 단일 스케일인 경우
            scale_key = list(constrained_results.keys())[0] if constrained_results else None
            if scale_key is not None:
                final_selection_scores = constrained_results[scale_key]['quality_scores']
            else:
                final_selection_scores = torch.ones(B, K, H, W, device=device) / K

        # 2. 순수 픽셀 복사로 최종 이미지 생성
        selected_image, selection_weights = self.region_selector(
            candidate_images, final_selection_scores, mask, partial_image
        )

        # 3. 지능적 fallback 처리
        selection_sum = selection_weights.sum().item()
        if selection_sum < 0.01:  # 선택이 거의 없는 경우
            if ENABLE_DEBUG:
                print(f"🎯 Intelligent fallback: selection_sum={selection_sum:.6f}")
            
            # 가장 안정적인 후보 선택
            global_scores = final_selection_scores.mean(dim=(2, 3))  # [B, K]
            best_candidate_idx = torch.argmax(global_scores, dim=1)
            
            # 직접 픽셀 복사
            for b in range(B):
                selected_image[b] = candidate_images[b, best_candidate_idx[b]]
            
            # 선택 가중치 업데이트
            selection_weights = torch.zeros_like(final_selection_scores)
            for b in range(B):
                selection_weights[b, best_candidate_idx[b]] = 1.0

        # 4. 신뢰도 예측
        try:
            # 특징 크기 조정
            if fused_features.shape[1] > 256:
                if not hasattr(self, 'feature_reducer'):
                    self.feature_reducer = nn.Conv2d(fused_features.shape[1], 256, 1).to(fused_features.device)
                reduced_features = self.feature_reducer(fused_features)
            else:
                reduced_features = fused_features
                
            reduced_features = F.interpolate(
                reduced_features, size=(H, W), mode='bilinear', align_corners=False
            )
            
            confidence_input = torch.cat([
                selected_image,
                mask,
                final_selection_scores,
                reduced_features
            ], dim=1)
            confidence_map = self.confidence_predictor(confidence_input)
            
        except Exception as e:
            if ENABLE_DEBUG:
                print(f"⚠️ Confidence prediction failed: {e}")
            # 기본 신뢰도
            confidence_map = torch.ones(B, 1, H, W, device=device) * 0.8

        # 5. 최종 이미지 구성 (visible 영역 보존 보장)
        visible_mask = mask.expand_as(selected_image)
        fill_mask = 1.0 - visible_mask
        
        # visible 영역은 반드시 원본 유지
        final_image = partial_image * visible_mask + selected_image * fill_mask

        # 성능 정보 수집
        selection_info = {
            'selection_weights': selection_weights,
            'final_selection_scores': final_selection_scores,
            'debug_info': {
                'mask_ratio': mask.mean().item(),
                'selection_active': (selection_weights.sum() > 0.01).item(),
                'selection_sum': selection_weights.sum().item(),
                'region_size': self.final_region_size,
                'pure_copy_mode': True,
                'fallback_used': selection_sum < 0.01
            }
        }

        return final_image, confidence_map, selection_info


class RegionSelector(nn.Module):
    """
    🎯 역할: 영역별/픽셀별 최적 후보 선택 및 순수 복사
    🔧 수정: 보간 없는 직접 픽셀 복사로 노이즈 완전 제거 + 디버깅 제어
    """

    def __init__(self, num_candidates=None, region_size=1):
        super().__init__()
        self.num_candidates = num_candidates
        self.region_size = region_size
        if ENABLE_DEBUG:
            print(f"🎯 RegionSelector - region_size={region_size} ({'💎 pure pixel copy' if region_size == 1 else '🔲 region copy'})")

    def forward(self, candidate_images, selection_scores, mask, partial_image):
        """선택 점수에 따라 후보에서 픽셀/영역을 직접 복사 (디버깅 제어)"""
        B, K, C, H, W = candidate_images.shape
        device = candidate_images.device

        if self.region_size == 1:
            # 픽셀 단위 순수 복사
            return self._pure_pixel_copy(candidate_images, selection_scores, mask, partial_image)
        else:
            # 영역 단위 순수 복사
            return self._pure_region_copy(candidate_images, selection_scores, mask, partial_image)

    def _pure_pixel_copy(self, candidate_images, selection_scores, mask, partial_image):
        """각 픽셀마다 최고 점수 후보에서 직접 복사 (디버깅 제어)"""
        B, K, C, H, W = candidate_images.shape
        device = candidate_images.device
        
        # 마스크 정리: visible 영역 보존 보장
        visible_mask = mask.expand_as(partial_image)
        fill_mask = 1.0 - visible_mask
        
        # 각 픽셀의 최고 후보 선택
        best_indices = torch.argmax(selection_scores, dim=1, keepdim=True)  # [B, 1, H, W]
        best_indices_expanded = best_indices.unsqueeze(2).expand(-1, -1, C, -1, -1)  # [B, 1, C, H, W]
        
        # 직접 픽셀 복사 (gather로 정확한 인덱싱)
        selected_pixels = torch.gather(candidate_images, 1, best_indices_expanded).squeeze(1)  # [B, C, H, W]
        
        # 완전한 순수 복사: visible 영역은 원본, fill 영역은 선택된 픽셀
        final_image = partial_image * visible_mask + selected_pixels * fill_mask
        
        # Hard selection 가중치
        selection_weights = torch.zeros_like(selection_scores)
        selection_weights.scatter_(1, best_indices, 1.0)
        
        return final_image, selection_weights

    def _pure_region_copy(self, candidate_images, selection_scores, mask, partial_image):
        """영역 단위로 최고 점수 후보에서 직접 복사 (디버깅 제어)"""
        B, K, C, H, W = candidate_images.shape
        device = candidate_images.device

        region_h = min(self.region_size, H)
        region_w = min(self.region_size, W)
        num_regions_h = math.ceil(H / region_h)
        num_regions_w = math.ceil(W / region_w)

        final_image = partial_image.clone()  # 원본으로 시작
        selection_weights = torch.zeros_like(selection_scores)

        for i in range(num_regions_h):
            for j in range(num_regions_w):
                start_h = i * region_h
                end_h = min((i + 1) * region_h, H)
                start_w = j * region_w
                end_w = min((j + 1) * region_w, W)

                # 현재 영역의 마스크 비율 확인
                region_mask = mask[:, :, start_h:end_h, start_w:end_w]
                mask_ratio = region_mask.mean().item()

                # visible 영역이 90% 이상이면 건드리지 않음
                if mask_ratio > 0.9:
                    selection_weights[:, :, start_h:end_h, start_w:end_w] = 0.0
                    continue

                # 영역별 평균 점수로 최고 후보 선택
                region_scores = selection_scores[:, :, start_h:end_h, start_w:end_w]
                region_avg_scores = region_scores.mean(dim=(2, 3))  # [B, K]
                
                # 유효한 점수가 있는지 확인
                max_scores = region_avg_scores.max(dim=1)[0]
                valid_mask = max_scores > 1e-6

                for b_idx in range(B):
                    if valid_mask[b_idx]:
                        best_k = torch.argmax(region_avg_scores[b_idx])
                        
                        # 영역별 순수 복사
                        fill_area_mask = (1.0 - region_mask[b_idx]).expand(C, -1, -1)
                        
                        # fill 영역만 후보에서 복사
                        candidate_region = candidate_images[b_idx, best_k, :, start_h:end_h, start_w:end_w]
                        final_image[b_idx, :, start_h:end_h, start_w:end_w] = \
                            final_image[b_idx, :, start_h:end_h, start_w:end_w] * (1 - fill_area_mask) + \
                            candidate_region * fill_area_mask

                        # 선택 가중치 설정
                        for k in range(K):
                            if k == best_k:
                                selection_weights[b_idx, k, start_h:end_h, start_w:end_w] = 1.0
                            else:
                                selection_weights[b_idx, k, start_h:end_h, start_w:end_w] = 0.0

        return final_image, selection_weights


# 🔧 추가: 메모리 및 CuDNN 최적화를 위한 유틸리티 함수들

def optimize_for_memory_and_cudnn():
    """
    🎯 역할: 메모리 및 CuDNN 안정성을 위한 전역 설정
    🔧 수정: 배치 사이즈 권장사항 및 CuDNN 최적화
    """
    import torch.backends.cudnn as cudnn
    
    # CuDNN 최적화
    cudnn.benchmark = False  # 고정된 입력 크기가 아니므로 False
    cudnn.deterministic = True  # 재현성을 위해
    cudnn.enabled = True
    
    # 메모리 관리 설정
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # GPU 메모리 파편화 방지
        torch.cuda.set_per_process_memory_fraction(0.9)  # 90%만 사용
    
    if ENABLE_DEBUG:
        print("🔧 [MEMORY] CuDNN and memory optimizations applied")
        print("📋 [MEMORY] Recommended batch sizes:")
        print("  - 24GB GPU: batch_size <= 8")
        print("  - 40GB GPU: batch_size <= 16")
        print("  - High resolution (>512): batch_size <= 4")

def get_recommended_batch_size(gpu_memory_gb, image_size, num_candidates=3):
    """
    🎯 역할: GPU 메모리와 이미지 크기에 따른 권장 배치 사이즈 계산
    🔧 수정: 보수적인 배치 사이즈로 안정성 확보
    """
    # 기본 계산
    if image_size <= 256:
        base_batch = min(16, gpu_memory_gb // 2)
    elif image_size <= 512:
        base_batch = min(12, gpu_memory_gb // 3)
    else:
        base_batch = min(8, gpu_memory_gb // 4)
    
    # 후보 수에 따른 조정
    if num_candidates > 3:
        base_batch = max(2, base_batch // 2)
    
    # 안전 마진 적용
    recommended = max(2, int(base_batch * 0.8))
    
    if ENABLE_DEBUG:
        print(f"🔧 [MEMORY] Recommended batch size: {recommended} "
              f"(GPU: {gpu_memory_gb}GB, Image: {image_size}, Candidates: {num_candidates})")
    
    return recommended

# 사용 예시:
# 1. 스크립트 시작 시:
# optimize_for_memory_and_cudnn()
#
# 2. 배치 사이즈 결정 시:
# batch_size = get_recommended_batch_size(40, 512, 3)  # 40GB GPU, 512x512 이미지, 3개 후보
#
# 3. 디버깅 끄기:
# 파일 상단의 ENABLE_DEBUG = False 로 설정