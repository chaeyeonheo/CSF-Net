# train_aft.py (성능 최우선 버전) - 고성능 픽셀 레벨 AFT-Net

import os
import yaml
import random
import numpy as np
from tqdm import tqdm
import argparse
import datetime
import logging
import traceback
import wandb
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from data_utils import CSFDataset
from models import CSFNetwork
from losses import CSFLoss
from losses.csf_losses import RegionBasedCSFLoss
from utils import (
    setup_logger, 
    save_checkpoint, 
    load_checkpoint, 
    visualize_and_save_batch_aft,
    save_tensor_image
)


class ConfigNamespace:
    def __init__(self, adict):
        self.__dict__.update(adict)
        for k, v in adict.items():
            if isinstance(v, dict):
                self.__dict__[k] = ConfigNamespace(v)


def load_config_from_yaml(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ConfigNamespace(config_dict)


def set_seed(seed, rank=0):
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)


def setup_ddp(rank, world_size, config):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group(
        backend=config.distributed.dist_backend,
        init_method=config.distributed.init_method,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(hours=6)  # 이 줄 추가
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


PROJECT_ROOT_FOR_PATHS = os.path.dirname(os.path.abspath(__file__))



# =============================================================================
# 1. 새로 추가할 함수들 (train_aft.py 상단에 추가)
# =============================================================================

def check_model_output_stability(model_output, batch_idx, epoch, rank, logger, 
                                stabilize_threshold=8.0, emergency_threshold=15.0):
    """더 강력한 모델 출력 안정화 - 임계값을 더 낮춤"""
    if isinstance(model_output, (list, tuple)):
        image_output = model_output[0]
        confidence_output = model_output[1] if len(model_output) > 1 else None
        extra_outputs = model_output[2:] if len(model_output) > 2 else []
    else:
        image_output = model_output
        confidence_output = None
        extra_outputs = []
    
    # NaN/Inf 체크
    if torch.isnan(image_output).any() or torch.isinf(image_output).any():
        return False, "NaN/Inf in output", model_output
    
    max_val = torch.abs(image_output).max().item()
    
    # Emergency 케이스: 완전히 스킵 (더 낮은 임계값)
    if max_val > emergency_threshold:
        if rank == 0:
            logger.warning(f"⚠️ Emergency skip - extreme output ({max_val:.2f}) at E{epoch+1} B{batch_idx}")
        return False, f"Emergency: extreme output {max_val:.2f}", model_output
    
    # 안정화 케이스: 더 강력한 안정화
    elif max_val > stabilize_threshold:
        if rank == 0:
            logger.warning(f"🔧 Strong stabilizing output ({max_val:.2f}) at E{epoch+1} B{batch_idx}")
        
        # 더 강력한 안정화: 하드 클리핑 + 스케일링
        stabilized_image = torch.clamp(image_output, -stabilize_threshold, stabilize_threshold)
        stabilized_image = stabilized_image * 0.5  # 추가로 50% 스케일링
        
        # confidence도 강력하게 안정화
        if confidence_output is not None:
            stabilized_conf = torch.clamp(confidence_output, -3.0, 3.0)
            stabilized_conf = stabilized_conf * 0.5
            
            stabilized_output = (stabilized_image, stabilized_conf) + tuple(extra_outputs)
        else:
            stabilized_output = stabilized_image
        
        return True, f"Strong stabilized from {max_val:.2f}", stabilized_output
    
    # 정상 케이스
    return True, "Normal", model_output


def apply_gradient_scaling(model, scale_factor=0.3):
    """더 강력한 gradient 스케일링"""
    with torch.no_grad():
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # 모든 gradient에 스케일링 적용 (더 공격적)
        if total_norm > 2.0:  # 매우 낮은 임계값
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(scale_factor)
        
        return total_norm


def enhanced_loss_scaling_check(loss_dict, max_individual_loss=5.0, max_total_loss=10.0):
    """더 강력한 손실값 안정화 - 임계값을 더 낮춤"""
    total_loss = loss_dict.get('total_loss', torch.tensor(0.0))
    
    # Total loss 체크 (더 엄격)
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        return False, "NaN/Inf in total loss", loss_dict
    
    total_val = total_loss.item()
    if total_val > max_total_loss:
        return False, f"Total loss too large: {total_val:.2f}", loss_dict
    
    # 개별 손실들 강력한 스케일링
    scaled_loss_dict = {}
    scale_factor = 1.0
    
    for key, value in loss_dict.items():
        if torch.is_tensor(value) and value.numel() == 1:
            val = value.item()
            if torch.isnan(value) or torch.isinf(value):
                return False, f"NaN/Inf in {key}", loss_dict
            
            # 더 작은 임계값으로 스케일링
            if val > max_individual_loss:
                scale_factor = min(scale_factor, max_individual_loss / (val + 1e-6))
            
            scaled_loss_dict[key] = value
        else:
            scaled_loss_dict[key] = value
    
    # 강력한 스케일링 적용
    if scale_factor < 0.9:  # 더 자주 스케일링
        for key, value in scaled_loss_dict.items():
            if torch.is_tensor(value) and value.numel() == 1 and 'loss' in key.lower():
                scaled_loss_dict[key] = value * scale_factor
        
        return True, f"Strong scaled by {scale_factor:.3f}", scaled_loss_dict
    
    return True, "Normal", loss_dict


def smart_gradient_check_only(model, emergency_threshold=1e6):  # 1e6 → 1e4 (매우 엄격)
    """
    🎯 역할: Emergency Gradient Explosion 완전 차단 (Ultra Strict)
    📍 위치: train_aft.py 내 기존 smart_gradient_check_only 함수를 이것으로 완전 교체
    
    🚨 Emergency: 5.56e+14 같은 극단적 gradient 즉시 차단
    ✅ 성능 유지: 정상 범위 gradient는 모두 허용
    """
    total_norm = 0.0
    extreme_layers = 0
    dangerous_layers = []
    
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                # 🚨 개별 레이어 극값 체크 (매우 엄격)
                if param_norm > 100:  # 1e3 → 100 (극도로 엄격)
                    extreme_layers += 1
                    dangerous_layers.append((name, param_norm))
                    
                # 🚨 특별 위험 레이어 즉시 차단
                if param_norm > 1e6:  # 극도로 위험한 레이어
                    print(f"[EMERGENCY] 극위험 레이어 발견: {name} = {param_norm:.2e}")
                    return False, param_norm, f"CRITICAL: Layer {name} gradient {param_norm:.2e}"
        
        total_norm = total_norm ** 0.5
    
    # 🚨 Ultra Strict 체크 (explosion 완전 차단)
    if total_norm > emergency_threshold:  # 1e4 (극도로 엄격)
        print(f"[EMERGENCY] Total gradient explosion: {total_norm:.2e}")
        return False, total_norm, f"Emergency: Total gradient {total_norm:.2e} > {emergency_threshold:.0e}"
    elif extreme_layers > 1:  # 3 → 1 (극도로 엄격)
        # 위험 레이어 상세 출력
        dangerous_layers.sort(key=lambda x: x[1], reverse=True)
        top_dangerous = dangerous_layers[:2]
        print(f"[EMERGENCY] 위험 레이어들: {[f'{name}:{norm:.2e}' for name, norm in top_dangerous]}")
        return False, total_norm, f"Extreme layers: {extreme_layers} (top: {[f'{name}:{norm:.2e}' for name, norm in top_dangerous]})"
    elif total_norm > 1e3:  # 추가 중간 단계 체크
        print(f"[WARNING] 높은 gradient 감지: {total_norm:.2e}")
        return False, total_norm, f"High gradient: {total_norm:.2e}"
    elif torch.isnan(torch.tensor(total_norm)) or torch.isinf(torch.tensor(total_norm)):
        return False, total_norm, "NaN/Inf gradient detected"
    else:
        return True, total_norm, "Normal"


def emergency_gradient_surgery(model):
    """
    🎯 역할: 위험한 gradient를 수술적으로 제거
    📍 위치: train_aft.py에 새로 추가할 함수
    
    🚨 Emergency: 위험한 gradient만 선택적으로 0으로 설정
    """
    surgery_count = 0
    
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                
                # 🚨 위험한 gradient 수술적 제거
                if param_norm > 1e3:
                    print(f"[SURGERY] 위험 gradient 제거: {name} ({param_norm:.2e} → 0)")
                    p.grad.data.zero_()
                    surgery_count += 1
                elif param_norm > 100:
                    # 부분적 스케일링
                    scale_factor = 100.0 / param_norm
                    p.grad.data.mul_(scale_factor)
                    print(f"[SURGERY] Gradient 스케일링: {name} ({param_norm:.2e} → {param_norm * scale_factor:.2e})")
                    surgery_count += 1
    
    return surgery_count


def ultra_safe_gradient_clipping(model, max_norm=1.0):
    """
    🎯 역할: 초강력 gradient clipping
    📍 위치: train_aft.py에 새로 추가할 함수
    
    🚨 Emergency: 매우 작은 max_norm으로 강력 clipping
    """
    # 1단계: 표준 clipping
    original_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    
    # 2단계: 추가 안전 clipping
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.grad is not None:
                # 개별 파라미터별 추가 clipping
                param_norm = p.grad.data.norm(2).item()
                if param_norm > max_norm:
                    scale = max_norm / (param_norm + 1e-8)
                    p.grad.data.mul_(scale)
    
    return original_norm


def minimal_output_check(model_output, emergency_threshold=50.0):  # 30.0 → 50.0 (성능 우선)
    """
    🎯 역할: 성능 우선 모델 출력 체크
    📍 위치: train_aft.py 내 기존 minimal_output_check 함수를 이것으로 완전 교체
    
    ✅ 성능 우선: 학습 진행을 최대한 허용
    🔧 안정성: 극단적인 경우만 차단
    """
    if isinstance(model_output, (list, tuple)):
        image_output = model_output[0]
    else:
        image_output = model_output
    
    # ✅ 성능 우선: 크기 검증만 유지 (VGG 크기 오류 해결용)
    if image_output.dim() != 4:
        return False, f"Wrong output dimension: {image_output.dim()}"
    
    B, C, H, W = image_output.shape
    if H <= 0 or W <= 0:
        return False, f"Invalid spatial dimensions: {H}x{W}"
    
    if C != 3:
        return False, f"Wrong channel count: {C} (expected 3)"
    
    # NaN/Inf 체크만 유지
    if torch.isnan(image_output).any() or torch.isinf(image_output).any():
        return False, "NaN/Inf detected"
    
    # ✅ 성능 우선: 임계값을 관대하게 (학습 진행 최우선)
    max_val = torch.abs(image_output).max().item()
    if max_val > emergency_threshold:  # 50.0으로 관대하게
        return False, f"Extreme output: {max_val:.2f} > {emergency_threshold}"
    
    return True, "Normal"
# =============================================================================
# 1. 새로 추가할 함수들 (train_aft.py 상단에 추가)
# =============================================================================






# train_aft.py의 get_adaptive_loss_weights 함수 (line 217 근처)
# 전체 함수를 다음으로 교체:

# def get_adaptive_loss_weights(lw, epoch, total_epochs, fusion_method):
#     """
#     🎯 역할: 학습 진행도에 따라 손실 가중치를 동적으로 조절하는 커리큘큘럼 학습을 적용합니다.
#     (수정) VGG Loss 폭주를 막기 위해 초반 가중치를 더욱 낮추고, 서서히 증가시킵니다.
#     """
#     progress = epoch / total_epochs
    
#     weights = {
#         'l1_masked_weight': getattr(lw, 'l1_masked_weight', 1.0),
#         'perceptual_weight': getattr(lw, 'perceptual_weight', 0.1),
#         'confidence_bce_weight': getattr(lw, 'confidence_bce_weight', 0.5),
#         'confidence_error_correlation_weight': getattr(lw, 'confidence_error_correlation_weight', 0.2),
#         'boundary_loss_weight': getattr(lw, 'boundary_loss_weight', 0.1),
#     }

#     if fusion_method == "region_based":
#         weights.update({
#             'region_consistency_weight': getattr(lw, 'region_consistency_weight', 0.2),
#             'hierarchical_consistency_weight': getattr(lw, 'hierarchical_consistency_weight', 0.1),
#             'selection_consistency_weight': getattr(lw, 'selection_consistency_weight', 0.1),
#             'edge_preservation_weight': getattr(lw, 'edge_preservation_weight', 0.1),
#             'color_consistency_weight': getattr(lw, 'color_consistency_weight', 0.1),
#         })

#     # ================= 🎯 강화된 커리큘럼 적용 🎯 =================
#     # 학습 극초반 (0% ~ 10%)
#     if progress < 0.1:
#         # Perceptual loss 가중치를 1% 수준으로 극도로 낮춰 거의 영향을 주지 않도록 합니다.
#         weights['perceptual_weight'] *= 0.01
#         # 오직 L1 loss에만 집중하여 기본적인 이미지 복원 능력을 안정적으로 학습합니다.
#         weights['l1_masked_weight'] *= 1.5

#     # 학습 초반 (10% ~ 30%)
#     elif progress < 0.3:
#         # Perceptual loss 가중치를 10% 수준으로 유지하며 서서히 영향을 주기 시작합니다.
#         weights['perceptual_weight'] *= 0.1
#         weights['l1_masked_weight'] *= 1.2
#         if fusion_method == "region_based":
#             weights['edge_preservation_weight'] *= 1.2



# train_aft.py 파일의 get_adaptive_loss_weights 함수를 아래 코드로 완전히 교체하세요.

# train_aft.py 파일의 get_adaptive_loss_weights 함수를 아래 코드로 완전히 교체하세요.

def get_adaptive_loss_weights(lw, epoch, total_epochs, fusion_method):
    """
    [최종 수정] 동적 스케줄링을 완전히 비활성화하고, config 파일의 가중치를 그대로 사용하는 함수.
    
    Args:
        lw (ConfigNamespace): config.yaml의 loss_weights 섹션에서 읽어온 가중치 객체.
        epoch (int): 현재 에포크 (이 함수에서는 사용되지 않음).
        total_epochs (int): 전체 에포크 (이 함수에서는 사용되지 않음).
        fusion_method (str): 모델의 fusion 방식.
    
    Returns:
        dict: 학습에 사용할 고정된 가중치 딕셔너리.
    """
    # config 파일에서 읽어온 기본 가중치를 그대로 사용합니다.
    weights = {
        'l1_masked_weight': getattr(lw, 'l1_masked_weight', 1.0),
        'perceptual_weight': getattr(lw, 'perceptual_weight', 0.1),
        'confidence_bce_weight': getattr(lw, 'confidence_bce_weight', 0.1),
        'confidence_error_correlation_weight': getattr(lw, 'confidence_error_correlation_weight', 0.05),
        'boundary_loss_weight': getattr(lw, 'boundary_loss_weight', 0.1),
    }

    if fusion_method == "region_based":
        weights.update({
            'region_consistency_weight': getattr(lw, 'region_consistency_weight', 0.05),
            'hierarchical_consistency_weight': getattr(lw, 'hierarchical_consistency_weight', 0.05),
            'selection_consistency_weight': getattr(lw, 'selection_consistency_weight', 0.05),
            'edge_preservation_weight': getattr(lw, 'edge_preservation_weight', 0.1),
            'color_consistency_weight': getattr(lw, 'color_consistency_weight', 0.05),
        })

    # 학습 시작 시 한 번만 고정된 가중치를 사용하고 있음을 로그로 명확히 알립니다.
    if epoch == 0:
        print("\n" + "="*50)
        print("✅ [INFO] Using FIXED loss weights from config file.")
        print(f"   - perceptual_weight: {weights['perceptual_weight']}")
        print("="*50 + "\n")
    
    # epoch에 따라 가중치를 바꾸는 모든 로직을 제거하고, 고정된 딕셔너리를 반환합니다.
    return weights


def get_current_lr(epoch, step, total_steps_per_epoch, base_lr, min_lr, total_epochs):
    """
    🎯 역할: 고성능 학습률 스케줄링 (기존 함수명 유지)
    📍 위치: train_aft.py의 기존 get_current_lr 함수를 이것으로 완전 교체
    
    ✅ 성능 유지: 기존 스케줄링 방식 그대로 유지
    🔧 안정성: warmup만 약간 길게 조정
    """
    total_steps = epoch * total_steps_per_epoch + step
    warmup_epochs = 12  # 기존 15에서 12로 (성능 유지)
    restart_epochs = [50, 100]  # 기존과 동일
    
    # 고성능 warmup (기존과 거의 동일)
    warmup_steps = warmup_epochs * total_steps_per_epoch
    if total_steps < warmup_steps:
        warmup_progress = total_steps / warmup_steps
        # 기존과 동일한 quadratic warmup (성능 유지)
        return base_lr * (warmup_progress * 0.1 + 0.9 * warmup_progress ** 2)
    
    # Cosine annealing with restarts (기존과 동일)
    effective_epoch = epoch - warmup_epochs
    for restart_epoch in sorted(restart_epochs, reverse=True):
        if epoch >= restart_epoch:
            effective_epoch = epoch - restart_epoch
            break
    
    if effective_epoch <= 0:
        return base_lr
    
    # 기존과 동일한 cosine scheduling (성능 유지)
    max_epoch = total_epochs - warmup_epochs
    progress = effective_epoch / max_epoch
    cosine_lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return max(cosine_lr, min_lr)



def update_ema_model(ema_model, model, ema_decay=0.999):
    """EMA 모델 업데이트 - 더 보수적인 decay"""
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)


def create_enhanced_criterion(fusion_method, loss_weights, device, rank, logger):
    """
    🎯 역할: 고성능 손실함수 생성 (기존 함수명 유지)
    📍 위치: train_aft.py의 기존 create_enhanced_criterion 함수를 이것으로 완전 교체
    
    ✅ 성능 유지: 기존 EnhancedRegionBasedCSFLoss 그대로 사용
    🔧 안정성: 예외 처리만 강화
    """
    
    if fusion_method == "region_based":
        try:
            # 기존과 동일: 최고 성능 버전 우선 시도
            from losses.csf_losses import EnhancedRegionBasedCSFLoss
            criterion = EnhancedRegionBasedCSFLoss(**loss_weights).to(device)
            if rank == 0: 
                logger.info(f"✅ Using EnhancedRegionBasedCSFLoss for maximum performance")
        except ImportError:
            try:
                # 두 번째 선택지
                from losses.csf_losses import AdvancedRegionBasedCSFLoss
                criterion = AdvancedRegionBasedCSFLoss(**loss_weights).to(device)
                if rank == 0: 
                    logger.info(f"✅ Using AdvancedRegionBasedCSFLoss for high performance")
            except ImportError:
                # 기본 버전
                from losses.csf_losses import RegionBasedCSFLoss
                criterion = RegionBasedCSFLoss(**loss_weights).to(device)
                if rank == 0: 
                    logger.info(f"✅ Using RegionBasedCSFLoss (standard version)")
        except Exception as e:
            # 예외 처리 강화 (안정성)
            if rank == 0:
                logger.warning(f"⚠️ Failed to create region-based loss, using basic CSFLoss: {e}")
            from losses.csf_losses import CSFLoss
            criterion = CSFLoss(**loss_weights).to(device)
    else:
        from losses.csf_losses import CSFLoss
        criterion = CSFLoss(**loss_weights).to(device)
        if rank == 0: 
            logger.info(f"✅ Using CSFLoss for fusion_method: {fusion_method}")
    
    return criterion


def enable_performance_optimizations(model, rank, logger):
    """추가 성능 향상 옵션들"""
    # PyTorch 2.0 컴파일 최적화
    if hasattr(torch, 'compile') and rank == 0:
        try:
            compiled_model = torch.compile(model, mode='max-autotune', fullgraph=False)
            if rank == 0:
                logger.info("🚀 Enabled torch.compile for maximum performance")
            return compiled_model
        except Exception as e:
            if rank == 0:
                logger.warning(f"⚠️ torch.compile failed: {e}")
    
    # Channels Last 메모리 최적화
    try:
        model = model.to(memory_format=torch.channels_last)
        if rank == 0:
            logger.info("🚀 Enabled channels_last memory optimization")
    except Exception as e:
        if rank == 0:
            logger.warning(f"⚠️ channels_last optimization failed: {e}")
    
    return model

# # train_aft.py의 emergency_vgg_content_check 함수 (line 280 근처)
# # 전체 함수를 다음으로 교체:

# def emergency_vgg_content_check(loss_dict_batch, epoch, batch_idx, rank, logger):
#     """
#     🎯 역할: VGG content loss 폭발 실시간 감지 및 차단 (역치 조절 버전)
#     """
#     total_loss = loss_dict_batch.get('total_loss', torch.tensor(1.0))
#     # total_loss가 0에 가까울 경우를 대비해 작은 값(epsilon)을 더해줍니다.
#     total_val = total_loss.item() + 1e-8
    
#     p_content = loss_dict_batch.get('p_content', torch.tensor(0.0))
#     if torch.is_tensor(p_content):
#         p_content_val = p_content.item()
#         # p_content에 곱해지는 perceptual_weight를 고려하여 실제 기여도를 계산
#         perceptual_weight = loss_dict_batch.get('perceptual', torch.tensor(0.0)).item() / (p_content_val + 1e-8)
#         content_contribution = (p_content_val * perceptual_weight)
#         content_ratio = (content_contribution / total_val) * 100

#         # ================= 🎯 역치 조절 🎯 =================
#         # Content loss의 기여도가 총 손실의 80% 이상일 때만 폭주로 판단 (기준 완화)
#         # 이제 loss 값 자체가 클리핑되므로, 이 비율이 높아지는 것은 다른 loss들이 작다는 의미일 수 있습니다.
#         if content_ratio > 80.0:
#             if rank == 0:
#                 logger.warning(f"🚨 VGG content loss dominance! p_content_contrib={content_contribution:.4f} ({content_ratio:.1f}%) at E{epoch+1} B{batch_idx}")
#             return False, f"VGG content dominance: {content_ratio:.1f}%"
#         # =======================================================

#     # 총 손실 폭발 감지 기준은 유지 (더 중요한 안전장치)
#     if total_val > 5.0:
#         if rank == 0:
#             logger.warning(f"🚨 Total loss explosion! total={total_val:.4f} at E{epoch+1} B{batch_idx}")
#         return False, f"Total loss explosion: {total_val:.4f}"
    
#     return True, "Normal"

# def emergency_bias_monitor(model, rank, logger, epoch, batch_idx):
#     """
#     🎯 역할: context_patch_embed.proj.bias gradient 실시간 모니터링
#     📍 위치: train_aft.py에 새로 추가할 함수
    
#     🚨 핵심: 문제의 bias layer만 집중 모니터링하여 조기 차단
#     """
#     dangerous_bias_layers = [
#         'module.context_patch_embed.proj.bias',
#         'module.candidate_patch_embed.proj.bias'
#     ]
    
#     for name, param in model.named_parameters():
#         if param.grad is not None and name in dangerous_bias_layers:
#             bias_grad_norm = param.grad.data.norm(2).item()
            
#             # 🚨 bias gradient 특별 임계값 (매우 엄격)
#             if bias_grad_norm > 1e3:  # bias는 더 엄격하게
#                 if rank == 0:
#                     logger.warning(f"🚨 BIAS EXPLOSION: {name} = {bias_grad_norm:.2e} at E{epoch+1} B{batch_idx}")
                
#                 # 즉시 해당 bias gradient 0으로 설정
#                 with torch.no_grad():
#                     param.grad.data.zero_()
                
#                 return True, f"Bias {name} explosion: {bias_grad_norm:.2e}"
    
#     return False, "Normal"


def train_worker(rank, world_size, config, args):
    """
    🎯 수정 사항:
    1. VGG 손실함수 중복 초기화 방지 (GPU 메모리 누수 해결)
    2. 시각화 안정성 강화 (모델 출력 None 문제 해결)
    3. DDP 동기화 개선 (프로세스 간 통신 오류 방지)
    """
    is_ddp = config.distributed.use_ddp
    if is_ddp:
        setup_ddp(rank, world_size, config)
    
    set_seed(config.seed, rank if is_ddp else 0)
    current_process_device = torch.device(f"cuda:{rank}" if is_ddp else ("cuda" if torch.cuda.is_available() else "cpu"))

    # WandB 초기화
    if rank == 0 and hasattr(config, 'wandb'):
        wandb.init(
            project=getattr(config.wandb, 'project', 'AFT-Net-HighPerformance'),
            name=f"{config.exp_name}",
            config=config.__dict__,
            tags=getattr(config.wandb, 'tags', []),
            notes=getattr(config.wandb, 'notes', '')
        )

    # 로깅 설정
    log_dir_path = os.path.join(config.output_dir, "logs")
    os.makedirs(log_dir_path, exist_ok=True)
    log_filename_suffix = f"_train_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_filename = f"{config.exp_name}{log_filename_suffix}"
    if is_ddp and rank != 0:
        log_filename = f"{config.exp_name}_rank{rank}{log_filename_suffix}"
    log_file_path = os.path.join(log_dir_path, log_filename)
    logger = setup_logger(config.exp_name, log_file_path, level=logging.INFO, rank=rank, world_size=world_size)
    
    if rank == 0:
        logger.info("🚀 AFT-Net High Performance Training Started")
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Using DDP: {is_ddp}, World Size: {world_size}")
    logger.info(f"Process {rank} assigned to device: {current_process_device}")

    # 데이터셋 및 데이터로더 초기화
    train_dataset = CSFDataset(
        root_dir=config.data.train_root,
        Kmax=config.data.Kmax,
        split='train',
        img_size=config.data.img_size,
        transform_params=config.data.transform_params.__dict__,
        config_data=config.data,
        device_for_scoring='cpu',  # GPU 교착 상태 방지
        logger=logger
    )

    if not train_dataset.samples:
        if rank == 0: logger.error("❌ No training samples loaded. Exiting.")
        if is_ddp: cleanup_ddp()
        return

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed) if is_ddp else None
    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size_per_gpu,
        shuffle=(train_sampler is None), num_workers=6,  # 0 → 4로 변경
        pin_memory=True, drop_last=True, sampler=train_sampler,
        persistent_workers=True,  # 워커 재사용
        prefetch_factor=4         # 미리 로딩
    )
    if rank == 0: logger.info(f"✅ Training dataset initialized: {len(train_dataset)} samples.")

    # 모델 초기화
    # ============== 🎯 메모리 최적화 모델 초기화 ==============
    if rank == 0: logger.info("🏗️ Initializing AFT-Net model...")

    # GPU 메모리 정리 및 체크
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(current_process_device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(current_process_device) / 1e9
        if rank == 0:
            logger.info(f"🔍 Pre-model GPU memory: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB")

    # 메모리 안전 모델 생성
    try:
        model = CSFNetwork(config).to(current_process_device)
        
        # 모델 생성 후 메모리 체크
        if torch.cuda.is_available():
            memory_allocated_after = torch.cuda.memory_allocated(current_process_device) / 1e9
            if rank == 0:
                logger.info(f"✅ Model created. GPU memory after: {memory_allocated_after:.2f}GB")
                
    except RuntimeError as e:
        if "out of memory" in str(e):
            if rank == 0:
                logger.error(f"❌ GPU OOM during model creation. Try reducing batch_size or model size.")
                logger.error(f"   Current GPU memory: {torch.cuda.memory_allocated(current_process_device) / 1e9:.2f}GB")
            raise e
        else:
            raise e

    if is_ddp:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    # ========================================================

    # 손실 함수 및 옵티마이저, 스케줄러, EMA 초기화
    if hasattr(config, 'loss_weights'): lw = config.loss_weights
    else: lw = ConfigNamespace({})
    vgg_loss_config_ns = getattr(config.loss_weights, 'vgg_perceptual_loss', ConfigNamespace({}))
    resolved_vgg_path = os.path.expanduser(getattr(vgg_loss_config_ns, 'local_vgg_weights_path', ''))
    
    optimizer = optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs, eta_min=config.train.min_lr)
    
    ema_model = deepcopy(model) if getattr(config.train, 'use_ema', False) and rank == 0 else None

    # ============== 🎯 핵심 수정 1: 손실함수 한 번만 생성 ==============
    # VGG 손실함수를 매 에포크마다 새로 생성하지 않고 한 번만 생성
    fusion_method = getattr(config.model, 'fusion_method', 'region_based')
    base_loss_weights = {
        'l1_masked_weight': getattr(lw, 'l1_masked_weight', 1.0),
        'perceptual_weight': getattr(lw, 'perceptual_weight', 0.1),
        'confidence_bce_weight': getattr(lw, 'confidence_bce_weight', 0.5),
        'confidence_error_correlation_weight': getattr(lw, 'confidence_error_correlation_weight', 0.2),
        'boundary_loss_weight': getattr(lw, 'boundary_loss_weight', 0.1),
        'vgg_feature_layers': tuple(getattr(vgg_loss_config_ns, 'feature_layer_indices', [])),
        'vgg_style_layers': tuple(getattr(vgg_loss_config_ns, 'style_layer_indices', [])),
        'vgg_style_weight': getattr(vgg_loss_config_ns, 'style_weight', 0.0),
        'vgg_loss_type': getattr(vgg_loss_config_ns, 'loss_type', 'l1'),
        'vgg_weights_path': resolved_vgg_path,
        'device_for_vgg': str(current_process_device), 
        'logger': logger
    }
    
    if fusion_method == "region_based":
        base_loss_weights.update({
            'region_consistency_weight': getattr(lw, 'region_consistency_weight', 0.2),
            'hierarchical_consistency_weight': getattr(lw, 'hierarchical_consistency_weight', 0.1),
            'selection_consistency_weight': getattr(lw, 'selection_consistency_weight', 0.1),
            'edge_preservation_weight': getattr(lw, 'edge_preservation_weight', 0.1),
            'color_consistency_weight': getattr(lw, 'color_consistency_weight', 0.1),
        })
    
    # 기본 손실함수 한 번만 생성 (VGG 메모리 누수 방지)
    base_criterion = create_enhanced_criterion(fusion_method, base_loss_weights, current_process_device, rank, logger)
    # ===============================================================

    # 체크포인트 로드
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, logger, current_process_device)
            if ema_model is not None:
                try:
                    ema_checkpoint_path = args.resume.replace('.pth', '_ema.pth')
                    if os.path.exists(ema_checkpoint_path):
                        ema_state = torch.load(ema_checkpoint_path, map_location=current_process_device)
                        ema_model.load_state_dict(ema_state['model_state_dict'])
                        logger.info(f"✅ EMA model loaded from {ema_checkpoint_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load EMA model: {e}")
        else:
            if rank == 0:
                logger.warning(f"⚠️ Resume checkpoint not found: {args.resume}. Starting from scratch.")

    if rank == 0:
        logger.info(f"🚀 Starting high-performance training from epoch {start_epoch + 1} to {config.train.epochs}...")

    last_batch_for_viz = None
    
    # 훈련 루프 시작
    for epoch in range(start_epoch, config.train.epochs):
        if is_ddp: train_sampler.set_epoch(epoch)
        model.train()
        
        # ============== 🎯 핵심 수정 2: 가중치만 업데이트, 손실함수 재생성 안함 ==============
        # 매 에포크마다 새로운 손실함수를 만들지 않고, 기존 손실함수의 가중치만 업데이트
        current_loss_weights = get_adaptive_loss_weights(lw, epoch, config.train.epochs, fusion_method)
        
        # 손실함수 가중치 업데이트 (메모리 누수 방지)
        if hasattr(base_criterion, 'update_weights'):
            base_criterion.update_weights(current_loss_weights)
        else:
            # 백업: 가중치 직접 업데이트
            for key, value in current_loss_weights.items():
                if hasattr(base_criterion, key):
                    setattr(base_criterion, key, value)
        
        criterion = base_criterion  # 기존 손실함수 재사용
        # ===============================================================================
        
        epoch_loss_aggregator = {
            'total_loss': 0.0, 'l1_masked': 0.0, 'perceptual': 0.0, 'conf_bce': 0.0, 
            'conf_err_corr': 0.0, 'boundary': 0.0, 'p_content': 0.0, 'p_style': 0.0,
            'region_consistency': 0.0, 'hierarchical_consistency': 0.0, 'selection_consistency': 0.0,
            'edge_preservation': 0.0, 'color_consistency': 0.0
        }

        current_lr = 0.0 

        progress_bar = tqdm(train_loader, desc=f"🎯 Epoch {epoch+1}/{config.train.epochs}", disable=(rank != 0), leave=True)
        for batch_idx, batch_data in enumerate(progress_bar):
            
            is_batch_valid = True
            if 'valid' in batch_data:
                valid_data = batch_data['valid']
                if isinstance(valid_data, torch.Tensor):
                    if not valid_data.all(): is_batch_valid = False
                elif not valid_data:
                    is_batch_valid = False
            
            if not is_batch_valid:
                if rank == 0: logger.warning(f"⚠️ Skipping invalid batch at E{epoch+1} B{batch_idx}.")
                continue

            partial_image = batch_data['partial_image'].to(current_process_device, non_blocking=True)
            original_mask = batch_data['original_mask'].to(current_process_device, non_blocking=True)
            candidate_images_kmax = batch_data['candidate_images_kmax'].to(current_process_device, non_blocking=True)
            ground_truth_image = batch_data['ground_truth_image'].to(current_process_device, non_blocking=True)

            # 🔧 새로 추가: candidate_scores 정보 처리
            candidate_scores = batch_data.get('candidate_scores', None)
            if candidate_scores is not None:
                candidate_scores = candidate_scores.to(current_process_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            try:
                with torch.cuda.amp.autocast(enabled=getattr(config.train, 'use_amp', False)):
                    # 🔧 모델에 candidate_scores 정보 전달 방법
                    # CSFNetwork에 임시로 저장하여 forward에서 사용
                    if hasattr(model, 'module'):  # DDP인 경우
                        model.module._current_candidate_scores = candidate_scores
                    else:
                        model._current_candidate_scores = candidate_scores
                    
                    model_outputs = model(partial_image, original_mask, candidate_images_kmax)
                    predicted_pixels_logits, confidence_map_logits, model_info_for_loss = model_outputs
                    
                    # Score 정보를 loss 계산에도 전달
                    if model_info_for_loss is None:
                        model_info_for_loss = {}
                    model_info_for_loss['candidate_scores'] = candidate_scores
                    
                    loss_dict_batch = criterion(
                        predicted_pixels_logits, confidence_map_logits, ground_truth_image, 
                        original_mask, model_info_for_loss
                    )
                    total_loss_batch = loss_dict_batch['total_loss']

                if torch.isnan(total_loss_batch) or torch.isinf(total_loss_batch) or total_loss_batch.item() > 5.0:
                    if rank == 0: logger.warning(f"🚨 Unstable loss: {total_loss_batch.item():.4f}. Skipping batch.")
                    continue


                if rank == 0:
                    print(
                        f"[Epoch {epoch+1} | Batch {batch_idx+1}] "
                        f"L1: {loss_dict_batch.get('l1_masked', torch.tensor(0)).item():.4f}, "
                        f"Percep: {loss_dict_batch.get('perceptual', torch.tensor(0)).item():.4f}, "
                        f"Region: {loss_dict_batch.get('region_consistency', torch.tensor(0)).item():.4f}, "
                        f"Hier: {loss_dict_batch.get('hierarchical_consistency', torch.tensor(0)).item():.4f}, "
                        f"Select: {loss_dict_batch.get('selection_consistency', torch.tensor(0)).item():.4f}, "
                        f"Edge: {loss_dict_batch.get('edge_preservation', torch.tensor(0)).item():.4f}, "
                        f"Color: {loss_dict_batch.get('color_consistency', torch.tensor(0)).item():.4f}, "
                        f"Total: {loss_dict_batch.get('total_loss', torch.tensor(0)).item():.4f}"
                    )



                total_loss_batch.backward()
                
                max_grad_norm = getattr(config.train, 'clip_grad_norm', 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    if rank == 0: logger.warning(f"NaN/Inf gradient norm. Skipping step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                current_lr = optimizer.param_groups[0]['lr']

            except Exception as e:
                if rank == 0: logger.error(f"Error during train step: {e}\n{traceback.format_exc(limit=1)}")
                continue

            if ema_model is not None and rank == 0:
                update_ema_model(ema_model.module if is_ddp else ema_model, model.module if is_ddp else model)
            
            for key in epoch_loss_aggregator.keys():
                if key in loss_dict_batch and torch.is_tensor(loss_dict_batch[key]):
                    item_loss_detached = loss_dict_batch[key].detach()
                    if is_ddp: dist.all_reduce(item_loss_detached, op=dist.ReduceOp.SUM)
                    epoch_loss_aggregator[key] += item_loss_detached.item() / (world_size if is_ddp else 1)
            
            if rank == 0 and (batch_idx + 1) % config.train.log_interval == 0:
                progress_bar.set_postfix(loss=f"{total_loss_batch.item():.4f}", lr=f"{current_lr:.2e}")

            if batch_idx == len(train_loader) - 1:
                last_batch_for_viz = {k: v.cpu() for k, v in batch_data.items() if isinstance(v, torch.Tensor)}
                if 'path' in batch_data: last_batch_for_viz['path'] = batch_data.get('path')

        if rank == 0:
            progress_bar.close()
        
        # 에포크 요약 로그
        if rank == 0 and len(train_loader) > 0:
            log_epoch_summary = [f"🎯 Epoch {epoch+1} Performance Summary LR: {current_lr:.2e}"]
            for lname, lval_total_epoch in epoch_loss_aggregator.items():
                avg_lval_epoch = lval_total_epoch / len(train_loader)
                log_epoch_summary.append(f"{lname[:12]}: {avg_lval_epoch:.4f}")
            logger.info(" | ".join(log_epoch_summary))

            wandb_epoch_log_data = {}
            for lname, lval_total_epoch in epoch_loss_aggregator.items():
                if len(train_loader) > 0:
                    avg_lval_epoch = lval_total_epoch / len(train_loader)
                    wandb_epoch_log_data[f"epoch_train/avg_{lname}"] = avg_lval_epoch
            
            wandb_epoch_log_data["epoch_train/epoch"] = epoch + 1
            wandb_epoch_log_data["epoch_train/learning_rate"] = current_lr
            wandb_epoch_log_data["epoch_train/training_progress"] = epoch / config.train.epochs
            wandb_epoch_log_data["epoch_train/performance_optimization"] = 1.0
            
            total_avg = epoch_loss_aggregator.get('total_loss', 0) / len(train_loader) if len(train_loader) > 0 else 0
            l1_avg = epoch_loss_aggregator.get('l1_masked', 0) / len(train_loader) if len(train_loader) > 0 else 0
            perceptual_avg = epoch_loss_aggregator.get('perceptual', 0) / len(train_loader) if len(train_loader) > 0 else 0
            
            wandb_epoch_log_data["epoch_train/performance_score"] = 1.0 / (1.0 + total_avg)
            wandb_epoch_log_data["epoch_train/pixel_accuracy"] = 1.0 / (1.0 + l1_avg)
            wandb_epoch_log_data["epoch_train/perceptual_quality"] = 1.0 / (1.0 + perceptual_avg)
            
            wandb.log(wandb_epoch_log_data, step=(epoch + 1) * len(train_loader))

        # 스케줄러 스텝
        if scheduler is not None:
            scheduler.step()
            if rank == 0:
                current_lr_after_step = scheduler.get_last_lr()[0]
                logger.info(f"📈 Epoch {epoch+1} completed. Scheduler updated LR to: {current_lr_after_step:.2e}")

        if rank == 0 and (epoch + 1) % config.train.save_interval == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, 
                            os.path.join(config.output_dir, "checkpoints", f"epoch_{epoch+1}.pth"), logger)

        # ============== 🎯 시각화 부분 전체 교체 ==============
        # 기존 시각화 코드 블록을 찾아서 아래로 교체: -> log만 저장하고 이미지 시각화는 안하는 부분. ( 일단 학습우선이면 이거 활성화 시키기 )
        visualize_interval = getattr(config.train, 'visualize_interval_epoch', 0)
        if rank == 0 and visualize_interval > 0 and (epoch + 1) % visualize_interval == 0:
            if last_batch_for_viz is not None:
                try:
                    logger.info(f"🎨 Attempting visualization for epoch {epoch + 1}...")
                    
                    # 간단한 시각화 대신 상태 정보만 로깅
                    logger.info(f"📊 Epoch {epoch + 1} Training Status:")
                    logger.info(f"   - Batch count: {len(train_loader)}")
                    logger.info(f"   - Last batch shape: {last_batch_for_viz.get('partial_image', torch.tensor([0])).shape}")
                    logger.info(f"   - GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                    
                    # 실제 시각화는 건너뛰기
                    logger.info(f"✅ Visualization status logged for epoch {epoch + 1}")
                    
                except Exception as e:
                    logger.error(f"Error during visualization logging: {e}")
        # ==============================================================



        # ============== 🎯 핵심 수정 4: DDP 동기화 강화 ==============
        # 모든 프로세스가 에포크 완료를 기다림 (동기화 문제 방지)
        if is_ddp:
            try:
                # PyTorch 1.x 호환성: timeout 파라미터 제거
                dist.barrier()
            except Exception as barrier_error:
                logger.error(f"❌ DDP barrier failed at epoch {epoch+1}: {barrier_error}")
                if rank == 0:
                    logger.error("🚨 DDP synchronization lost. Training may be unstable.")
        # ==========================================================

    # 훈련 완료 후 처리
    if rank == 0:
        logger.info("🎉 High-performance training completed!")
        final_model_path = os.path.join(config.output_dir, "checkpoints", f"{config.exp_name}_final_performance.pth")
        save_checkpoint(config.train.epochs - 1, model, optimizer, scheduler, final_model_path, logger, is_final=True)
        if ema_model is not None:
            final_ema_path = os.path.join(config.output_dir, "checkpoints", f"{config.exp_name}_final_ema_best_performance.pth")
            torch.save({
                'epoch': config.train.epochs - 1,
                'model_state_dict': ema_model.state_dict(),
                'ema_decay': getattr(config.train, 'ema_decay', 0.999),
            }, final_ema_path)
            logger.info(f"🏆 Best performance EMA model saved: {final_ema_path}")
            
        logger.info("🎯 Training Performance Summary:")
        logger.info(f"   - Fusion Method: {fusion_method}")
        logger.info(f"   - Pixel-level Optimization: {'✅ Enabled' if fusion_method == 'region_based' else '❌ Disabled'}")
        logger.info(f"   - EMA Model: {'✅ Used' if ema_model is not None else '❌ Not Used'}")
        logger.info(f"   - Curriculum Learning: ✅ Enhanced")
        logger.info(f"   - Advanced LR Schedule: ✅ Enabled")
        logger.info(f"   - Performance Mode: 🚀 Maximum")
    
    if is_ddp:
        cleanup_ddp()


def main():
    global PROJECT_ROOT_FOR_PATHS 
    PROJECT_ROOT_FOR_PATHS = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="AFT-Net High Performance Training Script")
    parser.add_argument('--config_path', type=str, default='./configs/aft_config.yaml', help='Path to the YAML config file.')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from.')
    parser.add_argument('--skip_visualization', action='store_true', help='Skip visualization steps during training.')
    parser.add_argument('--performance_mode', action='store_true', default=True, help='Enable maximum performance optimization.')
    args = parser.parse_args()
    
    config = load_config_from_yaml(args.config_path)
    
    # 성능 모드 활성화
    if args.performance_mode:
        print("🚀 High Performance Mode Enabled")
        # 성능 최적화 설정 강제 적용
        if hasattr(config.model, 'fusion_method'):
            if config.model.fusion_method != 'region_based':
                print(f"⚡ Switching to region_based fusion for maximum performance")
                config.model.fusion_method = 'region_based'
        
        # 픽셀 레벨 최적화 강제 활성화
        if hasattr(config.model, 'region_based_selector'):
            config.model.region_based_selector.final_region_size = 1
            print("💎 Pixel-level optimization enforced")
    
    if args.skip_visualization:
        if not hasattr(config.train, 'visualize_interval_epoch'):
            setattr(config.train, 'visualize_interval_epoch', 0)
        else:
            config.train.visualize_interval_epoch = 0
    
    # 디렉토리 생성
    is_main_process_for_dirs = not config.distributed.use_ddp or int(os.environ.get("RANK", "0")) == 0
    if is_main_process_for_dirs:
        try:
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(os.path.join(config.output_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(config.output_dir, "visualizations"), exist_ok=True)
            os.makedirs(os.path.join(config.output_dir, "logs"), exist_ok=True)
            print(f"📁 Output directories created: {config.output_dir}")
        except Exception as e: 
            print(f"❌ Error creating output dirs: {e}")
            return

    # 분산 훈련 시작
    if config.distributed.use_ddp:
        world_size_env = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        rank_env = int(os.environ.get("RANK", -1))
        if rank_env == -1: 
            print("❌ Error: RANK not set. Use 'torchrun'. Exiting.")
            return
        if hasattr(config, 'world_size') and config.world_size != world_size_env and rank_env == 0:
            print(f"⚠️ Warning: Config world_size ({config.world_size}) != DDP WORLD_SIZE ({world_size_env}). Using DDP's.")
        
        print(f"🔥 Starting DDP training with {world_size_env} GPUs")
        train_worker(rank_env, world_size_env, config, args)
    else:
        print("🖥️ Running in single GPU/CPU mode.")
        train_worker(0, 1, config, args)

    # WandB 종료
    if not config.distributed.use_ddp or int(os.environ.get("RANK", "0")) == 0:
        if wandb.run is not None:
            wandb.finish()
            print("📊 WandB session finished")


if __name__ == '__main__':
    print("🚀 AFT-Net High Performance Training")
    print("💎 Focus: Maximum Quality over Memory Efficiency")
    main()