# data_utils/candidate_processing.py
# 기존 함수들 + MSE+LPIPS 결합 스코어링 추가

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageStat
import cv2
import numpy as np
import scipy.ndimage as ndimage
import lpips

def detect_mask_shape_filling_relaxed(image_tensor, mask_tensor, original_mask_tensor, debug_info=None):
    """
    기존 코드 기반의 더 관대한 텍스처 필터링
    명백히 단순한 경우만 제거
    """
    if mask_tensor.sum() < 50:
        return True  # 작은 마스크는 제거
    
    try:
        # 마스크 차원 맞추기
        if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
            mask_2d = mask_tensor.squeeze(0)
        else:
            mask_2d = mask_tensor
        
        mask_bool = mask_2d > 0.5
        
        # 🔥 기존 코드의 관대한 임계값 사용
        
        # 1. 색상 분산 체크 (더 느슨하게)
        color_variances = []
        for c in range(image_tensor.shape[0]):
            channel = image_tensor[c]
            channel_pixels = channel[mask_bool]
            if len(channel_pixels) > 0:
                var = torch.var(channel_pixels).item()
                color_variances.append(var)
        
        avg_color_variance = np.mean(color_variances) if color_variances else 0.0
        low_color_variance = avg_color_variance < 0.02  # 🔥 0.05 → 0.02로 더 엄격하게 (명백한 경우만)
        
        # 2. 색상 범위 체크 (더 느슨하게)
        color_ranges = []
        for c in range(image_tensor.shape[0]):
            channel = image_tensor[c]
            channel_pixels = channel[mask_bool]
            if len(channel_pixels) > 0:
                color_range = (torch.max(channel_pixels) - torch.min(channel_pixels)).item()
                color_ranges.append(color_range)
        
        avg_color_range = np.mean(color_ranges) if color_ranges else 0.0
        narrow_color_range = avg_color_range < 0.1  # 🔥 0.3 → 0.1로 더 엄격하게
        
        # 3. 고유 색상 수 체크 (더 느슨하게)
        unique_colors_count = 0
        for c in range(image_tensor.shape[0]):
            channel = image_tensor[c]
            channel_pixels = channel[mask_bool]
            if len(channel_pixels) > 0:
                # 소수점 둘째자리까지만 고려하여 고유값 계산
                unique_vals = torch.unique(torch.round(channel_pixels * 100) / 100)
                unique_colors_count += len(unique_vals)
        
        avg_unique_colors = unique_colors_count / image_tensor.shape[0]
        few_unique_colors = avg_unique_colors < 5  # 🔥 20 → 5로 더 엄격하게
        
        # 4. 엣지 검출 (더 느슨하게)
        if image_tensor.shape[0] >= 3:  # RGB 이미지인 경우
            # 그레이스케일 변환
            gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            
            # 간단한 gradient 계산
            gray_np = gray.cpu().numpy()
            mask_np = mask_bool.cpu().numpy()
            
            # Sobel 필터 적용
            grad_x = ndimage.sobel(gray_np, axis=0)
            grad_y = ndimage.sobel(gray_np, axis=1)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 마스크 영역 내 평균 gradient
            masked_gradient = gradient_magnitude[mask_np]
            avg_gradient = np.mean(masked_gradient) if len(masked_gradient) > 0 else 0
            
            low_gradient = avg_gradient < 0.01  # 🔥 0.05 → 0.01로 더 엄격하게
        else:
            low_gradient = False
        
        # 5. 표준편차 체크 (더 느슨하게)
        std_devs = []
        for c in range(image_tensor.shape[0]):
            channel = image_tensor[c]
            channel_pixels = channel[mask_bool]
            if len(channel_pixels) > 0:
                std_dev = torch.std(channel_pixels).item()
                std_devs.append(std_dev)
        
        avg_std_dev = np.mean(std_devs) if std_devs else 0.0
        low_std_dev = avg_std_dev < 0.02  # 🔥 0.1 → 0.02로 더 엄격하게
        
        # 🔥 조건 조합 - 모든 조건을 동시에 만족해야 제거 (AND 조건)
        # 명백히 단순한 경우만 제거
        is_simple_filling = (
            low_color_variance and 
            narrow_color_range and 
            few_unique_colors and
            low_gradient and
            low_std_dev
        )
        
        if debug_info:
            sample_id, idx = debug_info
            print(f"  [관대한 텍스처 분석] Sample {sample_id}, Candidate {idx}:")
            print(f"    색상 분산: {avg_color_variance:.6f} (< 0.02: {low_color_variance})")
            print(f"    색상 범위: {avg_color_range:.4f} (< 0.1: {narrow_color_range})")
            print(f"    고유 색상 수: {avg_unique_colors:.1f} (< 5: {few_unique_colors})")
            print(f"    평균 gradient: {avg_gradient:.6f} (< 0.01: {low_gradient})")
            print(f"    표준편차: {avg_std_dev:.6f} (< 0.02: {low_std_dev})")
            print(f"    → {'단순 채우기 (제거)' if is_simple_filling else '복잡한 텍스처 (유지)'}")
        
        return is_simple_filling
        
    except Exception as e:
        if debug_info:
            sample_id, idx = debug_info
            print(f"  [텍스처 분석 오류] Sample {sample_id}, Candidate {idx}: {e}")
        return False  # 🔥 오류 발생시 통과 (관대하게)

def detect_mask_shape_filling(image_tensor, mask_tensor, original_mask_tensor, debug_info=None):
    """
    🔥 극도로 엄격한 필터링 - 완벽한 텍스처만 통과
    조금이라도 의심되는 것은 모두 제거
    기존 함수명 유지하여 호환성 보장
    """
    if mask_tensor.sum() < 100:  # 작은 마스크는 무조건 제거
        return True
    
    try:
        # 마스크 차원 맞추기
        if mask_tensor.dim() == 3 and mask_tensor.shape[0] == 1:
            mask_2d = mask_tensor.squeeze(0)
        else:
            mask_2d = mask_tensor
        
        mask_bool = mask_2d > 0.5
        mask_area = mask_bool.sum().item()
        
        if mask_area < 200:  # 너무 작은 영역은 무조건 제거
            return True
        
        # ============ 🔥 Level 1: 기본 통계 검사 (매우 엄격) ============
        
        # 1. 극도로 엄격한 색상 분산 검사
        color_variances = []
        color_means = []
        color_stds = []
        
        for c in range(image_tensor.shape[0]):
            channel = image_tensor[c]
            channel_pixels = channel[mask_bool]
            if len(channel_pixels) > 0:
                var = torch.var(channel_pixels).item()
                mean = torch.mean(channel_pixels).item()
                std = torch.std(channel_pixels).item()
                
                color_variances.append(var)
                color_means.append(mean)
                color_stds.append(std)
        
        avg_variance = np.mean(color_variances) if color_variances else 0.0
        avg_std = np.mean(color_stds) if color_stds else 0.0
        
        # 🔥 극도로 높은 임계값
        if avg_variance < 0.15:  # 분산이 0.15 미만이면 제거
            return True
        if avg_std < 0.2:  # 표준편차가 0.2 미만이면 제거
            return True
        
        # 2. 색상 범위 검사 (매우 엄격)
        color_ranges = []
        for c in range(image_tensor.shape[0]):
            channel = image_tensor[c]
            channel_pixels = channel[mask_bool]
            if len(channel_pixels) > 0:
                color_range = (torch.max(channel_pixels) - torch.min(channel_pixels)).item()
                color_ranges.append(color_range)
        
        avg_range = np.mean(color_ranges) if color_ranges else 0.0
        if avg_range < 0.6:  # 색상 범위가 0.6 미만이면 제거
            return True
        
        # ============ 🔥 Level 2: 고급 텍스처 분석 ============
        
        # 3. 엣지 검출 (매우 엄격)
        if image_tensor.shape[0] >= 3:
            gray = 0.299 * image_tensor[0] + 0.587 * image_tensor[1] + 0.114 * image_tensor[2]
            gray_np = gray.cpu().numpy()
            mask_np = mask_bool.cpu().numpy()
            
            # Sobel 엣지 검출
            grad_x = ndimage.sobel(gray_np, axis=1)
            grad_y = ndimage.sobel(gray_np, axis=0)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Canny 엣지 검출 추가 (opencv 없을 수도 있으니 try-except)
            try:
                gray_uint8 = (np.clip(gray_np, 0, 1) * 255).astype(np.uint8)
                edges = cv2.Canny(gray_uint8, 50, 150)
                edge_density = np.sum(edges[mask_np] > 0) / mask_area
            except:
                edge_density = 0
            
            masked_gradient = gradient_magnitude[mask_np]
            avg_gradient = np.mean(masked_gradient) if len(masked_gradient) > 0 else 0
            
            if avg_gradient < 0.15:  # 그래디언트가 0.15 미만이면 제거
                return True
            if edge_density < 0.1:  # 엣지 밀도가 10% 미만이면 제거
                return True
        
        # 디버깅 출력
        if debug_info:
            sample_id, idx = debug_info
            print(f"  [🔥 ULTRA STRICT] Sample {sample_id}, Candidate {idx}:")
            print(f"    분산: {avg_variance:.4f}, 표준편차: {avg_std:.4f}, 범위: {avg_range:.4f}")
            print(f"    → 복잡한 텍스처로 판정 (유지)")
        
        # 모든 검사를 통과한 경우에만 False 반환 (유지)
        return False
        
    except Exception as e:
        if debug_info:
            sample_id, idx = debug_info
            print(f"  [분석 오류] Sample {sample_id}, Candidate {idx}: {e}")
        # 오류 발생시 안전하게 제거
        return True

def filter_candidate_basic(candidate_pil_image, candidate_amodal_mask_pil,
                          original_mask_pil,
                          min_overlap_ratio=0.15,
                          min_amodal_area_ratio=0.01,
                          min_texture_variance=0.01,
                          img_H_W=(256,256),
                          idx=None, sample_id=None):
    """
    기존 필터링 로직 + mask 영역과의 겹침 체크 추가
    """
    if candidate_amodal_mask_pil is None: 
        return False
    
    to_tensor = TF.to_tensor
    try:
        # 마스크 해석
        # candidate_amodal_mask: 0=객체존재, 255=빈영역
        # original_mask: 0=mask영역(채워야할부분), 255=visible영역
        cand_amodal_mask_tensor = (to_tensor(candidate_amodal_mask_pil.convert('L')) < 0.5).float()  # 객체 영역을 1로
        orig_mask_tensor = (to_tensor(original_mask_pil.convert('L')) > 0.5).float()  # visible 영역을 1로
        candidate_tensor = to_tensor(candidate_pil_image.resize(img_H_W, Image.BILINEAR))
        
    except Exception as e:
        if sample_id and idx is not None:
            print(f"Sample {sample_id}, Candidate {idx}: 텐서 변환 실패 - {e}")
        return False
    
    total_pixels = float(img_H_W[0] * img_H_W[1])
    amodal_area = torch.sum(cand_amodal_mask_tensor)
    
    # 1. 면적 체크
    if (amodal_area / total_pixels) < min_amodal_area_ratio:
        if sample_id and idx is not None:
            print(f"Sample {sample_id}, Candidate {idx}: 면적 부족 {amodal_area/total_pixels:.4f} < {min_amodal_area_ratio}")
        return False
    
    # 2. 🔥 핵심 수정: mask 영역(채워야 할 부분)과의 겹침 확인
    orig_mask_area = 1.0 - orig_mask_tensor  # mask 영역 (0이었던 부분을 1로)
    
    # 후보 객체가 mask 영역과 겹치는 부분
    intersection_with_mask = torch.sum(cand_amodal_mask_tensor * orig_mask_area)
    
    # 겹침 비율: (후보 ∩ mask영역) / 후보전체
    overlap_with_mask_area = intersection_with_mask / (amodal_area + 1e-8)
    
    # 상세한 디버깅 정보 출력
    if sample_id and idx is not None:
        orig_visible_area = torch.sum(orig_mask_tensor)
        orig_mask_area_sum = torch.sum(orig_mask_area)
        print(f"Sample {sample_id}, Candidate {idx}:")
        print(f"  원본 visible 영역: {orig_visible_area.item()}/{total_pixels} ({orig_visible_area/total_pixels:.4f})")
        print(f"  원본 mask 영역: {orig_mask_area_sum.item()}/{total_pixels} ({orig_mask_area_sum/total_pixels:.4f})")
        print(f"  후보 객체 영역: {amodal_area.item()}/{total_pixels} ({amodal_area/total_pixels:.4f})")
        print(f"  교집합 (후보 ∩ mask): {intersection_with_mask.item()}/{total_pixels} ({intersection_with_mask/total_pixels:.4f})")
        print(f"  겹침 비율 (교집합/후보): {overlap_with_mask_area:.4f} (threshold: {min_overlap_ratio})")
    
    # 3. 🔥 핵심: mask 영역과 겹치지 않으면 무조건 제거
    if overlap_with_mask_area < min_overlap_ratio:
        if sample_id and idx is not None:
            print(f"  → mask 영역과 겹침 부족으로 필터링 OUT (후보가 채워야 할 영역을 채우지 않음)")
        return False
    
    # 4. 추가 체크: mask 영역과 아예 겹치지 않는 경우 무조건 제거
    if intersection_with_mask.item() < 10:  # 최소 10픽셀은 겹쳐야 함
        if sample_id and idx is not None:
            print(f"  → mask 영역과 거의 겹치지 않아서 필터링 OUT")
        return False
    
    # 5. 텍스처 검사 (관대하게)
    try:
        orig_mask_for_analysis = (to_tensor(original_mask_pil.resize(img_H_W, Image.NEAREST).convert('L')) > 0.5).float()
        
        debug_info = (sample_id, idx) if sample_id and idx is not None else None
        is_mask_filling = detect_mask_shape_filling_relaxed(
            candidate_tensor, 
            cand_amodal_mask_tensor,
            orig_mask_for_analysis,
            debug_info=debug_info
        )
        
        if is_mask_filling:
            if sample_id and idx is not None:
                print(f"  → 텍스처 검사로 필터링 OUT")
            return False
            
    except Exception as e:
        if sample_id and idx is not None:
            print(f"  → 텍스처 분석 실패, 통과시킴: {e}")
        # 분석 실패시 다른 조건만으로 판단 (통과)
    
    if sample_id and idx is not None:
        print(f"  → 모든 필터링 PASS")

    return True

def calculate_lpips_consistency_score(candidate_pil, original_pil, original_mask_pil, 
                                    candidate_amodal_mask_pil, lpips_model, device, img_size=256):
    """
    기존 LPIPS 일치도 점수 계산 함수
    """
    try:
        # 이미지들을 텐서로 변환하고 크기 맞추기
        to_tensor = TF.to_tensor
        
        candidate_tensor = to_tensor(candidate_pil.resize((img_size, img_size), Image.BILINEAR)).to(device)
        original_tensor = to_tensor(original_pil.resize((img_size, img_size), Image.BILINEAR)).to(device)
        
        # 마스크 처리
        original_mask_tensor = (to_tensor(original_mask_pil.resize((img_size, img_size), Image.NEAREST)) > 0.5).float().to(device)
        candidate_amodal_mask_tensor = (to_tensor(candidate_amodal_mask_pil.resize((img_size, img_size), Image.NEAREST)) < 0.5).float().to(device)
        
        # visible 영역에서의 비교를 위한 마스크
        comparison_mask = original_mask_tensor * candidate_amodal_mask_tensor
        
        if comparison_mask.sum() < 10:  # 비교할 영역이 너무 작으면
            return 0.0
        
        # LPIPS 계산을 위해 전체 이미지 사용 (마스킹은 나중에)
        candidate_for_lpips = candidate_tensor.unsqueeze(0)  # [1, 3, H, W]
        original_for_lpips = original_tensor.unsqueeze(0)    # [1, 3, H, W]
        
        # LPIPS 계산 (0~1 범위, 낮을수록 유사)
        with torch.no_grad():
            lpips_distance = lpips_model(candidate_for_lpips, original_for_lpips).item()
        
        # LPIPS를 0~1 점수로 변환 (높을수록 좋음)
        lpips_score = max(0.0, 1.0 - lpips_distance)
        
        return lpips_score
        
    except Exception as e:
        print(f"LPIPS 계산 중 오류: {e}")
        return 0.0

def calculate_mse_consistency_score(candidate_pil, original_pil, original_mask_pil, 
                                   candidate_amodal_mask_pil, device, img_size=256,
                                   method="original"):
    """
    MSE 기반 픽셀 일치도 점수 계산 (호환성 유지 + 개선 옵션)
    
    Args:
        candidate_pil: 후보 이미지
        original_pil: 원본 이미지  
        original_mask_pil: 원본 마스크 (visible 영역)
        candidate_amodal_mask_pil: 후보의 amodal 마스크
        device: 계산 디바이스
        img_size: 이미지 크기
        method: "original" (기존방식) 또는 "size_adjusted" (개선방식)
    
    Returns:
        float: MSE 일치도 점수 (높을수록 좋음, 0~1 범위)
    """
    try:
        # 이미지들을 텐서로 변환하고 크기 맞추기
        to_tensor = TF.to_tensor
        
        candidate_tensor = to_tensor(candidate_pil.resize((img_size, img_size), Image.BILINEAR)).to(device)
        original_tensor = to_tensor(original_pil.resize((img_size, img_size), Image.BILINEAR)).to(device)
        
        # 마스크 처리 (visible 영역만 비교)
        original_mask_tensor = (to_tensor(original_mask_pil.resize((img_size, img_size), Image.NEAREST)) > 0.5).float().to(device)
        candidate_amodal_mask_tensor = (to_tensor(candidate_amodal_mask_pil.resize((img_size, img_size), Image.NEAREST)) < 0.5).float().to(device)
        
        # visible 영역에서의 비교를 위한 마스크 (original visible & candidate object)
        comparison_mask = original_mask_tensor * candidate_amodal_mask_tensor
        
        if comparison_mask.sum() < 10:  # 비교할 영역이 너무 작으면
            return 0.0
        
        # visible 영역에서만 MSE 계산
        masked_candidate = candidate_tensor * comparison_mask
        masked_original = original_tensor * comparison_mask
        
        # 픽셀별 제곱 오차
        squared_diff = (masked_candidate - masked_original) ** 2
        
        # 비교 영역에서의 평균 MSE
        mse = squared_diff.sum() / (comparison_mask.sum() * candidate_tensor.shape[0] + 1e-8)
        
        if method == "size_adjusted":
            # 🔥 개선된 방식: 크기 보정 적용
            comparison_area = comparison_mask.sum().item()
            total_pixels = img_size * img_size
            area_ratio = comparison_area / total_pixels
            
            # 크기 보정: 작은 영역일수록 더 관대하게 평가
            if area_ratio < 0.05:
                size_adjustment = 1.0 + (0.05 - area_ratio) * 2.0  # 최대 1.1배 보정
            else:
                size_adjustment = 1.0
            
            # 적응적 임계값: 비교 영역이 작을수록 더 관대한 변환
            if area_ratio < 0.02:
                conversion_factor = 3.0  # 매우 작은 영역: 더 관대하게
            elif area_ratio < 0.05:
                conversion_factor = 4.0  # 작은 영역: 조금 관대하게  
            else:
                conversion_factor = 5.0  # 큰 영역: 기존대로
            
            mse_score = torch.exp(-mse * conversion_factor).item() * size_adjustment
        else:
            # 🔥 기존 방식 (original): 호환성 유지
            # MSE를 0~1 점수로 변환 (낮은 MSE = 높은 점수)
            mse_score = torch.exp(-mse * 5).item()  # 5는 조정 가능한 하이퍼파라미터
        
        return max(0.0, min(1.0, mse_score))
        
    except Exception as e:
        print(f"MSE 계산 중 오류: {e}")
        return 0.0

# 🔥 개선된 대안 방식 (주석처리된 버전)
"""
def calculate_mse_consistency_score_alternative(candidate_pil, original_pil, original_mask_pil, 
                                               candidate_amodal_mask_pil, device, img_size=256):
    '''
    대안적 MSE 계산 방식: 정규화된 비교 (샘플링 기반)
    후보 크기에 관계없이 공정한 평가
    '''
    try:
        to_tensor = TF.to_tensor
        
        candidate_tensor = to_tensor(candidate_pil.resize((img_size, img_size), Image.BILINEAR)).to(device)
        original_tensor = to_tensor(original_pil.resize((img_size, img_size), Image.BILINEAR)).to(device)
        
        original_mask_tensor = (to_tensor(original_mask_pil.resize((img_size, img_size), Image.NEAREST)) > 0.5).float().to(device)
        candidate_amodal_mask_tensor = (to_tensor(candidate_amodal_mask_pil.resize((img_size, img_size), Image.NEAREST)) < 0.5).float().to(device)
        
        comparison_mask = original_mask_tensor * candidate_amodal_mask_tensor
        
        if comparison_mask.sum() < 10:
            return 0.0
        
        # 샘플링 기반 비교
        comparison_pixels = comparison_mask.sum().item()
        
        if comparison_pixels > 100:
            # 큰 영역에서는 랜덤 샘플링
            mask_indices = torch.nonzero(comparison_mask.view(-1), as_tuple=False).squeeze()
            if len(mask_indices) > 100:
                sampled_indices = mask_indices[torch.randperm(len(mask_indices))[:100]]
            else:
                sampled_indices = mask_indices
                
            candidate_flat = candidate_tensor.view(candidate_tensor.shape[0], -1)
            original_flat = original_tensor.view(original_tensor.shape[0], -1)
            
            candidate_sampled = candidate_flat[:, sampled_indices]
            original_sampled = original_flat[:, sampled_indices]
            
            mse = torch.mean((candidate_sampled - original_sampled) ** 2).item()
        else:
            # 작은 영역에서는 전체 픽셀 사용
            masked_candidate = candidate_tensor * comparison_mask
            masked_original = original_tensor * comparison_mask
            squared_diff = (masked_candidate - masked_original) ** 2
            mse = squared_diff.sum() / (comparison_mask.sum() * candidate_tensor.shape[0] + 1e-8)
            mse = mse.item()
        
        mse_score = torch.exp(-torch.tensor(mse) * 5).item()
        return max(0.0, min(1.0, mse_score))
        
    except Exception as e:
        print(f"대안 MSE 계산 중 오류: {e}")
        return 0.0
"""

def calculate_combined_consistency_score(candidate_pil, original_pil, original_mask_pil,
                                       candidate_amodal_mask_pil, lpips_model, device,
                                       img_size=256, mse_weight=0.5, lpips_weight=0.5,
                                       mse_method="original"):
    """
    MSE + LPIPS 결합 점수 계산 (호환성 유지 + MSE 계산 방식 선택)
    
    Args:
        candidate_pil: 후보 이미지
        original_pil: 원본 이미지
        original_mask_pil: 원본 마스크
        candidate_amodal_mask_pil: 후보의 amodal 마스크  
        lpips_model: LPIPS 모델
        device: 계산 디바이스
        img_size: 이미지 크기
        mse_weight: MSE 점수 가중치 (기본 0.5)
        lpips_weight: LPIPS 점수 가중치 (기본 0.5)
        mse_method: MSE 계산 방식 ("original" 또는 "size_adjusted")
    
    Returns:
        dict: {'combined_score': float, 'mse_score': float, 'lpips_score': float}
    """
    # 기존 LPIPS 점수 계산 (기존 함수 재사용)
    lpips_score = calculate_lpips_consistency_score(
        candidate_pil, original_pil, original_mask_pil, 
        candidate_amodal_mask_pil, lpips_model, device, img_size
    )
    
    # MSE 점수 계산 (방식 선택 가능)
    mse_score = calculate_mse_consistency_score(
        candidate_pil, original_pil, original_mask_pil,
        candidate_amodal_mask_pil, device, img_size, method=mse_method
    )
    
    # 가중 결합
    combined_score = mse_weight * mse_score + lpips_weight * lpips_score
    
    return {
        'combined_score': combined_score,
        'mse_score': mse_score,
        'lpips_score': lpips_score,
        'mse_method': mse_method
    }

def calculate_cross_attention_relevance_score(partial_pil, candidate_pil, 
                                            feature_extractor, cross_attn_scorer, 
                                            device, img_size_feat=64):
    """
    기존 cross attention 점수 계산 함수 (호환성을 위해 유지)
    실제로는 사용하지 않지만 기존 코드에서 호출할 수 있으므로 더미 구현
    """
    # 더미 구현 - 항상 0.5 반환
    return 0.5

def score_and_select_candidates(
    original_pil, original_mask_pil, partial_pil,
    candidate_pil_list, candidate_amodal_mask_pil_list,
    Kmax,
    filter_params,
    lpips_model, feature_extractor, cross_attn_scorer,  # 기존 파라미터 유지 (호환성)
    device,
    img_size_orig=256,
    img_size_feat=64,
    mse_weight=0.5,  # 새로 추가
    lpips_weight=0.5,  # 새로 추가
    use_mse=True,  # 새로 추가
    logger=print
):
    """
    기존 함수명 유지, cross_attn 제거하고 MSE+LPIPS만 사용
    """
    if not candidate_pil_list:
        return []

    # 기존 필터링 로직 유지 (기존 함수 재사용)
    initially_filtered_candidates = []
    for i, (cand_pil, cand_amodal_mask_pil) in enumerate(zip(candidate_pil_list, candidate_amodal_mask_pil_list)):
        if filter_candidate_basic(
            cand_pil, cand_amodal_mask_pil, original_mask_pil,
            min_overlap_ratio=filter_params.get('min_overlap_ratio', 0.4),
            min_amodal_area_ratio=filter_params.get('min_amodal_area_ratio', 0.03),
            img_H_W=(img_size_orig, img_size_orig),
            idx=i, sample_id="debug"
        ):
            initially_filtered_candidates.append({'image': cand_pil, 'amodal_mask': cand_amodal_mask_pil})
    
    logger(f"필터링 결과: {len(initially_filtered_candidates)}/{len(candidate_pil_list)} 후보 생존")
    
    if not initially_filtered_candidates:
        logger(f"모든 후보가 필터링에서 제거됨")
        return []

    # 스코어링 로직: MSE + LPIPS만 사용
    scored_candidates = []
    for cand_info in initially_filtered_candidates:
        cand_pil = cand_info['image']
        cand_amodal_mask_pil = cand_info['amodal_mask']
        
        if use_mse:
            # MSE + LPIPS 결합 점수 계산
            result = calculate_combined_consistency_score(
                cand_pil, original_pil, original_mask_pil, cand_amodal_mask_pil,
                lpips_model, device, img_size_orig, mse_weight, lpips_weight
            )
            final_score = result['combined_score']
            score_details = result
        else:
            # LPIPS만 사용
            lpips_score = calculate_lpips_consistency_score(
                cand_pil, original_pil, original_mask_pil, cand_amodal_mask_pil,
                lpips_model, device, img_size=img_size_orig
            )
            final_score = lpips_score
            score_details = {'lpips_score': lpips_score, 'mse_score': None}

        scored_candidates.append({
            'image': cand_pil,
            'amodal_mask': cand_amodal_mask_pil,
            'score': final_score,
            **score_details
        })

    # 점수 기준 정렬 및 상위 Kmax개 선택
    scored_candidates.sort(key=lambda x: x['score'], reverse=True)
    selected_top_Kmax = scored_candidates[:Kmax]

    return selected_top_Kmax

