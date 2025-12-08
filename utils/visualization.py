import os
import torch
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 비대화형 백엔드 사용
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils


def debug_color_values(gt_img, pred_pix, denorm_params, logger, sample_id="debug"):
    """색상 값 디버깅을 위한 함수"""
    
    if logger:
        logger.info(f"=== 색상 디버깅 - Sample {sample_id} ===")
        
        # 1. 원본 이미지 (정규화된 상태) 통계
        logger.info(f"GT 이미지 (정규화됨): min={gt_img.min():.3f}, max={gt_img.max():.3f}, mean={gt_img.mean():.3f}")
        
        # 2. 예측 이미지 (sigmoid 후) 통계  
        logger.info(f"예측 이미지 (sigmoid): min={pred_pix.min():.3f}, max={pred_pix.max():.3f}, mean={pred_pix.mean():.3f}")
        
        # 3. 정규화 해제 파라미터
        if denorm_params:
            logger.info(f"정규화 해제 파라미터: mean={denorm_params['mean']}, std={denorm_params['std']}")
            
            # 4. GT 이미지 정규화 해제 후
            gt_denorm = denormalize_image_for_viz(gt_img, denorm_params['mean'], denorm_params['std'])
            logger.info(f"GT 정규화 해제 후: min={gt_denorm.min():.3f}, max={gt_denorm.max():.3f}, mean={gt_denorm.mean():.3f}")
            
            # 5. 예측 이미지 정규화 해제 후
            pred_denorm = denormalize_image_for_viz(pred_pix, denorm_params['mean'], denorm_params['std'])
            logger.info(f"예측 정규화 해제 후: min={pred_denorm.min():.3f}, max={pred_denorm.max():.3f}, mean={pred_denorm.mean():.3f}")
        
        # 6. 색상 채널별 분석
        for c, channel_name in enumerate(['R', 'G', 'B']):
            gt_channel = gt_img[:, c, :, :].mean()
            pred_channel = pred_pix[:, c, :, :].mean()
            logger.info(f"{channel_name} 채널 - GT: {gt_channel:.3f}, 예측: {pred_channel:.3f}")

def copy_final_image_to_collection(final_img_path, epoch, sample_idx, collection_dir, config, logger=None):
    """
    final.png 이미지를 별도의 수집 디렉토리에 복사합니다.
    """
    import os
    import shutil
    
    try:
        # 수집 디렉토리가 없으면 생성
        os.makedirs(collection_dir, exist_ok=True)
        
        # 고유한 파일 이름 생성 (에포크와 샘플 인덱스 사용)
        dest_filename = f"epoch_{epoch}_sample_{sample_idx}.png"
        dest_path = os.path.join(collection_dir, dest_filename)
        
        # 파일 복사
        shutil.copy2(final_img_path, dest_path)
        
        if logger:
            if hasattr(logger, 'debug'):
                logger.debug(f"Final image copied to collection: {dest_path}")
            else:
                print(f"Final image copied to collection: {dest_path}")
    except Exception as e:
        if logger:
            if hasattr(logger, 'error'):
                logger.error(f"Failed to copy final image to collection: {e}")
            else:
                print(f"Error: Failed to copy final image to collection: {e}")

def denormalize_image_for_viz(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    주어진 평균과 표준편차로 정규화된 이미지 텐서를 [0, 1] 범위로 되돌립니다.
    tensor: [C, H, W] 또는 [B, C, H, W]
    """
    if tensor is None:
        return None
    
    try:
        # 이미 CPU에 있는지 확인하고, 아니면 이동
        if tensor.device.type != 'cpu':
            tensor = tensor.detach().cpu()
        
        _mean = torch.tensor(mean, device='cpu')
        _std = torch.tensor(std, device='cpu')

        if tensor.dim() == 4:  # Batch
            _mean = _mean.view(1, -1, 1, 1)
            _std = _std.view(1, -1, 1, 1)
        elif tensor.dim() == 3:  # Single image
            _mean = _mean.view(-1, 1, 1)
            _std = _std.view(-1, 1, 1)
        else:  # Unexpected
            return tensor  # Or raise error

        tensor = tensor * _std + _mean
        return torch.clamp(tensor, 0, 1)
    except Exception as e:
        print(f"Error in denormalize_image_for_viz: {e}")
        return tensor  # 에러 발생 시 원본 텐서 반환

def tensor_to_pil(tensor, denorm_params=None):
    """
    이미지 텐서(단일 이미지, [C,H,W] 또는 [1,C,H,W] 또는 [H,W] 또는 [1,H,W])를 PIL 이미지로 변환합니다.
    """
    if tensor is None:
        return None
    
    try:
        # 이미 CPU에 있는지 확인하고, 아니면 이동
        if tensor.device.type != 'cpu':
            tensor = tensor.detach().cpu()
        
        if tensor.dim() == 4 and tensor.size(0) == 1:  # [1, C, H, W] -> [C, H, W]
            tensor = tensor.squeeze(0)
        
        if tensor.dim() == 3 and tensor.size(0) == 1:  # [1, H, W] for mask/conf -> [H, W]
            tensor = tensor.squeeze(0)
            
        if denorm_params and tensor.dim() == 3 and tensor.size(0) == 3:  # RGB 이미지일 때만 정규화 해제
            try:
                tensor = denormalize_image_for_viz(tensor, mean=denorm_params['mean'], std=denorm_params['std'])
            except Exception as e:
                print(f"Warning: Could not denormalize tensor: {e}. Using original tensor.")

        # [0, 1] 범위로 클램핑 (안전 조치)
        tensor = torch.clamp(tensor, 0, 1)
        
        # PIL 이미지로 변환
        if tensor.dim() == 3 and tensor.shape[0] == 3:  # RGB 이미지
            img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(img_np)
        elif tensor.dim() == 2 or (tensor.dim() == 3 and tensor.shape[0] == 1):  # 그레이스케일
            if tensor.dim() == 3:
                tensor = tensor.squeeze(0)
            img_np = (tensor.numpy() * 255).astype(np.uint8)
            return Image.fromarray(img_np, mode='L')
        else:
            print(f"Warning: Unsupported tensor shape for PIL conversion: {tensor.shape}. Returning blank image.")
            return Image.new('RGB', (256, 256), (0, 0, 0))
    except Exception as e:
        print(f"Error in tensor_to_pil: {e}")
        # 실패하면 빈 이미지 리턴
        return Image.new('RGB', (256, 256), (0, 0, 0))

def save_pil_image(pil_img, path, logger=None):
    """PIL 이미지를 디스크에 저장합니다."""
    if pil_img is None:
        if logger:
            if hasattr(logger, 'warning'):
                logger.warning(f"Attempted to save a None PIL image to {path}")
            else:
                print(f"Warning: Attempted to save a None PIL image to {path}")
        return
    
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pil_img.save(path)
        if logger:
            if hasattr(logger, 'debug'):
                logger.debug(f"Saved image to {path}")
            else:
                print(f"Saved image to {path}")
    except Exception as e:
        if logger:
            if hasattr(logger, 'error'):
                logger.error(f"Failed to save image to {path}: {e}")
            else:
                print(f"Error: Failed to save image to {path}: {e}")

def save_tensor_image(tensor, path, denorm_params=None, logger=None, debug_name="", force_range=None):
    """완전히 수정된 이미지 저장 함수"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if tensor.device.type != 'cpu':
            tensor = tensor.detach().cpu()
        
        if tensor.dim() == 4:
            if tensor.size(0) == 1: # 배치 크기가 1이면 squeeze
                tensor = tensor.squeeze(0)
            else: # 배치 크기가 1보다 크면 첫 번째 이미지만 사용 (또는 그리드 저장 로직 분리)
                # if logger:
                #     logger.warning(f"🔍 [{debug_name}] 저장: 배치 크기 {tensor.size(0)} > 1. 첫 번째 이미지만 저장합니다.")
                tensor = tensor[0]
        
        # if logger and debug_name:
        #     logger.info(f"🔍 [{debug_name}] 저장 전:")
        #     logger.info(f"  Shape: {tensor.shape}")
        #     logger.info(f"  Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        
        # 처리 순서 변경: 1. 역정규화 (필요시) -> 2. [0,1] 클램핑
        # 이 로직은 텐서가 어떤 상태인지(예: 이미 [0,1]인지, 아니면 정규화된 상태인지)에 따라 달라짐

        processed_tensor = tensor.clone() # 원본 변경 방지

        if denorm_params and processed_tensor.dim() == 3 and processed_tensor.size(0) == 3:
            mean_val = torch.tensor(denorm_params['mean'], device=processed_tensor.device).view(3, 1, 1)
            std_val = torch.tensor(denorm_params['std'], device=processed_tensor.device).view(3, 1, 1)

            # # std 값이 너무 작은 경우 경고 (회색 이미지의 원인이 될 수 있음)
            # if logger and (std_val < 1e-4).any():
            #     logger.warning(f"⚠️ [{debug_name}] 경고: denorm_params의 std 값이 매우 작습니다! std: {denorm_params['std']}")
            
            processed_tensor = processed_tensor * std_val + mean_val
            # if logger:
            #     logger.info(f"  역정규화 적용 후 Range: [{processed_tensor.min():.3f}, {processed_tensor.max():.3f}]")
        
        # 모든 이미지 텐서는 최종적으로 [0,1] 범위로 클램핑 후 저장
        # (마스크나 단일 채널 feature map도 이 경로를 탐)
        processed_tensor = torch.clamp(processed_tensor, 0, 1)
        # if logger:
        #     logger.info(f"  최종 Clamp 후 Range: [{processed_tensor.min():.3f}, {processed_tensor.max():.3f}]")

        # 저장
        vutils.save_image(processed_tensor, path, normalize=False) # normalize=False가 중요
        
        # if logger:
        #     logger.info(f"  ✅ 저장 완료: {path}")
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"  ❌ 저장 실패 {path}: {e}\n{traceback.format_exc()}")
        return False


def save_with_pil_direct(tensor, path):
    """PIL을 사용한 직접 저장"""
    import numpy as np
    from PIL import Image
    
    # [C, H, W] → [H, W, C]
    if tensor.dim() == 3:
        if tensor.size(0) == 3:  # RGB
            tensor = tensor.permute(1, 2, 0)
            image_np = (tensor.numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
        elif tensor.size(0) == 1:  # Grayscale
            tensor = tensor.squeeze(0)
            image_np = (tensor.numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np, mode='L')
        else:
            raise ValueError(f"지원하지 않는 채널 수: {tensor.size(0)}")
    elif tensor.dim() == 2:  # Grayscale
        image_np = (tensor.numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np, mode='L')
    else:
        raise ValueError(f"지원하지 않는 텐서 차원: {tensor.dim()}")
    
    pil_image.save(path)


def save_with_matplotlib(tensor, path):
    """matplotlib을 사용한 저장"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 8))
    
    if tensor.dim() == 3:
        if tensor.size(0) == 3:  # RGB
            plt.imshow(tensor.permute(1, 2, 0).numpy())
        elif tensor.size(0) == 1:  # Grayscale
            plt.imshow(tensor.squeeze(0).numpy(), cmap='gray')
    elif tensor.dim() == 2:  # Grayscale
        plt.imshow(tensor.numpy(), cmap='gray')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()


def create_blended_images_with_multiple_thresholds(gt_img, pred_pix, conf_map, orig_mask, 
                                                    thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], 
                                                    device=None):
    """
    다양한 confidence threshold로 블렌딩된 이미지들을 생성합니다.
    *** 핵심 원칙: 검은색=마스크(예측할 부분), 흰색=보이는 영역 ***
    
    Args:
        gt_img: Ground truth 이미지 [B, 3, H, W]
        pred_pix: 예측된 픽셀 [B, 3, H, W] 
        conf_map: Confidence map [B, 1, H, W]
        orig_mask: 원본 마스크 [B, 1, H, W] (0=마스크 영역, 1=보이는 영역)
        thresholds: 테스트할 confidence threshold 리스트
        device: 텐서가 있는 장치
    
    Returns:
        dict: threshold별 결과 이미지들과 통계
    """
    results = {}
    
    if device is None:
        device = gt_img.device
    
    # Confidence 통계
    conf_stats = {
        'min': float(conf_map.min()),
        'max': float(conf_map.max()),
        'mean': float(conf_map.mean()),
        'std': float(conf_map.std()),
        'median': float(conf_map.median())
    }
    results['confidence_stats'] = conf_stats
    
    for threshold in thresholds:
        # ===== 마스크 해석: 검은색=마스크, 흰색=보이는 영역 =====
        # orig_mask: 1=보이는 영역(흰색), 0=마스크 영역(검은색)
        visible_mask = orig_mask        # 1=보이는 영역, 0=마스크 영역
        mask_region = 1.0 - orig_mask   # 1=마스크 영역, 0=보이는 영역
        
        # AFT가 채운 마스크 영역: confidence > threshold인 마스크 부분
        M_aft_filled = (conf_map > threshold).float() * mask_region
        
        # AFT가 채우지 못한 마스크 영역 (검은색으로 남을 부분)
        M_remaining = mask_region - M_aft_filled
        M_remaining = torch.clamp(M_remaining, 0, 1)
        
        # ===== 올바른 블렌딩 로직 =====
        
        # 1. Basic Blend: 보이는 영역(원본) + AFT가 채운 마스크(예측) + 나머지 마스크(검은색)
        visible_part = gt_img * visible_mask                # 보이는 영역: 원본
        aft_filled_part = pred_pix * M_aft_filled           # AFT가 채운 마스크: 예측
        remaining_black = torch.zeros_like(pred_pix) * M_remaining  # 나머지 마스크: 검은색
        basic_blend = visible_part + aft_filled_part + remaining_black
        
        # 2. Full Blend: 보이는 영역(원본) + 전체 마스크 영역(예측)
        # visible_part = gt_img * visible_mask              # 보이는 영역: 원본 (already defined)
        full_mask_prediction = pred_pix * mask_region     # 전체 마스크 영역: 예측
        full_blend = visible_part + full_mask_prediction
        
        # 3. Soft Blend: 보이는 영역(원본) + 마스크 영역(Confidence 가중 블렌딩)
        # visible_part = gt_img * visible_mask              # 보이는 영역: 원본 (already defined)
        # 마스크 영역에서만 confidence 적용: 검은색과 예측 사이의 가중 평균
        confidence_weight = conf_map * mask_region
        black_background = torch.zeros_like(pred_pix)
        mask_soft_blend = black_background * (1.0 - confidence_weight) + pred_pix * confidence_weight
        soft_blend = visible_part + mask_soft_blend
        
        # 채워진 비율 계산
        aft_filled_ratio = float(M_aft_filled.sum() / mask_region.sum()) if mask_region.sum() > 0 else 0.0
        
        results[f'thresh_{threshold}'] = {
            'M_aft_filled': M_aft_filled,    # AFT가 채운 마스크 영역 (흰색=채움)
            'M_remaining': M_remaining,      # AFT가 못 채운 마스크 영역 (흰색=못 채움)
            'basic_blend': basic_blend,
            'full_blend': full_blend,
            'soft_blend': soft_blend,
            'aft_filled_ratio': aft_filled_ratio
        }
    
    return results


def normalize_to_01(tensor):
    """텐서를 [0,1] 범위로 정규화"""
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val > min_val:
        return (tensor - min_val) / (max_val - min_val)
    else:
        return torch.clamp(tensor, 0, 1)


def visualize_and_save_batch_aft(batch_data, model_outputs, epoch, output_dir, config, logger, num_samples_to_show=1, denorm_params=None):
    """
    RegionBasedSelector를 사용하는 AFT-Net의 입력과 출력을 시각화하고 저장합니다.
    (수정된 버전)
    """
    if logger:
        logger.info(f"--- Visualizing batch for epoch {epoch} ---")

    try:
        # --- 1. 입력 데이터 유효성 검사 및 준비 ---
        if not batch_data or not isinstance(batch_data, dict):
            if logger: logger.error("Invalid batch_data (None or not dict)")
            return

        partial_img = batch_data.get('partial_image')
        orig_mask = batch_data.get('original_mask') # Config에 따라 0=채울 영역, 1=보이는 영역 (또는 반대)
        gt_img = batch_data.get('ground_truth_image')
        candidate_images = batch_data.get('candidate_images') # 선택적: 후보 이미지 로깅용

        if partial_img is None: raise ValueError("visualize_and_save_batch_aft: partial_image is None")
        if orig_mask is None: raise ValueError("visualize_and_save_batch_aft: original_mask is None")
        if gt_img is None: raise ValueError("visualize_and_save_batch_aft: ground_truth_image is None")

        # --- 2. 모델 출력 유효성 검사 및 언패킹 ---
        if not model_outputs or not isinstance(model_outputs, tuple):
            if logger: logger.error("Invalid model_outputs (None or not tuple)")
            return

        if len(model_outputs) == 3:
            final_completed_image, confidence_map_direct, selection_info = model_outputs
            if logger: logger.info("Model outputs unpacked: final_completed_image, confidence_map_direct, selection_info")
        elif len(model_outputs) == 2: # 이전 모델 또는 (final_image, conf_map) 형식
            logger.warning(f"Model_outputs has {len(model_outputs)} elements. Assuming (final_image, conf_map). selection_info will be None.")
            final_completed_image, confidence_map_direct = model_outputs
            selection_info = None
        else:
            if logger: logger.error(f"Unexpected number of model_outputs: {len(model_outputs)}. Expected 2 or 3.")
            return

        if final_completed_image is None: raise ValueError("visualize_and_save_batch_aft: final_completed_image from model_outputs is None")
        if confidence_map_direct is None: raise ValueError("visualize_and_save_batch_aft: confidence_map_direct from model_outputs is None")
        
        if denorm_params:
            if logger: logger.info(f"Using denorm_params: mean={denorm_params.get('mean')}, std={denorm_params.get('std')}")
        else:
            if logger: logger.warning("denorm_params is None. Images requiring denormalization might not display correctly.")


        # --- 3. 저장 간격 및 경로 설정 ---
        save_interval = getattr(config.train, 'visualize_interval_epoch', 1)
        if epoch % save_interval != 0:
            if logger: logger.debug(f"Skipping image saving for epoch {epoch} (save_interval: {save_interval})")
            return

        results_epoch_dir = os.path.join(output_dir, f"results_epoch_{epoch}")
        os.makedirs(results_epoch_dir, exist_ok=True)


        # --- 4. 각 샘플에 대한 이미지 저장 ---
        batch_size = partial_img.size(0)
        for i in range(min(batch_size, num_samples_to_show)):
            
            sample_path_from_batch = batch_data.get('path')
            current_sample_orig_path = None
            if isinstance(sample_path_from_batch, list) and i < len(sample_path_from_batch):
                current_sample_orig_path = sample_path_from_batch[i]
            elif isinstance(sample_path_from_batch, str): 
                current_sample_orig_path = sample_path_from_batch 
            
            sample_filename_no_ext = ""
            current_sample_sub_dir = ""

            if current_sample_orig_path and isinstance(current_sample_orig_path, str):
                train_root_dir = os.path.expanduser(getattr(config.data, 'train_root', ''))
                if current_sample_orig_path.startswith(train_root_dir) and train_root_dir:
                    rel_path = os.path.relpath(current_sample_orig_path, train_root_dir)
                else:
                    rel_path = os.path.basename(current_sample_orig_path)
                
                current_sample_sub_dir = os.path.dirname(rel_path)
                sample_filename_no_ext = os.path.splitext(os.path.basename(rel_path))[0]
            else: 
                current_sample_sub_dir = f"sample_{i}"
                sample_filename_no_ext = f"idx{i}"

            sample_specific_output_dir = os.path.join(results_epoch_dir, current_sample_sub_dir)
            os.makedirs(sample_specific_output_dir, exist_ok=True)
            result_prefix = os.path.join(sample_specific_output_dir, sample_filename_no_ext)

            if logger:
                logger.info(f"--- Saving images for sample {i} (epoch {epoch}, id: {sample_filename_no_ext}) ---")
                logger.info(f"  Output prefix: {result_prefix}")

            save_tensor_image(gt_img[i:i+1], f"{result_prefix}_0_ground_truth.png", 
                                denorm_params, logger, debug_name=f"GT_{sample_filename_no_ext}")
            save_tensor_image(partial_img[i:i+1], f"{result_prefix}_1_partial_image.png", 
                                denorm_params, logger, debug_name=f"Partial_{sample_filename_no_ext}")
            save_tensor_image(orig_mask[i:i+1], f"{result_prefix}_2_original_mask.png", 
                                None, logger, debug_name=f"OrigMask_{sample_filename_no_ext}")
            save_tensor_image(final_completed_image[i:i+1], f"{result_prefix}_3_COMPLETED_final.png", 
                                denorm_params, logger, debug_name=f"NetOutput_{sample_filename_no_ext}")
            save_tensor_image(confidence_map_direct[i:i+1], f"{result_prefix}_4_confidence_map.png", 
                                None, logger, debug_name=f"Confidence_{sample_filename_no_ext}")
            
            if candidate_images is not None and candidate_images.ndim == 5 and i < candidate_images.size(0):
                num_candidates_to_save = min(candidate_images.size(1), 3) 
                for k_idx in range(num_candidates_to_save):
                    save_tensor_image(candidate_images[i:i+1, k_idx], 
                                        f"{result_prefix}_candidate_{k_idx}.png", 
                                        denorm_params, logger, debug_name=f"Candidate{k_idx}_{sample_filename_no_ext}")

            if selection_info and isinstance(selection_info, dict):
                sel_weights = selection_info.get('selection_weights') 
                if sel_weights is not None and sel_weights.ndim == 4 and i < sel_weights.size(0):
                    num_candidates = sel_weights.size(1)
                    for k_idx in range(num_candidates):
                        save_tensor_image(sel_weights[i:i+1, k_idx:k_idx+1], 
                                            f"{result_prefix}_selection_weight_k{k_idx}.png",
                                            None, logger, debug_name=f"SelWeight_K{k_idx}_{sample_filename_no_ext}")
                
                final_sel_scores = selection_info.get('final_selection_scores')
                if final_sel_scores is not None and final_sel_scores.ndim == 4 and i < final_sel_scores.size(0):
                    for k_idx in range(final_sel_scores.size(1)):
                        score_map_k = normalize_to_01(final_sel_scores[i:i+1, k_idx:k_idx+1])
                        save_tensor_image(score_map_k,
                                            f"{result_prefix}_final_score_k{k_idx}.png",
                                            None, logger, debug_name=f"FinalScore_K{k_idx}_{sample_filename_no_ext}")

        # --- 5. 그리드 시각화 (주석 처리된 이전 버전은 여기에 있었음) ---
        # grid_dir = os.path.join(output_dir, "visualizations_grid_epoch_" + str(epoch))
        # ... (이전 주석 처리된 그리드 코드)

    except ValueError as ve:
        if logger: logger.error(f"Error in visualize_and_save_batch_aft (ValueError): {ve}\n{traceback.format_exc()}")
    except Exception as e:
        error_msg = f"Unexpected error in visualize_and_save_batch_aft main section for epoch {epoch}: {e}"
        if logger:
            logger.error(error_msg)
            logger.error(traceback.format_exc())
        # 오류 정보 파일 저장 (선택적)
        # ...
        return # 중요: 주 처리 로직에서 에러 발생 시 그리드 시각화로 넘어가지 않도록 리턴

    # ===== 그리드 시각화 (메인 try-except 블록 이후 실행) =====
    # 이 블록은 위의 try-except가 성공적으로 완료된 후에만 실행되거나, 
    # 에러 발생 시 visualize_and_save_batch_aft 함수가 중간에 return하므로 실행되지 않음.
    viz_dir = os.path.join(output_dir, "visualizations") # 에포크별 디렉토리가 아닌 공통 visualizations 디렉토리
    os.makedirs(viz_dir, exist_ok=True)
    
    try:
        grid_size = min(partial_img.size(0), 4) # partial_img 등은 함수 초반에 정의됨
        
        # 그리드 이미지 생성 및 저장 (GPU에서 직접 처리)
        # 변수명 수정: pred_pix -> final_completed_image, conf_map -> confidence_map_direct, final_display_img -> final_completed_image
        partial_grid = vutils.make_grid(partial_img[:grid_size], nrow=2, normalize=False, scale_each=False)
        mask_grid = vutils.make_grid(orig_mask[:grid_size], nrow=2, normalize=False, scale_each=False)
        gt_grid = vutils.make_grid(gt_img[:grid_size], nrow=2, normalize=False, scale_each=False)
        
        # final_completed_image와 confidence_map_direct는 model_outputs에서 가져옴
        pred_grid = vutils.make_grid(final_completed_image[:grid_size], nrow=2, normalize=False, scale_each=False)
        conf_grid = vutils.make_grid(confidence_map_direct[:grid_size], nrow=2, normalize=False, scale_each=False)
        final_grid = vutils.make_grid(final_completed_image[:grid_size], nrow=2, normalize=False, scale_each=False) # final_display_img 대신 final_completed_image 사용
        
        # 그리드 이미지 저장
        # denorm_params는 RGB 이미지(partial, gt, pred, final)에만 적용
        save_tensor_image(partial_grid, os.path.join(viz_dir, f"epoch_{epoch}_partial_grid.png"), denorm_params, logger, debug_name=f"GridPartial_Ep{epoch}")
        save_tensor_image(mask_grid, os.path.join(viz_dir, f"epoch_{epoch}_mask_grid.png"), None, logger, debug_name=f"GridMask_Ep{epoch}")
        save_tensor_image(gt_grid, os.path.join(viz_dir, f"epoch_{epoch}_gt_grid.png"), denorm_params, logger, debug_name=f"GridGT_Ep{epoch}")
        save_tensor_image(pred_grid, os.path.join(viz_dir, f"epoch_{epoch}_pred_grid.png"), denorm_params, logger, debug_name=f"GridPred_Ep{epoch}") # pred_pix 대신 final_completed_image 사용
        save_tensor_image(conf_grid, os.path.join(viz_dir, f"epoch_{epoch}_conf_grid.png"), None, logger, debug_name=f"GridConf_Ep{epoch}") # conf_map 대신 confidence_map_direct 사용
        save_tensor_image(final_grid, os.path.join(viz_dir, f"epoch_{epoch}_final_grid.png"), denorm_params, logger, debug_name=f"GridFinal_Ep{epoch}") # final_display_img 대신 final_completed_image 사용
        
        comparison_grid = vutils.make_grid(
            torch.cat([partial_img[:grid_size], final_completed_image[:grid_size]], dim=0), # final_display_img 대신 final_completed_image
            nrow=grid_size, normalize=False, scale_each=False
        )
        save_tensor_image(comparison_grid, os.path.join(viz_dir, f"epoch_{epoch}_comparison.png"), denorm_params, logger, debug_name=f"GridCompare_Ep{epoch}")
        
        if logger:
            if hasattr(logger, 'info'):
                logger.info(f"Saved visualization grids for epoch {epoch} to {viz_dir}")
            else:
                print(f"Saved visualization grids for epoch {epoch} to {viz_dir}")
    except Exception as grid_err:
        if logger:
            if hasattr(logger, 'error'):
                logger.error(f"Error creating visualization grids for epoch {epoch}: {grid_err}\n{traceback.format_exc()}")
            else:
                print(f"Error creating visualization grids for epoch {epoch}: {grid_err}")
    
    # 파이썬 오브젝트만 삭제 (GPU 메모리는 그대로 유지)
    # try:
    #     del partial_grid, mask_grid, gt_grid, pred_grid, conf_grid, final_grid, comparison_grid
    # except NameError: # 변수가 정의되지 않았을 수 있음 (예: grid_size=0)
    #     pass
    # except Exception:
    #     pass # 다른 예외 발생 시 무시

    # visualize_and_save_batch_aft 함수의 마지막 에러 핸들링은 이미 메인 try-except에 있음.
    # 이 구조에서는 그리드 시각화는 별도의 try-except로 처리됨.


