# inference_aft.py (수정된 버전 - 분리된 PatchEmbed 지원)

import os
import yaml
import glob
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import argparse
import json
from tqdm import tqdm

from models.csf_network import CSFNetwork
from utils.logger import setup_logger

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

def load_model(config, checkpoint_path, device, logger):
    """🔧 수정: 분리된 PatchEmbed를 고려한 모델 로딩"""
    logger.info(f"Initializing AFT-Net model for inference...")
    
    # 🎯 핵심: 수정된 모델 구조 사용
    model = CSFNetwork(config).to(device)
    
    if not os.path.isfile(checkpoint_path):
        logger.error(f"Checkpoint file not found at {checkpoint_path}. Exiting.")
        return None

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
    # DDP 접두사 제거
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("✅ Model checkpoint loaded successfully with separated PatchEmbed support.")
    except RuntimeError as e:
        logger.error(f"❌ Error loading state_dict: {e}")
        logger.info("🔧 Trying with strict=False...")
        try:
            model.load_state_dict(new_state_dict, strict=False)
            logger.warning("⚠️ Model loaded with strict=False. Some weights may be missing.")
        except Exception as e2:
            logger.error(f"❌ Complete failure loading model: {e2}")
            return None
        
    model.eval()
    return model

def preprocess_inputs(original_pil, mask_pil, candidate_pils, config, device):
    """🔧 수정: 정규화 범위를 모델에 맞게 조정"""
    img_size = config.data.img_size
    transform_params = config.data.transform_params.__dict__ if hasattr(config.data.transform_params, '__dict__') else config.data.transform_params
    
    # 기본 전처리
    to_tensor = transforms.ToTensor()
    normalize_mean = transform_params.get('mean', [0.485, 0.456, 0.406])
    normalize_std = transform_params.get('std', [0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)
    
    # 1. 원본 이미지 전처리
    original_resized = original_pil.resize((img_size, img_size), Image.BILINEAR)
    original_tensor = to_tensor(original_resized)
    original_tensor_norm = normalize(original_tensor)
    
    # 2. 마스크 전처리 (바이너리 마스크)
    mask_resized = mask_pil.resize((img_size, img_size), Image.NEAREST)
    mask_tensor = to_tensor(mask_resized)
    mask_binary = (mask_tensor > 0.5).float()  # [1, H, W], 1=visible, 0=fill
    
    # 3. 🎯 핵심 수정: Partial 이미지 생성
    # visible 영역은 원본, fill 영역은 0으로 설정
    partial_tensor_norm = original_tensor_norm * mask_binary
    
    # 4. 후보 이미지들 전처리
    candidate_tensors = []
    for cand_pil in candidate_pils:
        cand_resized = cand_pil.resize((img_size, img_size), Image.BILINEAR)
        cand_tensor = to_tensor(cand_resized)
        cand_tensor_norm = normalize(cand_tensor)
        candidate_tensors.append(cand_tensor_norm)
    
    if candidate_tensors:
        candidate_stack = torch.stack(candidate_tensors)  # [K, 3, H, W]
    else:
        # 빈 후보 텐서
        candidate_stack = torch.empty((0, 3, img_size, img_size), dtype=original_tensor_norm.dtype)
    
    # 디바이스로 이동
    partial_tensor_norm = partial_tensor_norm.to(device)
    mask_binary = mask_binary.to(device)
    candidate_stack = candidate_stack.to(device)
    
    return partial_tensor_norm, mask_binary, candidate_stack, normalize_mean, normalize_std

def denormalize_image(tensor, mean, std, clip_0_1=True):
    """정규화 해제"""
    if tensor is None: 
        return None
    
    is_batch = tensor.dim() == 4
    if not is_batch:
        tensor = tensor.unsqueeze(0)
    
    mean_t = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    denormalized_tensor = tensor * std_t + mean_t
    
    if clip_0_1:
        denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)
    
    return denormalized_tensor.squeeze(0) if not is_batch else denormalized_tensor

def save_image_tensor(tensor, path, denorm_params=None, is_mask_or_single_channel=False):
    """텐서를 이미지로 저장"""
    if tensor is None:
        print(f"Warning: Tensor to save at {path} is None.")
        return
    
    tensor_to_save = tensor.cpu().detach()
    
    # 배치 차원 제거
    if tensor_to_save.dim() == 4 and tensor_to_save.size(0) == 1:
        tensor_to_save = tensor_to_save.squeeze(0)
    
    # 정규화 해제 (마스크가 아닌 경우)
    if denorm_params and not is_mask_or_single_channel:
        tensor_to_save = denormalize_image(tensor_to_save, denorm_params['mean'], denorm_params['std'], clip_0_1=True)
    elif is_mask_or_single_channel and tensor_to_save.dim() == 3 and tensor_to_save.size(0) == 1:
        tensor_to_save = tensor_to_save.squeeze(0)
    elif is_mask_or_single_channel and tensor_to_save.dim() != 2:
        print(f"Warning: Mask/Single-channel tensor for {path} has unexpected shape {tensor_to_save.shape}.")
    
    try:
        pil_img = TF.to_pil_image(tensor_to_save)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pil_img.save(path)
    except Exception as e:
        print(f"Error saving tensor to {path}: {e}")
        print(f"Tensor shape: {tensor_to_save.shape}, dtype: {tensor_to_save.dtype}")

def run_inference(model, partial_tensor, mask_tensor, candidate_tensor, candidate_scores_tensor, device, logger):
    """🔧 수정: Region-based Selection 출력 처리"""
    with torch.no_grad():
        # 배치 차원 추가
        if partial_tensor.dim() == 3:
            partial_tensor = partial_tensor.unsqueeze(0)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        if candidate_tensor.dim() == 3:
            candidate_tensor = candidate_tensor.unsqueeze(0)
        elif candidate_tensor.dim() == 4 and candidate_tensor.size(0) > 0:
            candidate_tensor = candidate_tensor.unsqueeze(0)  # [1, K, 3, H, W]
        
        # 🎯 핵심: 수정된 모델 호출
        try:
            # 🔧 모델에 candidate_scores 정보 전달
            if hasattr(model, 'module'):  # DDP인 경우
                model.module._current_candidate_scores = candidate_scores_tensor
            else:
                model._current_candidate_scores = candidate_scores_tensor
            
            # Region-based Selection 모델의 출력: (image, confidence, selection_info)
            predicted_image, confidence_map, selection_info = model(
                partial_tensor, mask_tensor, candidate_tensor
            )
            
            logger.info(f"✅ Model inference successful with score information")
            
            logger.info(f"✅ Model inference successful")
            logger.debug(f"Output shapes: image={predicted_image.shape}, confidence={confidence_map.shape}")
            
            # 🎯 Region-based Selection 출력 처리
            if hasattr(model, 'fusion_method') and model.fusion_method == "region_based":
                # Region-based Selection: 이미  정규화된 범위의 이미지
                logger.info("🎨 Processing Region-based Selection output")
                final_image = predicted_image.squeeze(0)
                final_confidence = confidence_map.squeeze(0)
                
                # selection_info 로깅
                if isinstance(selection_info, dict):
                    debug_info = selection_info.get('debug_info', {})
                    logger.info(f"Region selection info: {debug_info}")
            else:
                # 다른 융합 방식: 로짓 출력 → 시그모이드 적용
                logger.info("🎨 Processing standard fusion output (applying sigmoid)")
                final_image = torch.sigmoid(predicted_image.squeeze(0))
                final_confidence = torch.sigmoid(confidence_map.squeeze(0))
            
            return final_image, final_confidence, selection_info
            
        except Exception as e:
            logger.error(f"❌ Model inference failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, None, None

def process_single_sample(sample_id, candidate_json_path, actual_data_root, config, model, device, output_dir, logger, save_debug=False):
    """단일 샘플 처리"""
    logger.info(f"🔄 Processing sample: {sample_id}")
    
    # 1. 실제 이미지 경로 설정
    current_sample_dir = os.path.join(actual_data_root, sample_id)
    original_img_path = os.path.join(current_sample_dir, "original.png")
    mask_img_path = os.path.join(current_sample_dir, "mask.png")
    
    # 파일 존재 확인
    if not os.path.exists(original_img_path) or not os.path.exists(mask_img_path):
        logger.error(f"❌ Required files missing for {sample_id}")
        return False
    
    # 2. 이미지 로딩
    try:
        original_pil = Image.open(original_img_path).convert("RGB")
        mask_pil = Image.open(mask_img_path).convert("L")
        logger.info(f"✅ Images loaded for {sample_id}")
    except Exception as e:
        logger.error(f"❌ Error loading images for {sample_id}: {e}")
        return False
    
    # 3. 후보 이미지 로딩 (JSON 기반)
    candidate_pils = []
    candidate_scores = []  # 새로 추가
    try:
        with open(candidate_json_path, 'r') as f:
            candidate_data = json.load(f)
        
        top_k_infos = candidate_data.get('candidates', [])[:config.data.Kmax]
        
        for cand_info in top_k_infos:
            img_rel_path = cand_info.get('image_path_relative')
            score = cand_info.get('score', 0.5)  # 새로 추가
            candidate_scores.append(score)  # Score 정보 저장
            
            if img_rel_path:
                img_abs_path = os.path.join(current_sample_dir, img_rel_path)
                if os.path.exists(img_abs_path):
                    cand_img = Image.open(img_abs_path).convert("RGB")
                    candidate_pils.append(cand_img)
                else:
                    logger.warning(f"⚠️ Candidate image not found: {img_abs_path}")
                    # 더미 이미지 추가
                    candidate_pils.append(Image.new('RGB', (256, 256), (128, 128, 128)))
        
        logger.info(f"✅ Loaded {len(candidate_pils)} candidates for {sample_id}")
        
    except Exception as e:
        logger.error(f"❌ Error loading candidates for {sample_id}: {e}")
        candidate_pils = []
    
    # Kmax까지 더미 패딩
    # Kmax까지 더미 패딩 (Score도 함께)
    while len(candidate_pils) < config.data.Kmax:
        candidate_pils.append(Image.new('RGB', (256, 256), (128, 128, 128)))
        candidate_scores.append(0.1)  # 더미 후보는 낮은 점수

    # Score를 텐서로 변환
    candidate_scores_tensor = torch.tensor(candidate_scores[:config.data.Kmax], dtype=torch.float32).unsqueeze(0).to(device)  # [1, K]
    
    # 4. 전처리
    partial_tensor, mask_tensor, candidate_tensor, norm_mean, norm_std = preprocess_inputs(
        original_pil, mask_pil, candidate_pils, config, device
    )
    
    # 5. 추론 실행
    predicted_image, confidence_map, selection_info = run_inference(
        model, partial_tensor, mask_tensor, candidate_tensor, candidate_scores_tensor, device, logger
    )
    
    if predicted_image is None:
        logger.error(f"❌ Inference failed for {sample_id}")
        return False
    
    # 6. 결과 저장
    sample_output_dir = os.path.join(output_dir, "results", sample_id)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    denorm_params = {'mean': norm_mean, 'std': norm_std}
    
    # 입력 이미지들 저장
    save_image_tensor(transforms.ToTensor()(original_pil), 
                     os.path.join(sample_output_dir, f"{sample_id}_input_original.png"))
    save_image_tensor(mask_tensor, 
                     os.path.join(sample_output_dir, f"{sample_id}_input_mask.png"), 
                     is_mask_or_single_channel=True)
    save_image_tensor(partial_tensor, 
                     os.path.join(sample_output_dir, f"{sample_id}_input_partial.png"), 
                     denorm_params=denorm_params)
    
    # 🎯 핵심: 완성된 이미지 저장
    save_image_tensor(predicted_image, 
                     os.path.join(sample_output_dir, f"{sample_id}_completed_AFT.png"), 
                     denorm_params=denorm_params)
    
    # 신뢰도 맵 저장
    if confidence_map is not None:
        save_image_tensor(confidence_map, 
                         os.path.join(sample_output_dir, f"{sample_id}_confidence_map.png"), 
                         is_mask_or_single_channel=True)
    
    # 디버그 정보 저장
    # 기존 디버그 정보 저장 부분
    if save_debug and selection_info and isinstance(selection_info, dict):
        if 'selection_weights' in selection_info and selection_info['selection_weights'] is not None:
            sw_tensor = selection_info['selection_weights'].cpu()
            if sw_tensor.dim() == 4 and sw_tensor.size(0) == 1:
                sw_tensor = sw_tensor.squeeze(0)
            
            if sw_tensor.dim() == 3:
                for k_idx in range(min(sw_tensor.size(0), config.data.Kmax)):
                    save_image_tensor(sw_tensor[k_idx].unsqueeze(0),
                                    os.path.join(sample_output_dir, f"{sample_id}_debug_selection_weight_cand{k_idx}.png"),
                                    is_mask_or_single_channel=True)

    # 🔧 Score 정보 저장 (새로 추가) - 여기에 추가!
    if save_debug:
        score_info = {
            'sample_id': sample_id,
            'candidate_scores': candidate_scores[:config.data.Kmax],
            'score_statistics': {
                'min_score': min(candidate_scores[:config.data.Kmax]),
                'max_score': max(candidate_scores[:config.data.Kmax]),
                'mean_score': sum(candidate_scores[:config.data.Kmax]) / len(candidate_scores[:config.data.Kmax]),
            }
        }
        
        with open(os.path.join(sample_output_dir, f"{sample_id}_score_info.json"), 'w') as f:
            json.dump(score_info, f, indent=2)

    logger.info(f"✅ Sample {sample_id} processed successfully")
    return True

def main_inference_loop(args, config, model, device, logger):
    """메인 추론 루프"""
    json_files_dir = os.path.expanduser(args.input_dir)
    actual_data_root = os.path.expanduser(args.actual_data_root)
    
    # 디렉토리 확인
    if not os.path.isdir(json_files_dir):
        logger.error(f"❌ JSON input directory not found: {json_files_dir}")
        return
    if not os.path.isdir(actual_data_root):
        logger.error(f"❌ Actual data root directory not found: {actual_data_root}")
        return
    
    # JSON 파일 찾기
    json_files = sorted(glob.glob(os.path.join(json_files_dir, "*_candidates.json")))
    if not json_files:
        logger.error(f"❌ No '*_candidates.json' files found in {json_files_dir}")
        return
    
    logger.info(f"📁 Found {len(json_files)} JSON files to process")
    
    # 출력 디렉토리 설정
    base_output_name = f"{config.exp_name}_{os.path.splitext(os.path.basename(args.checkpoint_path))[0]}"
    output_dir = os.path.join(args.output_base_dir, base_output_name)
    
    # 처리 시작
    success_count = 0
    for candidate_json_path in tqdm(json_files, desc="Processing Samples"):
        sample_id = os.path.basename(candidate_json_path).replace("_candidates.json", "")
        
        success = process_single_sample(
            sample_id, candidate_json_path, actual_data_root, config, 
            model, device, output_dir, logger, args.save_debug_info
        )
        
        if success:
            success_count += 1
    
    logger.info(f"🎉 Processing completed: {success_count}/{len(json_files)} samples successful")

def main(args):
    """메인 함수"""
    config = load_config_from_yaml(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # 출력 디렉토리 및 로거 설정
    base_output_name = f"{config.exp_name}_{os.path.splitext(os.path.basename(args.checkpoint_path))[0]}"
    log_dir = os.path.join(args.output_base_dir, base_output_name, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    logger = setup_logger(f"AFT_Inference_{base_output_name}", 
                         os.path.join(log_dir, "inference.log"))
    
    logger.info("🚀 AFT-Net Inference Started")
    logger.info(f"📝 Config: {args.config_path}")
    logger.info(f"🏗️ Checkpoint: {args.checkpoint_path}")
    logger.info(f"📁 Input JSON dir: {args.input_dir}")
    logger.info(f"📁 Actual data root: {args.actual_data_root}")
    logger.info(f"💾 Output base dir: {args.output_base_dir}")
    logger.info(f"🖥️ Device: {device}")
    
    # 모델 로딩
    model = load_model(config, args.checkpoint_path, device, logger)
    if model is None:
        logger.error("❌ Model loading failed. Exiting.")
        return
    
    # 추론 실행
    main_inference_loop(args, config, model, device, logger)
    
    logger.info("🎉 AFT-Net Inference Completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AFT-Net Inference Script (Updated for separated PatchEmbed)")
    parser.add_argument('--config_path', type=str, required=True, help='Path to the YAML configuration file.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the *_candidates.json files.')
    parser.add_argument('--actual_data_root', type=str, required=True, help='Root directory where actual image subdirectories are located.')
    parser.add_argument('--output_base_dir', type=str, default="./inference_results_aft", help='Base directory to save all inference results.')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'], help='Dataset split.')
    parser.add_argument('--cpu', action='store_true', help='Force use CPU for inference.')
    parser.add_argument('--save_debug_info', action='store_true', help='Save debug information like selection_weights.')
    
    args = parser.parse_args()
    main(args)