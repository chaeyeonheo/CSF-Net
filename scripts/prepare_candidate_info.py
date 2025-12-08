# scripts/prepare_candidate_info.py


'''
기본 동작 (--overwrite 없음):
python scripts/prepare_candidate_info.py --config_path ./configs/aft_config.yaml --split train --use_gpu

이미 존재하는 JSON 파일은 건드리지 않고 스킵
새로운 샘플만 처리
중단된 작업을 재개할 때 유용


--overwrite 사용:
python scripts/prepare_candidate_info.py --config_path ./configs/aft_config.yaml --split train --use_gpu --overwrite

이미 존재하는 JSON 파일도 새로 계산해서 덮어쓰기
모든 샘플을 처음부터 다시 처리
설정이 바뀌었거나 완전히 새로 시작할 때 사용

'''

import os
import sys
import argparse
import yaml
import json
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import torch
import lpips # pip install lpips
import glob 
import numpy as np

# 프로젝트 루트를 sys.path에 추가 (모듈 임포트 위함)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# candidate_processing 모듈에서 필요한 함수들 임포트
from data_utils.candidate_processing import (
    filter_candidate_basic, 
    calculate_lpips_consistency_score, 
    calculate_combined_consistency_score,
    calculate_cross_attention_relevance_score  # 호환성을 위해 유지
)

# 더미 모델 클래스들 (실제로는 사용하지 않지만 호환성을 위해)
class SimpleFeatureExtractor:
    def __init__(self, in_channels=3, output_dim=128, img_size=64):
        pass
    def to(self, device):
        return self
    def eval(self):
        return self

class CrossAttentionScorer:
    def __init__(self, feature_dim=128, num_heads=4):
        pass
    def to(self, device):
        return self
    def eval(self):
        return self

from torchvision.transforms import functional as TF

# config 로드 헬퍼
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

def _load_pil_image(path, target_size=None, mode='RGB', is_mask=False, logger=print):
    try:
        img = Image.open(path)
        if is_mask: 
            img = img.convert('L') # 마스크는 그레이스케일
        else: 
            img = img.convert(mode)     # 일반 이미지는 RGB
        
        # target_size가 주어지고, 현재 이미지 크기와 다를 경우에만 리사이즈
        if target_size and img.size != target_size:
            interp_mode = TF.InterpolationMode.NEAREST if is_mask else TF.InterpolationMode.BILINEAR
            # TF.resize는 [H, W] 순서의 크기를 받음
            img = TF.resize(img, [target_size[1], target_size[0]], interpolation=interp_mode)
        return img
    except FileNotFoundError: 
        logger(f"DEBUG: Image file not found at {path}")
    except UnidentifiedImageError: 
        logger(f"DEBUG: Cannot identify image file {path}")
    except Exception as e: 
        logger(f"DEBUG: Error loading image {path}: {e}")
    return None

def process_single_sample_offline(sample_dir, config, lpips_model, feat_extractor, attn_scorer, device_for_scoring, logger, pbar_instance=None):
    """
    기존 함수명 유지, MSE+LPIPS 결합 스코어링으로 수정
    cross_attn 제거하고 lpips와 mse만 사용
    """
    sample_id = os.path.basename(sample_dir)
    
    scorer_params = getattr(config.data, 'candidate_scorer_params', ConfigNamespace({}))
    filter_params = getattr(config.data, 'candidate_filter_params', ConfigNamespace({}))

    # MSE와 LPIPS 가중치 설정 (config에서 읽어오거나 기본값 사용)
    mse_weight = getattr(scorer_params, 'score_weight_mse', 0.5)
    lpips_weight = getattr(scorer_params, 'score_weight_lpips', 0.5)
    use_mse = getattr(scorer_params, 'use_mse', True)  # MSE 사용 여부

    # 기존 이미지 로딩 로직 유지
    img_size_lpips_w, img_size_lpips_h = getattr(scorer_params, 'lpips_img_size', config.data.img_size), getattr(scorer_params, 'lpips_img_size', config.data.img_size)
    
    original_pil_lpips = _load_pil_image(os.path.join(sample_dir, "original.png"), target_size=(img_size_lpips_w, img_size_lpips_h), logger=logger)
    original_mask_pil_lpips = _load_pil_image(os.path.join(sample_dir, "mask.png"), target_size=(img_size_lpips_w, img_size_lpips_h), is_mask=True, logger=logger)
    
    if not all([original_pil_lpips, original_mask_pil_lpips]):
        logger(f"INFO: Skipping sample {sample_id} due to missing base images for scoring.")
        return None

    all_comp_files = sorted(glob.glob(os.path.join(sample_dir, 'comp[0-9]*.png')))
    if not all_comp_files:
        return {'sample_id': sample_id, 'candidates': []} 

    scored_candidates_info = []
    num_all_comps = len(all_comp_files)
    num_passed_initial_filter = 0
    num_actually_scored = 0

    for comp_idx, comp_path in enumerate(all_comp_files):
        if pbar_instance:
            pbar_instance.set_description_str(f"Sample {sample_id} (Cand {comp_idx+1}/{num_all_comps}, Scored {num_actually_scored})")
        
        # 기존 이미지 로딩 로직 유지
        comp_pil_for_lpips = _load_pil_image(comp_path, target_size=(img_size_lpips_w, img_size_lpips_h), logger=logger)
        
        comp_mask_filename = os.path.basename(comp_path).replace("comp", "comp_mask_")
        comp_mask_path = os.path.join(sample_dir, comp_mask_filename)
        
        comp_amodal_mask_pil_for_filtering = None
        actual_mask_file_used = False

        # 기존 마스크 로딩 로직 유지
        if os.path.exists(comp_mask_path):
            comp_amodal_mask_pil_for_filtering = _load_pil_image(comp_mask_path, 
                                                                 target_size=(img_size_lpips_w, img_size_lpips_h), 
                                                                 is_mask=True, logger=logger)
            if comp_amodal_mask_pil_for_filtering is not None:
                actual_mask_file_used = True
        
        if not actual_mask_file_used:
            if comp_pil_for_lpips is not None:
                try:
                    img_arr = np.array(comp_pil_for_lpips.convert('RGB'))
                    if img_arr.ndim == 3 and img_arr.shape[2] == 3:
                         bg_pixels = np.all(img_arr >= 254, axis=-1)
                         mask_arr = np.where(bg_pixels, 255, 0).astype(np.uint8)
                         comp_amodal_mask_pil_for_filtering = Image.fromarray(mask_arr, mode='L')
                    else:
                         comp_amodal_mask_pil_for_filtering = Image.new('L', (img_size_lpips_w, img_size_lpips_h), 255)
                except Exception as e_derive_mask:
                    logger(f"DEBUG: Error deriving mask from {comp_path}: {e_derive_mask}. Using blank mask.")
                    comp_amodal_mask_pil_for_filtering = Image.new('L', (img_size_lpips_w, img_size_lpips_h), 255)
            else: 
                comp_amodal_mask_pil_for_filtering = Image.new('L', (img_size_lpips_w, img_size_lpips_h), 255)

        if comp_pil_for_lpips is None or comp_amodal_mask_pil_for_filtering is None:
            continue

        # 기존 필터링 로직 유지 (기존 함수 사용)
        is_valid_initial = filter_candidate_basic(
            comp_pil_for_lpips, comp_amodal_mask_pil_for_filtering, original_mask_pil_lpips,
            min_overlap_ratio=getattr(filter_params, 'min_overlap_ratio', 0.15),
            min_amodal_area_ratio=getattr(filter_params, 'min_amodal_area_ratio', 0.01),
            img_H_W=(img_size_lpips_h, img_size_lpips_w),
            idx=comp_idx, sample_id=sample_id
        )
        
        if not is_valid_initial:
            continue
        num_passed_initial_filter += 1

        if not lpips_model:
            logger("Warning: LPIPS model not available, cannot score candidates.")
            continue 

        # ========== MSE + LPIPS 결합 스코어링 (cross_attn 완전 제거) ==========
        if use_mse:
            # MSE + LPIPS 결합 점수 계산
            result = calculate_combined_consistency_score(
                comp_pil_for_lpips, original_pil_lpips, original_mask_pil_lpips,
                comp_amodal_mask_pil_for_filtering, lpips_model, device_for_scoring,
                img_size=img_size_lpips_h, mse_weight=mse_weight, lpips_weight=lpips_weight
            )
            final_score = result['combined_score']
            score_details = {
                'lpips_score': result['lpips_score'],
                'mse_score': result['mse_score'],
                'combined_score': result['combined_score'],
                'mse_weight': mse_weight,
                'lpips_weight': lpips_weight,
                'scoring_mode': 'mse_lpips_combined'
            }
        else:
            # LPIPS만 사용 (기존 방식)
            lpips_score = calculate_lpips_consistency_score(
                comp_pil_for_lpips, original_pil_lpips, original_mask_pil_lpips, 
                comp_amodal_mask_pil_for_filtering, 
                lpips_model, device_for_scoring, img_size=img_size_lpips_h
            )
            final_score = lpips_score
            score_details = {
                'lpips_score': lpips_score,
                'mse_score': None,
                'scoring_mode': 'lpips_only'
            }
        
        num_actually_scored += 1
        
        scored_candidates_info.append({
            'image_path_relative': os.path.basename(comp_path),
            'amodal_mask_path_relative': comp_mask_filename if actual_mask_file_used else "derived_from_compN",
            'score': final_score,
            **score_details  # 모든 세부 점수들
        })
    
    logger(f"--- Summary for sample {sample_id}: Total comps: {num_all_comps}, Passed initial filter: {num_passed_initial_filter}, Actually scored & added: {len(scored_candidates_info)} ---")

    scored_candidates_info.sort(key=lambda x: x['score'], reverse=True)

    return {'sample_id': sample_id, 'candidates': scored_candidates_info}


def get_relative_path_from_data_root(sample_dir, data_root):
    """
    data_root를 기준으로 sample_dir의 상대 경로를 반환
    예: data_root="/path/to/train", sample_dir="/path/to/train/a/abbey/00000001"
    -> "a/abbey/00000001"
    """
    return os.path.relpath(sample_dir, data_root)

def collect_sample_dirs(data_root):
    """샘플 디렉토리 수집"""
    sample_dirs = []
    for root, dirs, files in os.walk(data_root):
        # original/mask/compN 등 필요한 파일이 있으면 샘플로 간주
        has_mask = any('mask' in f for f in files)
        has_img = any(f.endswith('.png') and 'mask' not in f for f in files)
        if has_mask and has_img:
            sample_dirs.append(root)
    return sample_dirs

def run_preprocessing(args):
    """오프라인 후보 정보 전처리 실행"""
    config = load_config_from_yaml(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device} for candidate scoring models.")

    scorer_params = getattr(config.data, 'candidate_scorer_params', ConfigNamespace({}))

    try:
        # LPIPS 모델만 초기화 (MSE+LPIPS 스코어링용)
        lpips_m = lpips.LPIPS(net=getattr(scorer_params, 'lpips_net', 'vgg'), verbose=False).to(device).eval()
        
        # 더미 모델들 (호환성을 위해)
        feat_extractor_m = SimpleFeatureExtractor()
        attn_scorer_m = CrossAttentionScorer()
        
        print("LPIPS model initialized successfully.")
    except Exception as e:
        print(f"Error initializing LPIPS model: {e}. Exiting.")
        import traceback
        traceback.print_exc()
        return

    splits_to_process = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    for current_split in splits_to_process:
        data_root = None
        if current_split == 'train' and hasattr(config.data, 'train_root'):
            data_root = os.path.expanduser(config.data.train_root)
        elif current_split in ['val', 'test']:
            # val, test 모두 같은 방식으로 처리
            key = f"{current_split}_root"
            if hasattr(config.data, key):
                data_root = os.path.expanduser(getattr(config.data, key))
                
        if not data_root or not os.path.isdir(data_root):
            print(f"Root directory for split '{current_split}' not found or not configured. Skipping.")
            continue
            
        output_base_dir = os.path.expanduser(config.data.offline_candidate_info_base_dir)
        output_dir_split = os.path.join(output_base_dir, current_split)
        os.makedirs(output_dir_split, exist_ok=True)
        print(f"\nProcessing '{current_split}' data from: {data_root}")
        print(f"Preprocessed info will be saved to: {output_dir_split}")

        # === 샘플 디렉토리 자동 수집 ===
        sample_dirs_map = {}
        for sample_dir_path in collect_sample_dirs(data_root):
            relative_path = os.path.relpath(sample_dir_path, data_root)
            sample_dirs_map[relative_path] = sample_dir_path
        print(f"Found {len(sample_dirs_map)} samples in '{current_split}' set.") 

        pbar = tqdm(sample_dirs_map.items(), desc=f"Processing {current_split}")
        for relative_path, sample_dir in pbar:
            # 상대 경로 기반으로 출력 디렉토리 구조 생성
            output_subdir = os.path.join(output_dir_split, os.path.dirname(relative_path))
            os.makedirs(output_subdir, exist_ok=True)
            
            # JSON 파일명은 sample_id_candidates.json 형식 유지
            sample_id = os.path.basename(sample_dir)
            output_json_path = os.path.join(output_subdir, f"{sample_id}_candidates.json")
            
            if os.path.exists(output_json_path) and not args.overwrite:
                continue         
            
            processed_data = process_single_sample_offline(
                sample_dir, config, lpips_m, feat_extractor_m, attn_scorer_m, 
                device, logger=print, pbar_instance=pbar
            )
            
            if processed_data and processed_data.get('candidates'):
                try:
                    with open(output_json_path, 'w') as f:
                        json.dump(processed_data, f, indent=2)
                except Exception as e_json:
                    print(f"Error saving JSON for {sample_id}: {e_json}")
            elif processed_data and not processed_data.get('candidates'):
                 print(f"No valid candidates to save for {sample_id} (candidates list was empty).")
            elif processed_data is None:     
                 print(f"Processing failed or no candidates for {sample_id}, no JSON saved.")
                 
    print("Offline candidate processing finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline Candidate Information Preprocessing for AFT-Net")    
    parser.add_argument('--config_path', type=str, default='./configs/aft_config.yaml',
                        help='Path to the main YAML configuration file.')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test', 'all'],
                        help="Which data split to process ('all' for train and val).")
    parser.add_argument('--use_gpu', action='store_true', help="Use GPU for scoring models if available.")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite existing JSON files if they exist.")
    
    args = parser.parse_args()
    run_preprocessing(args)