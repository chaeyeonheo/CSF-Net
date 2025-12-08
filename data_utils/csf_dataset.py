# data_utils/csf_dataset.py (오프라인 우선, 온라인 Fallback)

import os
import glob
import random
import json
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import lpips 

from .candidate_processing import score_and_select_candidates, filter_candidate_basic
from models.components.simple_feature_extractor import SimpleFeatureExtractor
from models.components.cross_attention_scorer import CrossAttentionScorer

class ConfigNamespace: # 임시 ConfigNamespace (실제로는 train_aft.py의 것 사용)
    def __init__(self, adict):
        self.__dict__.update(adict)
        for k, v in adict.items():
            if isinstance(v, dict): self.__dict__[k] = ConfigNamespace(v)

class CSFDataset(Dataset):

    def __init__(self, root_dir, Kmax, split='train', img_size=256, 
                transform_params=None, config_data=None, device_for_scoring='cpu', logger=print,
                use_candidate_scores=True, score_weight_alpha=0.3):  # 새로 추가
        self.root_dir = os.path.expanduser(root_dir)
        self.Kmax = Kmax
        self.split = split
        self.img_size = img_size
        self.transform_params = transform_params if transform_params else {}
        self.logger = logger if logger else print
        self.config_data = config_data if config_data else ConfigNamespace({}) 
    
        self.device_for_scoring = torch.device(device_for_scoring)
        self.use_offline_candidates = False # 기본값 False
        self.offline_info_dir_for_split = None
        self.use_candidate_scores = use_candidate_scores
        self.score_weight_alpha = score_weight_alpha  # Score 가중치 비율 (0.0~1.0)
    
        # ===== 수정된 부분: 더 강건한 설정 읽기 =====
        log_func_info = getattr(self.logger, 'info', print) # 로깅 함수 설정    
        log_func_warning = getattr(self.logger, 'warning', print) # 로깅 함수 설정
        log_func_error = getattr(self.logger, 'error', print) # 로깅 함수 설정
    
        # ConfigNamespace, dict, 또는 None 모든 경우 처리
        _use_offline_config = True # 오프라인 후보 사용 여부 기본 = True -> offline 모드 사용
        _offline_base_dir_config = None # 오프라인 후보 정보 디렉토리 경로
        
        # 디버깅 정보 먼저 출력
        log_func_info(f"CSFDataset Init: config_data 타입: {type(self.config_data)}")
        if hasattr(self.config_data, '__dict__'):
            available_attrs = list(self.config_data.__dict__.keys())
            log_func_info(f"CSFDataset Init: config_data 사용 가능한 속성들: {available_attrs}")
        
        if self.config_data is not None:
            # ConfigNamespace 객체인 경우
            if hasattr(self.config_data, '__dict__'):
                log_func_info(f"CSFDataset Init: ConfigNamespace 방식으로 설정 읽기 시도")
                
                # use_offline_candidates 속성 확인 및 읽기
                if hasattr(self.config_data, 'use_offline_candidates'):
                    _use_offline_config = self.config_data.use_offline_candidates
                    log_func_info(f"CSFDataset Init: use_offline_candidates 직접 접근 성공: {_use_offline_config}")
                else:
                    _use_offline_config = getattr(self.config_data, 'use_offline_candidates', False)
                    log_func_info(f"CSFDataset Init: use_offline_candidates getattr 사용: {_use_offline_config}")
                
                # offline_candidate_info_base_dir 속성 확인 및 읽기
                if hasattr(self.config_data, 'offline_candidate_info_base_dir'):
                    _offline_base_dir_config = self.config_data.offline_candidate_info_base_dir
                    log_func_info(f"CSFDataset Init: offline_candidate_info_base_dir 직접 접근 성공: {_offline_base_dir_config}")
                else:
                    _offline_base_dir_config = getattr(self.config_data, 'offline_candidate_info_base_dir', None)
                    log_func_info(f"CSFDataset Init: offline_candidate_info_base_dir getattr 사용: {_offline_base_dir_config}")
                    
            # dict인 경우 (fallback)
            elif hasattr(self.config_data, 'get'):
                log_func_info(f"CSFDataset Init: dict get() 방식으로 설정 읽기")
                _use_offline_config = self.config_data.get('use_offline_candidates', False)
                _offline_base_dir_config = self.config_data.get('offline_candidate_info_base_dir', None)
                
            # 직접 dict 접근 (fallback)
            elif isinstance(self.config_data, dict):
                log_func_info(f"CSFDataset Init: 직접 dict 접근")
                _use_offline_config = self.config_data.get('use_offline_candidates', False)
                _offline_base_dir_config = self.config_data.get('offline_candidate_info_base_dir', None)
                
            else:
                log_func_warning(f"CSFDataset Init: config_data 타입을 인식할 수 없음: {type(self.config_data)}")
                # 강제로 기본값 시도
                try:
                    _use_offline_config = getattr(self.config_data, 'use_offline_candidates', False)
                    _offline_base_dir_config = getattr(self.config_data, 'offline_candidate_info_base_dir', None)
                    log_func_info(f"CSFDataset Init: 강제 getattr 시도 성공")
                except Exception as e:
                    log_func_error(f"CSFDataset Init: 강제 getattr도 실패: {e}")
        else:
            log_func_info(f"CSFDataset Init: config_data가 None")
    
        # 최종 읽은 값들 로깅
        log_func_info(f"CSFDataset Init: 최종 읽은 'use_offline_candidates': {_use_offline_config} (타입: {type(_use_offline_config)})")
        log_func_info(f"CSFDataset Init: 최종 읽은 'offline_candidate_info_base_dir': {_offline_base_dir_config} (타입: {type(_offline_base_dir_config)})")
    
        # 실제 설정 적용
        if _use_offline_config and _offline_base_dir_config:
            expanded_base_dir = os.path.expanduser(_offline_base_dir_config)
            self.offline_info_dir_for_split = os.path.join(expanded_base_dir, self.split)
            log_func_info(f"CSFDataset Init: Constructed offline_info_dir_for_split: {self.offline_info_dir_for_split}")
            log_func_info(f"CSFDataset Init: Absolute path: {os.path.abspath(self.offline_info_dir_for_split)}")
            
            if os.path.isdir(self.offline_info_dir_for_split):
                self.use_offline_candidates = True 
                log_func_info(f"Offline mode ENABLED. Will load candidate info from: {self.offline_info_dir_for_split}")
            else:
                log_func_warning(f"Offline candidate info directory NOT FOUND: {self.offline_info_dir_for_split}")
                log_func_warning(f"Absolute path: {os.path.abspath(self.offline_info_dir_for_split)}")
                log_func_warning(f"Falling back to on-the-fly scoring.")
                self.use_offline_candidates = False 
        else:
            if not _use_offline_config:
                log_func_info("Offline mode disabled: use_offline_candidates is False.")
            if not _offline_base_dir_config:
                log_func_info("Offline mode disabled: offline_candidate_info_base_dir not configured.")
            log_func_info("Using on-the-fly candidate scoring.")
            self.use_offline_candidates = False
    
        # 샘플 수집
        self.samples = self._collect_samples()
        self._setup_transforms()
    
        # 온라인 스코어링 모델 초기화 -> 오프라인 모드 사용 여부에 따라 초기화 여부 결정
        self.lpips_model = None
        self.feature_extractor = None
        self.cross_attn_scorer = None
    
        if not self.use_offline_candidates: 
            log_func_info(f"Initializing candidate scoring models on {self.device_for_scoring} for on-the-fly scoring...")
            try:
                scorer_params_dict = {}
                
                # candidate_scorer_params 안전하게 읽기
                if hasattr(self.config_data, 'candidate_scorer_params'):
                    scorer_params_obj = getattr(self.config_data, 'candidate_scorer_params')
                    if isinstance(scorer_params_obj, ConfigNamespace):
                        scorer_params_dict = scorer_params_obj.__dict__
                    elif isinstance(scorer_params_obj, dict):
                        scorer_params_dict = scorer_params_obj
                    else:
                        log_func_warning(f"candidate_scorer_params 타입 인식 불가: {type(scorer_params_obj)}")
                
                self.lpips_model = lpips.LPIPS(net=scorer_params_dict.get('lpips_net', 'alex'), verbose=False).to(self.device_for_scoring).eval()
                feat_ext_out_dim = scorer_params_dict.get('feature_extractor_output_dim', 128)
                feat_ext_in_ch = scorer_params_dict.get('feature_extractor_in_channels', 3)
                self.feature_extractor = SimpleFeatureExtractor(
                    in_channels=feat_ext_in_ch, output_dim=feat_ext_out_dim,
                    img_size=scorer_params_dict.get('feature_extractor_img_size', 64)
                ).to(self.device_for_scoring).eval()
                self.cross_attn_scorer = CrossAttentionScorer(
                    feature_dim=feat_ext_out_dim,
                    num_heads=scorer_params_dict.get('cross_attention_heads', 4)
                ).to(self.device_for_scoring).eval()
                log_func_info(f"On-the-fly candidate scoring models initialized.")
            except Exception as e:
                log_func_error(f"Failed to initialize on-the-fly candidate scoring models: {e}. On-the-fly scoring might fail.")
        
        if not self.samples:
            log_func_warning(f"No samples found for split '{self.split}' in directory '{self.root_dir}'. Training/evaluation might not proceed.")
        else:
            log_func_info(f"CSFDataset initialized successfully with {len(self.samples)} samples. Offline mode: {self.use_offline_candidates}")



    

    def _setup_transforms(self): # 이미지 변환 설정
        self.to_tensor = transforms.ToTensor() # 이미지를 텐서로 변환 
        self.normalize = transforms.Normalize(
            mean=self.transform_params.get('mean', [0.485, 0.456, 0.406]), # 평균값 -> 기본값 사용
            std=self.transform_params.get('std', [0.229, 0.224, 0.225]) # 표준편차 -> 기본값 사용
        )
        self.use_augmentation = self.transform_params.get('use_augmentation', False) and self.split == 'train' # 데이터 증강 사용 여부
        self.hflip_prob = self.transform_params.get('hflip_prob', 0.0) # 수평 뒤집기 확률

    def _collect_samples(self): # 샘플 디렉토리 수집
        samples = [] # 샘플 디렉토리 리스트 초기화
        if not os.path.isdir(self.root_dir): # 데이터 루트 디렉토리 존재 여부 확인
            self.logger(f"Error: Root directory not found: {self.root_dir}") # 데이터 루트 디렉토리 존재 여부 확인
            return samples
        for alphabet_folder in sorted(os.listdir(self.root_dir)): # 알파벳 폴더 순회
            alphabet_path = os.path.join(self.root_dir, alphabet_folder) # 알파벳 폴더 경로 설정
            if not os.path.isdir(alphabet_path) or alphabet_folder == 'processing_progress.txt': continue # 알파벳 폴더가 디렉토리가 아니거나 processing_progress.txt 파일이면 넘어감
            for class_folder in sorted(os.listdir(alphabet_path)): # 클래스 폴더 순회
                class_path = os.path.join(alphabet_path, class_folder) # 클래스 폴더 경로 설정
                if not os.path.isdir(class_path): continue # 클래스 폴더가 디렉토리가 아니면 넘어감
                for sample_id_folder in sorted(os.listdir(class_path)): # 샘플 아이디 폴더 순회
                    sample_dir = os.path.join(class_path, sample_id_folder) # 샘플 아이디 폴더 경로 설정
                    if not os.path.isdir(sample_dir): continue # 샘플 아이디 폴더가 디렉토리가 아니면 넘어감
                    
                    original_path = os.path.join(sample_dir, 'original.png') # 원본 이미지 경로 설정
                    mask_path = os.path.join(sample_dir, 'mask.png') # 마스크 이미지 경로 설정
                    # JSON 파일 존재 여부는 __getitem__에서 확인하므로 여기서는 기본 파일만 체크
                    if os.path.exists(original_path) and os.path.exists(mask_path): # 원본 이미지와 마스크 이미지 존재 여부 확인
                        samples.append(sample_dir) # 샘플 디렉토리 리스트에 추가
                        
        log_func = getattr(self.logger, 'info', print) # 로깅 함수 설정     
        log_func(f"Collected {len(samples)} potential samples for split '{self.split}'. Validity will be checked in __getitem__.")
        return samples

    # def _load_image(self, path, mode='RGB', is_mask=False, target_size=None):
    #     _target_size_hw = target_size if target_size else (self.img_size, self.img_size)
    #     try:
    #         img = Image.open(path)
    #         if is_mask: img = img.convert('L')
    #         else: img = img.convert(mode)
    #         if img.size != _target_size_hw:
    #             interpolation_mode = TF.InterpolationMode.NEAREST if is_mask else TF.InterpolationMode.BILINEAR
    #             img = TF.resize(img, [_target_size_hw[1], _target_size_hw[0]], interpolation=interpolation_mode) # PIL H,W vs TF H,W
    #         return img
    #     except FileNotFoundError: self.logger(f"Error: Image file not found at {path} during __getitem__")
    #     except UnidentifiedImageError: self.logger(f"Error: Cannot identify image file {path} during __getitem__")
    #     except Exception as e: self.logger(f"Error loading image {path} during __getitem__: {e}")
    #     return None

    # data_utils/csf_dataset.py (_load_image 함수 수정)

    def _load_image(self, path, mode='RGB', is_mask=False, target_size=None): # 이미지 로드
        _target_size_hw = target_size if target_size else (self.img_size, self.img_size) # 대상 크기 설정
        try:
            img = Image.open(path) # 이미지 로드
            if is_mask: img = img.convert('L')
            else: img = img.convert(mode) # RGB로 고정
            if img.size != _target_size_hw:
                interpolation_mode = TF.InterpolationMode.NEAREST if is_mask else TF.InterpolationMode.BILINEAR
                # TF.resize는 [height, width] 순서의 크기를 받음
                img = TF.resize(img, [_target_size_hw[1], _target_size_hw[0]], interpolation=interpolation_mode)
            return img
        except FileNotFoundError: 
            # 수정된 로거 호출 방식
            if hasattr(self.logger, 'error'): self.logger.error(f"Image file not found at {path} in _load_image")
            else: print(f"Error: Image file not found at {path} in _load_image")
        except UnidentifiedImageError: 
            if hasattr(self.logger, 'error'): self.logger.error(f"Cannot identify image file {path} in _load_image")
            else: print(f"Error: Cannot identify image file {path} in _load_image")
        except Exception as e: 
            if hasattr(self.logger, 'error'): self.logger.error(f"Error loading image {path} in _load_image: {e}")
            else: print(f"Error loading image {path} in _load_image: {e}")
        return None
    
    # data_utils/csf_dataset.py의 _apply_shared_transforms 메서드를 다음과 같이 수정:

    def _apply_shared_transforms(self, pil_images_dict_with_maybe_lists):
        """
        🎯 역할: 모든 이미지에 일관된 변환 적용
        🔧 수정: Ground Truth와 후보 이미지 모두 동일한 정규화 적용
        """
        transformed_tensors = {}
        hflip_applied = self.use_augmentation and random.random() < self.hflip_prob
        
        for key, item_pil in pil_images_dict_with_maybe_lists.items():
            is_mask_key = 'mask' in key.lower()
            # 🔧 VGG 계산을 위해 모든 RGB 이미지는 동일하게 정규화
            should_normalize = not is_mask_key  # 마스크가 아닌 모든 이미지 정규화
            is_candidate_list = isinstance(item_pil, list)
            
            if is_candidate_list:
                tensor_list = []
                for img_pil_single_cand in item_pil:
                    if img_pil_single_cand is None: 
                        tensor_list.append(torch.zeros(3, self.img_size, self.img_size))
                        continue
                    if hflip_applied: 
                        img_pil_single_cand = TF.hflip(img_pil_single_cand)
                    tensor_img_single_cand = self.to_tensor(img_pil_single_cand)
                    # 🔧 후보 이미지도 정규화 적용 (VGG 일관성을 위해)
                    if should_normalize: 
                        tensor_img_single_cand = self.normalize(tensor_img_single_cand)
                        # print(f"[DATASET_DEBUG] 후보 이미지 정규화 적용. 범위: [{tensor_img_single_cand.min():.4f}, {tensor_img_single_cand.max():.4f}]")
                        
                    tensor_list.append(tensor_img_single_cand)
                transformed_tensors[key] = tensor_list
            else:
                img_pil = item_pil
                if img_pil is None:
                    num_channels = 1 if is_mask_key else 3
                    transformed_tensors[key] = torch.zeros(num_channels, self.img_size, self.img_size)
                    continue
                if hflip_applied: 
                    img_pil = TF.hflip(img_pil)
                tensor_img = self.to_tensor(img_pil)
                
                # 🔧 모든 RGB 이미지 정규화 (VGG 계산 일관성을 위해)
                if should_normalize: 
                    tensor_img = self.normalize(tensor_img)
                    if key == 'original':
                        # print(f"[DATASET_DEBUG] Ground Truth 정규화 적용. 범위: [{tensor_img.min():.4f}, {tensor_img.max():.4f}]")
                        pass
                        
                transformed_tensors[key] = tensor_img
        return transformed_tensors


    def __getitem__(self, idx): # 샘플 디렉토리 인덱스 접근
        sample_dir = self.samples[idx] # 샘플 디렉토리 경로 설정
        sample_id = os.path.basename(sample_dir) # 샘플 아이디 설정
    
        dummy_rgb = torch.zeros(3, self.img_size, self.img_size) # 더미 RGB 텐서 생성
        dummy_mask_tensor = torch.zeros(1, self.img_size, self.img_size) # 더미 마스크 텐서 생성
        dummy_candidates_tensor = torch.stack([dummy_rgb] * self.Kmax) # 더미 후보 텐서 생성
        error_return_dict = {'partial_image': dummy_rgb, 'original_mask': dummy_mask_tensor, # 에러 반환 딕셔너리 생성
                             'candidate_images_kmax': dummy_candidates_tensor,
                             'ground_truth_image': dummy_rgb, 'path': sample_dir, 'valid': False} # 에러 반환 딕셔너리 생성
    
        original_pil_model_input = self._load_image(os.path.join(sample_dir, 'original.png')) # 원본 이미지 로드    
        original_mask_pil_model_input = self._load_image(os.path.join(sample_dir, 'mask.png'), is_mask=True) # 마스크 이미지 로드
    
        if original_pil_model_input is None or original_mask_pil_model_input is None: # 원본 이미지 또는 마스크 이미지 로드 실패 시
            if hasattr(self.logger, 'error'): # 로깅 함수 설정
                self.logger.error(f"Base images for {sample_dir} could not be loaded. Returning dummy.")
            else: # 로깅 함수 설정
                self.logger(f"Error: Base images for {sample_dir} could not be loaded. Returning dummy.")
            return error_return_dict
    
        selected_candidate_final_pils = [] # 선택된 후보 이미지 리스트 초기화
        loaded_from_json = False # 오프라인 JSON 사용 여부 초기화
    
        if self.use_offline_candidates and self.offline_info_dir_for_split: # 오프라인 모드 활성화 및 경로 유효 여부 확인
            # === 수정된 부분: 계층적 구조에 맞게 JSON 경로 생성 ===
            data_root = os.path.expanduser(self.root_dir)
            try:
                # 상대 경로 계산 (예: sample_dir="/path/to/train/a/abbey/00000001" -> rel_path="a/abbey/00000001")
                rel_path = os.path.relpath(sample_dir, data_root)
                # JSON 파일이 저장된 디렉토리 (예: "./preprocessed_candidate_info_30/train/a/abbey/")
                json_subdir = os.path.join(self.offline_info_dir_for_split, os.path.dirname(rel_path))
                # 최종 JSON 파일 경로 (예: "./preprocessed_candidate_info_30/train/a/abbey/00000001_candidates.json")
                candidate_json_path = os.path.join(json_subdir, f"{sample_id}_candidates.json")
            except Exception as e:
                # 상대 경로 계산 실패 시 기존 방식으로 fallback
                if hasattr(self.logger, 'warning'):
                    self.logger.warning(f"Failed to calculate relative path for {sample_dir}: {e}. Using flat structure.")
                else:
                    self.logger(f"Warning: Failed to calculate relative path for {sample_dir}: {e}. Using flat structure.")
                candidate_json_path = os.path.join(self.offline_info_dir_for_split, f"{sample_id}_candidates.json")
            # === 수정 완료 ===
            
            if os.path.exists(candidate_json_path): # JSON 파일 존재 여부 확인
                try:
                    with open(candidate_json_path, 'r') as f: # JSON 파일 열기
                        candidate_data = json.load(f)
                    top_k_infos_from_json = candidate_data.get('candidates', [])[:self.Kmax] # 점수 정렬된 상태로 저장됨

                    # #############################################################
                    # [휴리스틱 적용 예시 - 주석]
                    # 모든 후보의 score가 threshold 미만이면 후보를 사용하지 않고 빈 리스트로 처리
                    # (즉, 네트워크에는 partial 이미지만 입력하거나, dummy 후보로 채움)
                    #
                    # threshold = 0.3  # (예시값, 실험적으로 결정)
                    # scores = [cand.get('score', 0.0) for cand in top_k_infos_from_json]
                    # if all(score < threshold for score in scores):
                    #     # 모든 후보의 점수가 threshold 미만이면 후보를 사용하지 않음
                    #     selected_candidate_final_pils = []
                    # else:
                    #     # 기존대로 후보 이미지 불러오기
                    #     for cand_info in top_k_infos_from_json:
                    #         img_path_relative = cand_info.get('image_path_relative')
                    #         if img_path_relative:
                    #             img_abs_path = os.path.join(sample_dir, img_path_relative)
                    #             cand_pil = self._load_image(img_abs_path)
                    #             selected_candidate_final_pils.append(
                    #                 cand_pil if cand_pil else Image.new('RGB', (self.img_size, self.img_size), (255,255,255))
                    #             )
                    #         else:
                    #             selected_candidate_final_pils.append(
                    #                 Image.new('RGB', (self.img_size, self.img_size), (255,255,255))
                    #             )
                    #
                    # 이후 코드에서 selected_candidate_final_pils가 빈 리스트면,
                    # partial 이미지만 네트워크에 넘기거나, dummy 후보로 채워서 넘기면 됨
                    # #############################################################

                    candidate_scores = []  # 새로 추가
                    for cand_info in top_k_infos_from_json:
                        img_path_relative = cand_info.get('image_path_relative')
                        score = cand_info.get('score', 0.5)  # Score 정보 추출
                        candidate_scores.append(score)  # Score 저장
                        
                        if img_path_relative:
                            img_abs_path = os.path.join(sample_dir, img_path_relative)
                            cand_pil = self._load_image(img_abs_path)
                            selected_candidate_final_pils.append(cand_pil if cand_pil else Image.new('RGB', (self.img_size, self.img_size), (255,255,255)))
                        else:
                            selected_candidate_final_pils.append(Image.new('RGB', (self.img_size, self.img_size), (255,255,255)))

                    # Kmax까지 점수도 패딩
                    while len(candidate_scores) < self.Kmax:
                        candidate_scores.append(0.1)  # 더미 후보는 낮은 점수
                    candidate_scores = candidate_scores[:self.Kmax]
                    loaded_from_json = True
                    # if hasattr(self.logger, 'debug'): self.logger.debug(f"Loaded {len(selected_candidate_final_pils)} candidates from JSON for {sample_id}")
                except Exception as e:
                    if hasattr(self.logger, 'error'):
                        self.logger.error(f"Error reading JSON {candidate_json_path} for {sample_id}: {e}. Falling back.")
                    else:
                        self.logger(f"Error reading JSON {candidate_json_path} for {sample_id}: {e}. Falling back.")
                    selected_candidate_final_pils = [] 
                    loaded_from_json = False
            else:
                # JSON 파일이 없을 때만 디버그 메시지 (너무 많은 로그 방지)
                if hasattr(self.logger, 'debug'): 
                    self.logger.debug(f"Offline JSON not found at {candidate_json_path} for {sample_id}. Using on-the-fly.")
        
        if not loaded_from_json: # 오프라인 JSON 사용 안하거나 실패 시 실시간 점수화
            # 실시간 점수화 모델이 준비 안된 경우: 그냥 첫 Kmax개 후보만 로드
            if not (self.lpips_model and self.feature_extractor and self.cross_attn_scorer):
                if hasattr(self.logger, 'warning'):
                    #self.logger.warning(f"On-the-fly scoring models not available for {sample_dir}. Loading first Kmax candidates without scoring.")
                    pass
                else:
                    #self.logger(f"Warning: On-the-fly scoring models not available for {sample_dir}. Loading first Kmax candidates without scoring.")
                    pass
                all_comp_files_fallback = sorted(glob.glob(os.path.join(sample_dir, 'comp[0-9]*.png')))
                for comp_path in all_comp_files_fallback:
                    if len(selected_candidate_final_pils) < self.Kmax:
                        img = self._load_image(comp_path)
                        if img: selected_candidate_final_pils.append(img)
                    else: break
            else:
                # 실시간 점수화 모델이 준비된 경우: 후보 이미지와 마스크를 모두 불러와서
                # score_and_select_candidates 함수로 점수화 및 상위 Kmax개 선택
                # ===== 수정된 부분: ConfigNamespace 안전한 접근 =====
                # 점수 계산용 이미지 로드
                scorer_params_dict = {}
                filter_params_dict = {}
                
                # candidate_scorer_params 안전하게 읽기
                if hasattr(self.config_data, 'candidate_scorer_params'):
                    scorer_params_obj = getattr(self.config_data, 'candidate_scorer_params')
                    if hasattr(scorer_params_obj, '__dict__'):
                        scorer_params_dict = scorer_params_obj.__dict__
                    elif isinstance(scorer_params_obj, dict):
                        scorer_params_dict = scorer_params_obj
                
                # candidate_filter_params 안전하게 읽기
                if hasattr(self.config_data, 'candidate_filter_params'):
                    filter_params_obj = getattr(self.config_data, 'candidate_filter_params')
                    if hasattr(filter_params_obj, '__dict__'):
                        filter_params_dict = filter_params_obj.__dict__
                    elif isinstance(filter_params_obj, dict):
                        filter_params_dict = filter_params_obj
                # ===== 수정 완료 =====
                
                img_size_lpips = scorer_params_dict.get('lpips_img_size', self.img_size)
                img_size_feat_ext = scorer_params_dict.get('feature_extractor_img_size', 64)
    
                original_pil_for_lpips = self._load_image(os.path.join(sample_dir, 'original.png'), target_size=(img_size_lpips, img_size_lpips))
                original_mask_pil_for_scoring = self._load_image(os.path.join(sample_dir, 'mask.png'), is_mask=True, target_size=(img_size_lpips, img_size_lpips))
                
                temp_orig_for_partial = self._load_image(os.path.join(sample_dir, 'original.png'), target_size=(img_size_feat_ext, img_size_feat_ext))
                temp_mask_for_partial = self._load_image(os.path.join(sample_dir, 'mask.png'), is_mask=True, target_size=(img_size_feat_ext, img_size_feat_ext))
    
                if not all([original_pil_for_lpips, original_mask_pil_for_scoring, temp_orig_for_partial, temp_mask_for_partial]):
                    if hasattr(self.logger, 'error'):
                        self.logger.error(f"Base images for on-the-fly scoring for {sample_dir} could not be loaded. Insufficient candidates.")
                    else:
                        self.logger(f"Error: Base images for on-the-fly scoring for {sample_dir} could not be loaded. Insufficient candidates.")
                else:
                    partial_pil_for_scoring = TF.to_pil_image(TF.to_tensor(temp_orig_for_partial) * (1.0 - (TF.to_tensor(temp_mask_for_partial) > 0.5).float()))
                    all_comp_files_realtime = sorted(glob.glob(os.path.join(sample_dir, 'comp[0-9]*.png')))
                    candidate_pil_list_for_scoring = []
                    candidate_amodal_mask_pil_list_for_scoring = []
                    
                    # 실시간 점수화 부분
                    for comp_path in all_comp_files_realtime:
                        comp_pil = self._load_image(comp_path, target_size=(img_size_lpips, img_size_lpips))
                        if not comp_pil:
                            continue  # 컴프 이미지 로드 실패시 스킵
                        
                        # 마스크 파일 경로 구성 (파일명만 변경)
                        comp_filename = os.path.basename(comp_path)
                        comp_dir = os.path.dirname(comp_path)
                        comp_mask_filename = comp_filename.replace("comp", "comp_mask_")
                        comp_mask_path = os.path.join(comp_dir, comp_mask_filename)
                        
                        # 먼저 comp_mask 파일이 존재하는지 확인
                        if os.path.exists(comp_mask_path):
                            comp_amodal_mask_pil = self._load_image(comp_mask_path, mode='L', is_mask=True, target_size=(img_size_lpips, img_size_lpips))
                        else:
                            # comp_mask 파일이 없으면 compN.png에서 마스크 추정
                            if hasattr(self.logger, 'debug'):
                                self.logger.debug(f"Mask file not found: {comp_mask_path}. Creating mask from comp image.")
                            
                            # compN.png에서 배경 색상(일반적으로 흰색)을 기반으로 마스크 생성
                            # 배경이 흰색(255,255,255)인 경우
                            comp_array = np.array(comp_pil)
                            # 배경 색상 기준 (흰색)
                            background_color = [255, 255, 255]
                            # 배경 색상과 일치하는 픽셀은 0, 나머지는 1로 설정
                            is_foreground = ~np.all(comp_array == background_color, axis=2)
                            mask_array = is_foreground.astype(np.uint8) * 255
                            comp_amodal_mask_pil = Image.fromarray(mask_array, mode='L')
                            
                            # 필요시 리사이징
                            if comp_amodal_mask_pil.size != (img_size_lpips, img_size_lpips):
                                comp_amodal_mask_pil = comp_amodal_mask_pil.resize((img_size_lpips, img_size_lpips), Image.NEAREST)
                        
                        if comp_pil and comp_amodal_mask_pil:
                            candidate_pil_list_for_scoring.append(comp_pil)
                            candidate_amodal_mask_pil_list_for_scoring.append(comp_amodal_mask_pil)
                    
                    if candidate_pil_list_for_scoring:
                        top_k_candidates_info = score_and_select_candidates(
                            original_pil=original_pil_for_lpips, 
                            original_mask_pil=original_mask_pil_for_scoring,
                            partial_pil=partial_pil_for_scoring, 
                            candidate_pil_list=candidate_pil_list_for_scoring, 
                            candidate_amodal_mask_pil_list=candidate_amodal_mask_pil_list_for_scoring,
                            Kmax=self.Kmax,
                            filter_params=filter_params_dict,
                            lpips_model=self.lpips_model,
                            feature_extractor=self.feature_extractor,
                            cross_attn_scorer=self.cross_attn_scorer,
                            device=self.device_for_scoring,
                            img_size_orig=img_size_lpips, 
                            img_size_feat=img_size_feat_ext,
                            logger=self.logger
                        )
                        for cand_info in top_k_candidates_info:
                            img_to_use = cand_info['image'] 
                            if img_to_use.size != (self.img_size, self.img_size):
                                img_to_use = img_to_use.resize((self.img_size, self.img_size), Image.BILINEAR)
                            selected_candidate_final_pils.append(img_to_use)
        
        num_final_candidates = len(selected_candidate_final_pils)
        if num_final_candidates < self.Kmax:
            for _ in range(self.Kmax - num_final_candidates):
                selected_candidate_final_pils.append(Image.new('RGB', (self.img_size, self.img_size), (255,255,255)))
        elif num_final_candidates > self.Kmax: 
             selected_candidate_final_pils = selected_candidate_final_pils[:self.Kmax]
        
        # data_utils/csf_dataset.py의 __getitem__ 메서드 마지막 부분 수정
        # 약 340줄 근처를 다음과 같이 수정:

        images_to_transform_pil = {
            'original': original_pil_model_input,
            'original_mask': original_mask_pil_model_input,
            'candidate_images_kmax': selected_candidate_final_pils
        }
        transformed_tensors = self._apply_shared_transforms(images_to_transform_pil)
    
        original_tensor = transformed_tensors['original']
        original_mask_tensor = (transformed_tensors['original_mask'] > 0.5).float()
        candidate_images_kmax_tensor = torch.stack(transformed_tensors['candidate_images_kmax'])
        
        # 🔧 partial_image 계산 - Ground Truth가 정규화 안되었으므로 그대로 사용
        partial_image_tensor = original_tensor * original_mask_tensor
        
        # 🔧 디버깅 정보 추가
        # print(f"[DATASET_DEBUG] GT 최종 범위: [{original_tensor.min():.4f}, {original_tensor.max():.4f}]")
        # print(f"[DATASET_DEBUG] Partial 최종 범위: [{partial_image_tensor.min():.4f}, {partial_image_tensor.max():.4f}]")
        # print(f"[DATASET_DEBUG] 후보 최종 범위: [{candidate_images_kmax_tensor.min():.4f}, {candidate_images_kmax_tensor.max():.4f}]")
    
        return {
            'partial_image': partial_image_tensor, 
            'original_mask': original_mask_tensor,
            'candidate_images_kmax': candidate_images_kmax_tensor,
            'ground_truth_image': original_tensor,
            'candidate_scores': torch.tensor(candidate_scores, dtype=torch.float32) if 'candidate_scores' in locals() else torch.ones(self.Kmax) * 0.5,  # 새로 추가
            'path': sample_dir, 
            'valid': True
        }
    

    
    def __len__(self):
        return len(self.samples)