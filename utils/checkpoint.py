# utils/checkpoint.py
import os
import torch

def save_checkpoint(epoch, model, optimizer, scheduler, filepath, logger=None, is_final=False, is_best=False):
    """모델, 옵티마이저, 스케줄러 상태를 체크포인트 파일에 저장"""
    if logger:
        logger.info(f"Saving checkpoint to {filepath} (Epoch: {epoch+1})")
    
    # DDP 모델의 경우 model.module 접근
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    # 디렉토리 생성
    checkpoint_dir = os.path.dirname(filepath)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    torch.save(checkpoint, filepath)
    
    if is_final:
        final_path = os.path.join(os.path.dirname(filepath), "model_final.pth")
        torch.save(model_state_dict, final_path) # 최종 모델은 state_dict만 저장할 수도 있음
        if logger: logger.info(f"Final model state_dict saved to {final_path}")
    if is_best: # TODO: Best 모델 저장 로직 (별도 파일 또는 덮어쓰기)
        best_path = os.path.join(os.path.dirname(filepath), "model_best.pth")
        torch.save(checkpoint, best_path)
        if logger: logger.info(f"Best checkpoint saved to {best_path}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, logger=None, device=None):
    """체크포인트 파일에서 모델, 옵티마이저, 스케줄러 상태 로드"""
    if not os.path.isfile(filepath):
        if logger:
            logger.warning(f"Checkpoint file not found: {filepath}. Starting from scratch.")
        return 0 # 시작 에폭 반환
    
    if logger:
        logger.info(f"Loading checkpoint from {filepath}")
        
    checkpoint = torch.load(filepath, map_location=device if device else 'cpu')
    
    # DDP 모델의 경우 model.module 접근
    model_to_load = model.module if hasattr(model, 'module') else model
    
    # 체크포인트가 직접 state_dict인지 확인
    # 모델 파라미터 키들이 직접 있는지 체크
    model_param_keys = ['context_patch_embed', 'candidate_patch_embed', 'decoder_fusion_stages', 'final_pred_head']
    is_direct_state_dict = any(any(str(key).startswith(param_key) for key in checkpoint.keys()) for param_key in model_param_keys)
    
    if is_direct_state_dict:
        # 체크포인트 자체가 state_dict인 경우
        if logger:
            logger.info("Detected direct state_dict format checkpoint")
        try:
            model_to_load.load_state_dict(checkpoint, strict=True)
            if logger:
                logger.info("Successfully loaded model state from direct state_dict")
        except RuntimeError as e:
            if logger:
                logger.warning(f"Failed to load model state_dict strictly: {e}. Trying with module prefix handling.")
            # 'module.' 접두사 처리
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:] if k.startswith('module.') else k # 'module.' 접두사 제거
                new_state_dict[name] = v
            try:
                model_to_load.load_state_dict(new_state_dict, strict=True)
                if logger:
                    logger.info("Successfully loaded model state with module prefix handling")
            except RuntimeError as e2:
                if logger: 
                    logger.error(f"Failed to load model state_dict even with 'module.' prefix handling: {e2}")
                return 0
        
        # 직접 state_dict인 경우 epoch 정보가 없으므로 0으로 시작
        start_epoch = 0
        if logger:
            logger.info(f"Checkpoint loaded as direct state_dict. Starting from epoch {start_epoch}.")
        
    else:
        # 표준 체크포인트 형식인 경우 (기존 로직)
        try:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            if logger:
                logger.warning(f"Failed to load model state_dict strictly: {e}. Trying with strict=False.")
            # 부분 로드 또는 키 이름 불일치 시도 (예: DDP <-> 단일 GPU 간 전환)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] if k.startswith('module.') else k # 'module.' 접두사 제거
                new_state_dict[name] = v
            try:
                model_to_load.load_state_dict(new_state_dict, strict=True)
            except RuntimeError as e2:
                if logger: logger.error(f"Failed to load model state_dict even with 'module.' prefix handling: {e2}")
                return 0
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                if logger: logger.warning(f"Could not load optimizer state: {e}")
                
        if scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                if logger: logger.warning(f"Could not load scheduler state: {e}")
                
        start_epoch = checkpoint.get('epoch', -1) + 1 # 다음 에폭부터 시작
        
        if logger:
            logger.info(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
        
    return start_epoch