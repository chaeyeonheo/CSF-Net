# losses/csf_losses.py (Region-based Selection 손실 추가 - 완성본)

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg_perceptual_loss import VGGPerceptualLoss 


# def safe_vgg_perceptual_calculation(vgg_loss_fn, pred_image, gt_image, mask=None):
#     """
#     🎯 역할: 완전히 새로 작성된 VGG 계산 (고정값 완전 제거)
#     📍 위치: losses/csf_losses.py의 safe_vgg_perceptual_calculation 함수를 이것으로 완전 교체
    
#     🚨 핵심: VGG 함수를 직접 호출하여 실제 계산 강제 수행
#     """
    
#     print("[AFT_DEBUG] 🔥 완전 새로운 VGG 계산 시작...")
    
#     # 1. 기본 안전성 체크
#     if pred_image.dim() != 4 or pred_image.size(1) != 3:
#         print(f"[VGG_ERROR] 잘못된 입력 차원: {pred_image.shape}")
#         return torch.tensor(0.01, device=pred_image.device, requires_grad=True), \
#                torch.tensor(0.01, device=pred_image.device, requires_grad=True), \
#                torch.tensor(0.001, device=pred_image.device, requires_grad=True)
    
#     B, C, H, W = pred_image.shape
    
#     # 2. 🚨 정규화된 범위를 0-1로 강제 변환 (매우 중요!)
#     # 정규화된 범위 [-2.1, 2.6]을 [0, 1]로 변환
#     print(f"[VGG_DEBUG] 입력 범위 변환 전:")
#     print(f"  pred_image: [{pred_image.min():.4f}, {pred_image.max():.4f}]")
#     print(f"  gt_image: [{gt_image.min():.4f}, {gt_image.max():.4f}]")
    
#     # 정규화 해제 (ImageNet 기준)
#     mean = torch.tensor([0.485, 0.456, 0.406], device=pred_image.device).view(1, 3, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225], device=pred_image.device).view(1, 3, 1, 1)
    
#     pred_denorm = pred_image * std + mean
#     gt_denorm = gt_image * std + mean
    
#     # 0-1 범위로 클램핑
#     pred_01 = torch.clamp(pred_denorm, 0.0, 1.0)
#     gt_01 = torch.clamp(gt_denorm, 0.0, 1.0)
    
#     print(f"[VGG_DEBUG] 범위 변환 후:")
#     print(f"  pred_01: [{pred_01.min():.4f}, {pred_01.max():.4f}]")
#     print(f"  gt_01: [{gt_01.min():.4f}, {gt_01.max():.4f}]")
    
#     # 3. VGG 최소 크기 보장
#     min_size = 224
#     if H < min_size or W < min_size:
#         print(f"[VGG_DEBUG] 크기 조정: {H}x{W} → {min_size}x{min_size}")
#         pred_01 = F.interpolate(pred_01, size=(min_size, min_size), mode='bilinear', align_corners=False)
#         gt_01 = F.interpolate(gt_01, size=(min_size, min_size), mode='bilinear', align_corners=False)
#         if mask is not None:
#             mask = F.interpolate(mask, size=(min_size, min_size), mode='nearest')
    
#     # 4. 🚨 VGG 계산 강제 수행 (완전히 새로운 방식)
#     try:
#         print("[VGG_DEBUG] 🔥 VGG 직접 계산 수행...")
        
#         # 마스크 처리 (선택적)
#         if mask is not None:
#             fill_mask = (1.0 - mask).expand_as(pred_01)
#             # 매우 부드러운 마스킹
#             weight = 0.1 + 0.9 * fill_mask
#             masked_pred = pred_01 * weight
#             masked_gt = gt_01 * weight
#         else:
#             masked_pred = pred_01
#             masked_gt = gt_01
        
#         print(f"[VGG_DEBUG] VGG 입력 최종 범위:")
#         print(f"  masked_pred: [{masked_pred.min():.4f}, {masked_pred.max():.4f}]")
#         print(f"  masked_gt: [{masked_gt.min():.4f}, {masked_gt.max():.4f}]")
        
#         # 🚨 VGG 함수 직접 강제 호출
#         with torch.enable_grad():  # gradient 계산 보장
#             perceptual_raw, content_raw, style_raw = vgg_loss_fn(masked_pred, masked_gt)
        
#         print(f"[VGG_DEBUG] 🔥 VGG 원시 계산 결과:")
#         print(f"  perceptual_raw: {perceptual_raw.item():.8f}")
#         print(f"  content_raw: {content_raw.item():.8f}")
#         print(f"  style_raw: {style_raw.item():.8f}")
        
#         # 5. 실제 계산값인지 검증
#         if abs(perceptual_raw.item() - 0.05) < 1e-6 and abs(content_raw.item() - 0.025) < 1e-6:
#             print("[VGG_ERROR] 🚨 여전히 고정값이 나오고 있습니다!")
#             print("[VGG_ERROR] VGG 함수 내부에 문제가 있을 수 있습니다.")
            
#             # 수동으로 간단한 L2 loss 계산
#             manual_content = F.mse_loss(masked_pred, masked_gt)
#             manual_perceptual = manual_content * 1.5
#             manual_style = torch.tensor(0.001, device=pred_image.device, requires_grad=True)
            
#             print(f"[VGG_DEBUG] 수동 계산 결과:")
#             print(f"  manual_content: {manual_content.item():.8f}")
#             print(f"  manual_perceptual: {manual_perceptual.item():.8f}")
            
#             return manual_perceptual, manual_content, manual_style
        
#         # 6. 정상적인 후처리
#         perceptual_loss = torch.clamp(perceptual_raw, 0.001, 1.0)
#         content_loss = torch.clamp(content_raw, 0.001, 1.0)
#         style_loss = torch.clamp(style_raw, 0.0, 0.1)
        
#         print(f"[VGG_DEBUG] 🎯 최종 VGG 결과:")
#         print(f"  perceptual: {perceptual_loss.item():.8f}")
#         print(f"  content: {content_loss.item():.8f}")
#         print(f"  style: {style_loss.item():.8f}")
        
#         return perceptual_loss, content_loss, style_loss
        
#     except Exception as e:
#         print(f"[VGG_ERROR] VGG 계산 실패: {e}")
#         import traceback
#         print(f"[VGG_ERROR] 상세 오류:\n{traceback.format_exc()}")
        
#         # 실패 시 수동 계산
#         try:
#             manual_loss = F.mse_loss(pred_01, gt_01)
#             manual_perceptual = manual_loss * 2.0
#             manual_content = manual_loss
#             manual_style = torch.tensor(0.001, device=pred_image.device, requires_grad=True)
            
#             print(f"[VGG_DEBUG] 수동 대체 계산:")
#             print(f"  manual_perceptual: {manual_perceptual.item():.8f}")
#             print(f"  manual_content: {manual_content.item():.8f}")
            
#             return manual_perceptual, manual_content, manual_style
            
#         except Exception as e2:
#             print(f"[VGG_ERROR] 수동 계산도 실패: {e2}")
            
#             # 최후의 안전 장치
#             safe_perceptual = torch.tensor(0.05, device=pred_image.device, requires_grad=True)
#             safe_content = torch.tensor(0.03, device=pred_image.device, requires_grad=True)
#             safe_style = torch.tensor(0.001, device=pred_image.device, requires_grad=True)
#             return safe_perceptual, safe_content, safe_style


class CSFLoss(nn.Module):
    def __init__(self,
                 # --- Reconstruction Loss 구성 항목 ---
                 l1_masked_weight=1.0,  # L1 loss on masked region (λ₁)
                 perceptual_weight=0.1,  # VGG-based perceptual loss (λₚ)
                 boundary_loss_weight=0.1,  # Gradient loss near inpainting boundary (λ_b)

                 # --- 🎯 빠진 파라미터들 추가 ---
                 confidence_bce_weight=0.5,  # ← 이 줄 추가!
                 confidence_error_correlation_weight=0.2,  # ← 이 줄 추가!

                 # --- Selection Consistency Loss 항목 ---
                 region_consistency_weight=0.1,  # Neighboring pixel selection similarity (λ_r)
                 selection_consistency_weight=0.05,  # Soft map smoothness (λ_s)
                 connectivity_loss_weight=0.05,  # Spatial connectivity of binary selection masks (λ_c)

                 # --- Hierarchical Consistency Loss ---
                 hierarchical_consistency_weight=0.1,  # Coarse-to-fine alignment loss (λ_hier)

                 # --- (Optional) 추가 손실 항목 ---
                 edge_preservation_weight=0.0,  # Edge sharpness (Sobel)
                 color_consistency_weight=0.0,  # Gradient-based color smoothness

                 # --- VGG 설정 (Perceptual Loss에 사용됨) ---
                 vgg_feature_layers=(3, 8, 17, 26, 35),
                 vgg_style_layers=None,
                 vgg_style_weight=0.0,
                 vgg_loss_type='l1',
                 vgg_weights_path=None,
                 device_for_vgg='cpu',

                 logger=print
                ):

        super().__init__()
        self.l1_masked_weight = l1_masked_weight
        self.perceptual_weight = perceptual_weight
        
        # 🎯 빠진 속성들 추가
        self.confidence_bce_weight = confidence_bce_weight  # ← 이 줄 추가!
        self.confidence_error_correlation_weight = confidence_error_correlation_weight  # ← 이 줄 추가!
        
        self.boundary_loss_weight = boundary_loss_weight
        
        # Region-based Selection 전용 손실 가중치
        self.region_consistency_weight = region_consistency_weight
        self.hierarchical_consistency_weight = hierarchical_consistency_weight
        self.selection_consistency_weight = selection_consistency_weight

        # 추가 손실 가중치
        self.edge_preservation_weight = edge_preservation_weight
        self.color_consistency_weight = color_consistency_weight
        
        self.logger = logger # 로거 저장

        self.l1_loss_fn = nn.L1Loss(reduction='none')
        
        # 🎯 조건문 수정
        if self.confidence_bce_weight > 0 or self.confidence_error_correlation_weight > 0:
            self.bce_with_logits_fn = nn.BCEWithLogitsLoss(reduction='none')
            self.smooth_l1_loss_fn = nn.SmoothL1Loss(reduction='none')
            
        
        if self.perceptual_weight > 0:
            # 로거 메소드 호출 방식 수정
            log_msg = f"Initializing VGGPerceptualLoss on device: {device_for_vgg} with weights: {vgg_weights_path}"
            if hasattr(self.logger, 'info'):
                self.logger.info(log_msg)
            else:
                print(f"INFO: {log_msg}")
                
            self.perceptual_loss_fn = VGGPerceptualLoss(
                feature_layer_indices=vgg_feature_layers,
                style_layer_indices=vgg_style_layers,
                style_weight=vgg_style_weight,
                loss_type=vgg_loss_type,
                vgg_weights_path=vgg_weights_path,
                logger=self.logger
            ).to(torch.device(device_for_vgg)).eval()
        else:
            self.perceptual_loss_fn = None
        
        if self.boundary_loss_weight > 0:
            sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y_kernel = torch.tensor([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('sobel_x', sobel_x_kernel)
            self.register_buffer('sobel_y', sobel_y_kernel)

    def compute_boundary_loss(self, pred_pixels_sigmoid, original_mask):
        """경계 선명도 손실 계산"""
        if self.boundary_loss_weight <= 0: 
            return torch.tensor(0.0, device=pred_pixels_sigmoid.device)
        pred_gray = pred_pixels_sigmoid.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(pred_gray, self.sobel_x.to(pred_gray.device), padding=1)
        grad_y = F.conv2d(pred_gray, self.sobel_y.to(pred_gray.device), padding=1)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        boundary_loss = (-grad_magnitude * (1.0 - original_mask)).sum() / ((1.0 - original_mask).sum() + 1e-8)
        return boundary_loss

    def compute_region_consistency_loss(self, predicted_pixels_sigmoid, original_mask, region_size=8):
        """
        영역 간 일관성 손실 - 인접한 영역들 간의 부드러운 전환을 보장
        """
        if self.region_consistency_weight <= 0:
            return torch.tensor(0.0, device=predicted_pixels_sigmoid.device)
            
        B, C, H, W = predicted_pixels_sigmoid.shape
        device = predicted_pixels_sigmoid.device
        
        # 마스크 영역에서만 계산
        masked_pred = predicted_pixels_sigmoid * (1.0 - original_mask)
        
        # 수평 방향 일관성 (좌-우 인접 영역)
        horizontal_diff = torch.abs(masked_pred[:, :, :, 1:] - masked_pred[:, :, :, :-1])
        horizontal_mask = (1.0 - original_mask)[:, :, :, 1:] * (1.0 - original_mask)[:, :, :, :-1]
        horizontal_loss = (horizontal_diff * horizontal_mask).sum() / (horizontal_mask.sum() + 1e-8)
        
        # 수직 방향 일관성 (상-하 인접 영역)
        vertical_diff = torch.abs(masked_pred[:, :, 1:, :] - masked_pred[:, :, :-1, :])
        vertical_mask = (1.0 - original_mask)[:, :, 1:, :] * (1.0 - original_mask)[:, :, :-1, :]
        vertical_loss = (vertical_diff * vertical_mask).sum() / (vertical_mask.sum() + 1e-8)
        
        # 영역 크기 기반 가중치 (더 큰 영역일수록 더 부드러워야 함)
        region_weight = max(1.0, region_size / 8.0)
        
        return region_weight * (horizontal_loss + vertical_loss) / 2.0

    def compute_hierarchical_consistency_loss(self, predicted_pixels_sigmoid, original_mask):
        """
        계층적 일관성 손실 - 다중 스케일에서의 일관성 보장
        """
        if self.hierarchical_consistency_weight <= 0:
            return torch.tensor(0.0, device=predicted_pixels_sigmoid.device)
        
        device = predicted_pixels_sigmoid.device
        B, C, H, W = predicted_pixels_sigmoid.shape
        
        # 다중 스케일로 다운샘플링
        scales = [1, 2, 4]  # 원본, 1/2, 1/4 크기
        scale_predictions = []
        scale_masks = []
        
        for scale in scales:
            if scale == 1:
                scale_pred = predicted_pixels_sigmoid
                scale_mask = original_mask
            else:
                target_h, target_w = H // scale, W // scale
                if target_h < 4 or target_w < 4:  # 너무 작으면 스킵
                    continue
                    
                scale_pred = F.interpolate(
                    predicted_pixels_sigmoid, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                scale_mask = F.interpolate(
                    original_mask, 
                    size=(target_h, target_w), 
                    mode='nearest'
                )
            
            scale_predictions.append(scale_pred)
            scale_masks.append(scale_mask)
        
        if len(scale_predictions) < 2:
            return torch.tensor(0.0, device=device)
        
        # 스케일 간 일관성 계산
        consistency_loss = 0.0
        num_pairs = 0
        
        for i in range(len(scale_predictions) - 1):
            pred_coarse = scale_predictions[i]
            pred_fine = scale_predictions[i + 1]
            mask_coarse = scale_masks[i]
            
            # Fine을 Coarse 크기로 다운샘플링
            pred_fine_downsampled = F.interpolate(
                pred_fine, 
                size=pred_coarse.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            
            # 마스크 영역에서의 차이 계산
            diff = torch.abs(pred_coarse - pred_fine_downsampled)
            masked_diff = diff * (1.0 - mask_coarse)
            
            pair_loss = masked_diff.sum() / ((1.0 - mask_coarse).sum() + 1e-8)
            consistency_loss += pair_loss
            num_pairs += 1
        
        return consistency_loss / max(num_pairs, 1)

    def compute_selection_consistency_loss(self, selection_weights, original_mask):
        """
        선택 일관성 손실 - Region 선택이 공간적으로 일관되도록 보장
        Args:
            selection_weights: [B, K, H, W] - 각 후보의 선택 가중치
            original_mask: [B, 1, H, W] - 원본 마스크
        """
        if selection_weights is None or self.selection_consistency_weight <= 0:
            return torch.tensor(0.0, device=original_mask.device)
        
        B, K, H, W = selection_weights.shape
        device = selection_weights.device
        
        # 마스크 영역에서만 계산
        masked_weights = selection_weights * (1.0 - original_mask)  # [B, K, H, W]
        
        # 각 후보별로 선택 영역의 연결성 평가
        consistency_losses = []
        
        for k in range(K):
            weight_map = masked_weights[:, k:k+1, :, :]  # [B, 1, H, W]
            
            # 이진화 (threshold=0.5)
            binary_selection = (weight_map > 0.5).float()
            
            # 연결성 평가: 인접 픽셀 간 차이
            # 수평 연결성
            h_diff = torch.abs(binary_selection[:, :, :, 1:] - binary_selection[:, :, :, :-1])
            h_loss = h_diff.sum() / (h_diff.numel() + 1e-8)
            
            # 수직 연결성
            v_diff = torch.abs(binary_selection[:, :, 1:, :] - binary_selection[:, :, :-1, :])
            v_loss = v_diff.sum() / (v_diff.numel() + 1e-8)
            
            consistency_losses.append(h_loss + v_loss)
        
        return sum(consistency_losses) / len(consistency_losses)

# losses/csf_losses.py의 CSFLoss forward 메서드 앞부분을 다음과 같이 수정:

# losses/csf_losses.py의 CSFLoss.forward 메서드 (line 390 근처)
# 전체 메서드를 다음으로 교체:

    def forward(self, predicted_pixels_logits, confidence_map_logits, ground_truth_image, original_mask,  
                model_info=None): 
        """ 
        Args: 
            model_info: 모델에서 추가 정보 전달용 (Region-based Selection 정보 등) 
        """ 
        loss_dict = {} 
        
        # ======================================================================
        # 🎯 1. 입력 안정화 (가장 먼저 수행)
        # 모델 출력은 항상 sigmoid를 통과시켜 [0, 1] 범위로 만듭니다.
        predicted_pixels_sigmoid = torch.sigmoid(predicted_pixels_logits) 

        # Ground Truth 이미지가 [-2.1, 2.6] 같은 정규화된 범위일 경우, [0, 1]로 역정규화합니다.
        # 이는 모든 손실 계산의 일관성을 보장하는 핵심 단계입니다.
        if ground_truth_image.min() < -0.1 or ground_truth_image.max() > 1.1:
            mean = torch.tensor([0.485, 0.456, 0.406], device=ground_truth_image.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=ground_truth_image.device).view(1, 3, 1, 1)
            ground_truth_01 = ground_truth_image * std + mean
        else:
            ground_truth_01 = ground_truth_image
        
        # 최종적으로 모든 입력 이미지를 [0, 1] 범위로 강제 클램핑하여 안전성을 확보합니다.
        predicted_pixels_01 = torch.clamp(predicted_pixels_sigmoid, 0.0, 1.0)
        ground_truth_01 = torch.clamp(ground_truth_01, 0.0, 1.0)
        # ======================================================================

        fill_area = (1.0 - original_mask).sum()
        if fill_area == 0:
            # 채울 영역이 없는 경우, 모든 손실을 0으로 반환하여 학습 오류를 방지합니다.
            for key in ['total_loss', 'l1_masked', 'conf_bce', 'conf_err_corr', 'perceptual', 'p_content', 'p_style', 'boundary', 'region_consistency', 'hierarchical_consistency', 'selection_consistency', 'edge_preservation', 'color_consistency']:
                loss_dict[key] = torch.tensor(0.0, device=predicted_pixels_logits.device)
            return loss_dict

        # 기본 L1 손실 (마스크 영역)
        if self.l1_masked_weight > 0: 
            l1_pixel = self.l1_loss_fn(predicted_pixels_01, ground_truth_01) 
            fill_mask = (1.0 - original_mask).expand_as(l1_pixel)
            masked_l1 = l1_pixel * fill_mask
            total_elements = fill_area * predicted_pixels_01.size(1)
            l1_loss_value = masked_l1.sum() / (total_elements + 1e-8)
            loss_dict['l1_masked'] = self.l1_masked_weight * l1_loss_value
        else: 
            loss_dict['l1_masked'] = torch.tensor(0.0, device=predicted_pixels_logits.device)

        # 신뢰도 BCE 손실
        if self.confidence_bce_weight > 0 and hasattr(self, 'bce_with_logits_fn'): 
            with torch.no_grad(): 
                err_thresh = 0.1 
                pixel_err = torch.abs(predicted_pixels_01 - ground_truth_01).mean(dim=1,keepdim=True) 
                target_conf_bce = (pixel_err < err_thresh).float() 
            bce_pix = self.bce_with_logits_fn(confidence_map_logits, target_conf_bce) 
            masked_bce = bce_pix * (1.0 - original_mask)
            bce_loss_value = masked_bce.sum() / (fill_area + 1e-8)
            loss_dict['conf_bce'] = self.confidence_bce_weight * bce_loss_value
        else: 
            loss_dict['conf_bce'] = torch.tensor(0.0, device=predicted_pixels_logits.device)

        # 신뢰도-에러 상관관계 손실
        if self.confidence_error_correlation_weight > 0 and hasattr(self, 'smooth_l1_loss_fn'): 
            with torch.no_grad(): 
                pix_err_norm = torch.clamp(torch.abs(predicted_pixels_01 - ground_truth_01).mean(dim=1,keepdim=True),0,1) 
                target_conf_err_corr = 1.0 - pix_err_norm
            smooth_l1_pix = self.smooth_l1_loss_fn(torch.sigmoid(confidence_map_logits), target_conf_err_corr) 
            masked_smooth = smooth_l1_pix * (1.0 - original_mask)
            corr_loss_value = masked_smooth.sum() / (fill_area + 1e-8)
            loss_dict['conf_err_corr'] = self.confidence_error_correlation_weight * corr_loss_value
        else: 
            loss_dict['conf_err_corr'] = torch.tensor(0.0, device=predicted_pixels_logits.device)

        # 지각적 손실 (VGG)
        # if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
        #     try:
        #         p_loss_total, p_loss_content, p_loss_style = self.perceptual_loss_fn(
        #             predicted_pixels_01, ground_truth_01, mask=original_mask
        #         )
                
        #         # ================= 🎯 VGG Loss Clipping (핵심 해결책) 🎯 =================
        #         # VGG Loss 값이 비정상적으로 폭주하는 것을 막기 위해 상한선을 설정합니다.
        #         # 예를 들어, p_content 값이 100이 나오더라도 1.0으로 강제 제한됩니다.
        #         # 이 값은 경험적으로 조절할 수 있으며, 1.0 ~ 3.0 사이가 일반적입니다.
        #         p_loss_total_clipped = torch.clamp(p_loss_total, max=1.5)
        #         p_content_clipped = torch.clamp(p_content, max=1.5)
        #         p_style_clipped = torch.clamp(p_style, max=1.5)
        #         # ========================================================================
                
        #         loss_dict['perceptual'] = self.perceptual_weight * p_loss_total_clipped
        #         loss_dict['p_content'] = p_content_clipped # 정보용 로그에는 클리핑된 값을 기록
        #         loss_dict['p_style'] = p_style_clipped   # 정보용 로그에는 클리핑된 값을 기록

        #     except Exception as e:
        #         # print(f"[AFT_ERROR] ❌ 지각적 손실 계산 실패: {e}")
        #         loss_dict['perceptual'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        #         loss_dict['p_content'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        #         loss_dict['p_style'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        # else:
        #     loss_dict['perceptual'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        #     loss_dict['p_content'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        #     loss_dict['p_style'] = torch.tensor(0.0, device=predicted_pixels_logits.device)

        # 지각적 손실 (VGG) 부분을 찾아서 이렇게 교체:
        if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
            # print(f"[VGG_DEBUG] === VGG Loss 계산 시작 ===")
            # print(f"[VGG_DEBUG] self.perceptual_weight: {self.perceptual_weight}")
            # print(f"[VGG_DEBUG] self.perceptual_loss_fn: {type(self.perceptual_loss_fn)}")
            # print(f"[VGG_DEBUG] predicted_pixels_01 shape: {predicted_pixels_01.shape}")
            # print(f"[VGG_DEBUG] predicted_pixels_01 range: [{predicted_pixels_01.min():.4f}, {predicted_pixels_01.max():.4f}]")
            # print(f"[VGG_DEBUG] ground_truth_01 shape: {ground_truth_01.shape}")  
            # print(f"[VGG_DEBUG] ground_truth_01 range: [{ground_truth_01.min():.4f}, {ground_truth_01.max():.4f}]")
            # print(f"[VGG_DEBUG] original_mask shape: {original_mask.shape}")
            # print(f"[VGG_DEBUG] original_mask range: [{original_mask.min():.4f}, {original_mask.max():.4f}]")
            
            try:
                # print(f"[VGG_DEBUG] VGG 함수 호출 중...")
                p_loss_total, p_loss_content, p_loss_style = self.perceptual_loss_fn(
                    predicted_pixels_01, ground_truth_01, mask=original_mask
                )
                
                # print(f"[VGG_DEBUG] VGG 원시 출력:")
                # print(f"[VGG_DEBUG]   - p_loss_total: {p_loss_total.item():.8f}")
                # print(f"[VGG_DEBUG]   - p_loss_content: {p_loss_content.item():.8f}")
                # print(f"[VGG_DEBUG]   - p_loss_style: {p_loss_style.item():.8f}")
                
                # 클리핑 전 체크
                # if p_loss_total.item() == 0.0:
                    # print(f"[VGG_ERROR] 🚨 VGG 출력이 완전히 0입니다! 원인 조사 필요!")
                    # print(f"[VGG_ERROR] gradient 체크: p_loss_total.requires_grad = {p_loss_total.requires_grad}")
                    # print(f"[VGG_ERROR] device 체크: p_loss_total.device = {p_loss_total.device}")
                
                # VGG Loss 클리핑
                p_loss_total_clipped = torch.clamp(p_loss_total, max=1.5)
                p_content_clipped = torch.clamp(p_loss_content, max=1.5)
                p_style_clipped = torch.clamp(p_loss_style, max=1.5)
                
                # print(f"[VGG_DEBUG] 클리핑 후:")
                # print(f"[VGG_DEBUG]   - p_loss_total_clipped: {p_loss_total_clipped.item():.8f}")
                # print(f"[VGG_DEBUG]   - p_content_clipped: {p_content_clipped.item():.8f}")
                # print(f"[VGG_DEBUG]   - p_style_clipped: {p_style_clipped.item():.8f}")
                
                final_perceptual = self.perceptual_weight * p_loss_total_clipped
                # print(f"[VGG_DEBUG] 최종 계산: {self.perceptual_weight:.6f} * {p_loss_total_clipped.item():.8f} = {final_perceptual.item():.8f}")
                
                loss_dict['perceptual'] = final_perceptual
                loss_dict['p_content'] = p_content_clipped
                loss_dict['p_style'] = p_style_clipped
                
                # print(f"[VGG_DEBUG] === VGG Loss 계산 완료 ===")

            except Exception as e:
                # print(f"[VGG_ERROR] ❌ VGG 계산 실패: {e}")
                import traceback
                # print(f"[VGG_ERROR] 스택 트레이스:\n{traceback.format_exc()}")
                loss_dict['perceptual'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
                loss_dict['p_content'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
                loss_dict['p_style'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        else:
            # print(f"[VGG_DEBUG] VGG 비활성화됨:")
            # print(f"[VGG_DEBUG]   - perceptual_weight: {self.perceptual_weight}")
            # print(f"[VGG_DEBUG]   - perceptual_loss_fn is None: {self.perceptual_loss_fn is None}")
            loss_dict['perceptual'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
            loss_dict['p_content'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
            loss_dict['p_style'] = torch.tensor(0.0, device=predicted_pixels_logits.device)




        # 경계 손실
        if self.boundary_loss_weight > 0 and hasattr(self, 'sobel_x'): 
            boundary_loss = self.compute_boundary_loss(predicted_pixels_01, original_mask)
            loss_dict['boundary'] = self.boundary_loss_weight * torch.clamp(boundary_loss, -2.0, 0.0)
        else: 
            loss_dict['boundary'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        
        # ===== Region-based Selection 전용 손실들 ===== 
        region_size = model_info.get('region_size', 8) if model_info else 8

        # 영역 일관성 손실
        if self.region_consistency_weight > 0: 
            region_loss = self.compute_region_consistency_loss(predicted_pixels_01, original_mask, region_size)
            loss_dict['region_consistency'] = self.region_consistency_weight * region_loss
        else: 
            loss_dict['region_consistency'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        
        # 계층적 일관성 손실
        if self.hierarchical_consistency_weight > 0: 
            hierarchical_loss = self.compute_hierarchical_consistency_loss(predicted_pixels_01, original_mask)
            loss_dict['hierarchical_consistency'] = self.hierarchical_consistency_weight * hierarchical_loss
        else: 
            loss_dict['hierarchical_consistency'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        
        # 선택 일관성 손실
        selection_weights = model_info.get('selection_weights') if model_info else None
        if self.selection_consistency_weight > 0 and selection_weights is not None: 
            selection_consistency_loss = self.compute_selection_consistency_loss(selection_weights, original_mask) 
            loss_dict['selection_consistency'] = self.selection_consistency_weight * selection_consistency_loss 
        else: 
            loss_dict['selection_consistency'] = torch.tensor(0.0, device=predicted_pixels_logits.device)

        # 에지 보존 손실
        if hasattr(self, 'edge_preservation_weight') and self.edge_preservation_weight > 0:
            edge_loss = self.compute_edge_preservation_loss(predicted_pixels_01, ground_truth_01, original_mask)
            loss_dict['edge_preservation'] = self.edge_preservation_weight * edge_loss
        else:
            loss_dict['edge_preservation'] = torch.tensor(0.0, device=predicted_pixels_logits.device)

        # 색상 일관성 손실
        if hasattr(self, 'color_consistency_weight') and self.color_consistency_weight > 0:
            color_loss = self.compute_color_consistency_loss(predicted_pixels_01, ground_truth_01, original_mask)
            loss_dict['color_consistency'] = self.color_consistency_weight * color_loss
        else:
            loss_dict['color_consistency'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        
        # 전체 손실 계산
        total_loss = sum(loss for key, loss in loss_dict.items() if 'p_' not in key) # p_content, p_style은 정보용
        loss_dict['total_loss'] = total_loss 
        
        return loss_dict

class RegionBasedCSFLoss(CSFLoss):
    """
    Region-based Selection에 특화된 AFT 손실 함수
    기본 CSFLoss를 상속받아 Region 전용 기능 추가
    """
    
    def __init__(self, **kwargs):
        # Region-based에 최적화된 기본 가중치 설정
        region_defaults = {
            'l1_masked_weight': 1.2,
            'perceptual_weight': 0.4,
            'confidence_bce_weight': 0.3,
            'confidence_error_correlation_weight': 0.1,
            'boundary_loss_weight': 0.1,
            'region_consistency_weight': 0.2,
            'hierarchical_consistency_weight': 0.1
        }
        
        # 기본값과 사용자 설정 병합
        for key, default_value in region_defaults.items():
            if key not in kwargs:
                kwargs[key] = default_value
        
        super().__init__(**kwargs)
        
        # Region-based 전용 추가 설정
        self.region_scales = [32, 16, 8]  # Region 크기들
        self.selection_consistency_weight = kwargs.get('selection_consistency_weight', 0.1)
        
    def compute_selection_consistency_loss(self, selection_weights, original_mask):
        """
        선택 일관성 손실 - Region 선택이 공간적으로 일관되도록 보장
        
        Args:
            selection_weights: [B, K, H, W] - 각 후보의 선택 가중치
            original_mask: [B, 1, H, W] - 원본 마스크
        """
        if selection_weights is None or self.selection_consistency_weight <= 0:
            return torch.tensor(0.0, device=original_mask.device)
        
        B, K, H, W = selection_weights.shape
        device = selection_weights.device
        
        # 마스크 영역에서만 계산
        masked_weights = selection_weights * (1.0 - original_mask)  # [B, K, H, W]
        
        # 각 후보별로 선택 영역의 연결성 평가
        consistency_losses = []
        
        for k in range(K):
            weight_map = masked_weights[:, k:k+1, :, :]  # [B, 1, H, W]
            
            # 이진화 (threshold=0.5)
            binary_selection = (weight_map > 0.5).float()
            
            # 연결성 평가: 인접 픽셀 간 차이
            # 수평 연결성
            h_diff = torch.abs(binary_selection[:, :, :, 1:] - binary_selection[:, :, :, :-1])
            h_loss = h_diff.sum() / (h_diff.numel() + 1e-8)
            
            # 수직 연결성
            v_diff = torch.abs(binary_selection[:, :, 1:, :] - binary_selection[:, :, :-1, :])
            v_loss = v_diff.sum() / (v_diff.numel() + 1e-8)
            
            consistency_losses.append(h_loss + v_loss)
        
        return sum(consistency_losses) / len(consistency_losses)
    
    def compute_multi_scale_consistency_loss(self, predicted_pixels_sigmoid, original_mask, scale_predictions=None):
        """
        다중 스케일 일관성 손실 - Region의 계층적 구조 반영
        
        Args:
            predicted_pixels_sigmoid: [B, 3, H, W] - 최종 예측
            original_mask: [B, 1, H, W] - 원본 마스크
            scale_predictions: dict - 각 스케일별 예측 결과 (선택적)
        """
        if scale_predictions is None:
            # 스케일 예측이 없으면 기본 계층적 일관성 손실 사용
            return self.compute_hierarchical_consistency_loss(predicted_pixels_sigmoid, original_mask)
        
        device = predicted_pixels_sigmoid.device
        total_loss = 0.0
        num_comparisons = 0
        
        # 각 스케일 간 일관성 검사
        for scale_name, scale_pred in scale_predictions.items():
            if scale_pred is None:
                continue
                
            # 스케일 예측을 최종 크기로 업샘플링
            if scale_pred.shape != predicted_pixels_sigmoid.shape:
                scale_pred_upsampled = F.interpolate(
                    scale_pred, 
                    size=predicted_pixels_sigmoid.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                scale_pred_upsampled = scale_pred
            
            # 마스크 영역에서의 차이
            diff = torch.abs(predicted_pixels_sigmoid - scale_pred_upsampled)
            masked_diff = diff * (1.0 - original_mask)
            
            scale_loss = masked_diff.sum() / ((1.0 - original_mask).sum() + 1e-8)
            total_loss += scale_loss
            num_comparisons += 1
        
        return total_loss / max(num_comparisons, 1)
    
    def forward(self, predicted_pixels_logits, confidence_map_logits, ground_truth_image, original_mask, 
                model_info=None):
        """
        Region-based Selection에 특화된 손실 계산
        
        Args:
            model_info: {
                'selection_weights': [B, K, H, W] - 후보 선택 가중치
                'scale_predictions': dict - 스케일별 예측 결과
                'region_size': int - 사용된 region 크기
                'selection_info': dict - 추가 선택 정보
            }
        """
        # 기본 손실들 계산
        loss_dict = super().forward(
            predicted_pixels_logits, confidence_map_logits, 
            ground_truth_image, original_mask, model_info
        )
        
        # Region-based 전용 추가 손실들
        if model_info:
            # 선택 일관성 손실
            if 'selection_weights' in model_info:
                selection_consistency_loss = self.compute_selection_consistency_loss(
                    model_info['selection_weights'], original_mask
                )
                loss_dict['selection_consistency'] = self.selection_consistency_weight * selection_consistency_loss
                
                # 전체 손실에 추가
                loss_dict['total_loss'] += loss_dict['selection_consistency']
            else:
                loss_dict['selection_consistency'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
            
            # 다중 스케일 일관성 손실 (더 정교한 버전)
            if 'scale_predictions' in model_info and self.hierarchical_consistency_weight > 0:
                # 기존 hierarchical_consistency를 더 정교한 버전으로 대체
                multi_scale_loss = self.compute_multi_scale_consistency_loss(
                    torch.sigmoid(predicted_pixels_logits), original_mask, 
                    model_info['scale_predictions']
                )
                # 기존 hierarchical_consistency 대체
                loss_dict['total_loss'] -= loss_dict['hierarchical_consistency']
                loss_dict['hierarchical_consistency'] = self.hierarchical_consistency_weight * multi_scale_loss
                loss_dict['total_loss'] += loss_dict['hierarchical_consistency']
        else:
            loss_dict['selection_consistency'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        
        return loss_dict
    
# losses/csf_losses.py 파일 끝에 추가할 코드

class EnhancedRegionBasedCSFLoss(RegionBasedCSFLoss):
    """
    개선된 Region-based AFT 손실 함수 - 픽셀 레벨 선택 최적화
    """
    
    def __init__(self, **kwargs):
        # 픽셀 레벨에 최적화된 기본 가중치 설정
        pixel_optimized_defaults = {
            'l1_masked_weight': 1.5,  # 픽셀 정확도 중요
            'perceptual_weight': 0.3,  # 지각적 품질 강화
            'confidence_bce_weight': 0.4,
            'confidence_error_correlation_weight': 0.15,
            'boundary_loss_weight': 0.15,  # 경계 선명도 중요
            'region_consistency_weight': 0.25,  # 픽셀 간 일관성
            'hierarchical_consistency_weight': 0.2,
            'selection_consistency_weight': 0.2,  # 선택 일관성 강화
            'edge_preservation_weight': 0.1,  # 새로 추가
            'color_consistency_weight': 0.1   # 새로 추가
        }
        
        # 기본값과 사용자 설정 병합
        for key, default_value in pixel_optimized_defaults.items():
            if key not in kwargs:
                kwargs[key] = default_value
        
        super().__init__(**kwargs)
        
        # 픽셀 레벨 전용 손실 가중치
        self.edge_preservation_weight = kwargs.get('edge_preservation_weight', 0.1)
        self.color_consistency_weight = kwargs.get('color_consistency_weight', 0.1)
        
        # Sobel 필터 (에지 보존용)
        if self.edge_preservation_weight > 0:
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer('edge_sobel_x', sobel_x)
            self.register_buffer('edge_sobel_y', sobel_y)
    
    # losses/csf_losses.py의 EnhancedRegionBasedCSFLoss 클래스에 추가할 메서드들
    # 클래스 내부의 기존 메서드들 다음에 추가하세요

    def compute_edge_preservation_loss(self, pred_image, gt_image, mask): 
        """에지 보존 손실 - 픽셀 레벨 선택에서 중요""" 
        if not hasattr(self, 'edge_preservation_weight') or self.edge_preservation_weight <= 0: 
            return torch.tensor(0.0, device=pred_image.device) 
         
        if not hasattr(self, 'edge_sobel_x') or not hasattr(self, 'edge_sobel_y'):
            print("[AFT_WARNING] edge_sobel_x/y 필터가 없습니다. 에지 보존 손실을 0으로 반환합니다.")
            return torch.tensor(0.0, device=pred_image.device)
         
        # 그레이스케일 변환 
        pred_gray = pred_image.mean(dim=1, keepdim=True) 
        gt_gray = gt_image.mean(dim=1, keepdim=True) 
         
        # 에지 검출 
        try:
            pred_edge_x = F.conv2d(pred_gray, self.edge_sobel_x.to(pred_gray.device), padding=1) 
            pred_edge_y = F.conv2d(pred_gray, self.edge_sobel_y.to(pred_gray.device), padding=1) 
            pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8) 
             
            gt_edge_x = F.conv2d(gt_gray, self.edge_sobel_x.to(gt_gray.device), padding=1) 
            gt_edge_y = F.conv2d(gt_gray, self.edge_sobel_y.to(gt_gray.device), padding=1) 
            gt_edge = torch.sqrt(gt_edge_x**2 + gt_edge_y**2 + 1e-8) 
             
            # 마스크 영역에서의 에지 차이 
            fill_mask = (1.0 - mask).expand_as(pred_edge) 
            edge_diff = torch.abs(pred_edge - gt_edge) 
             
            edge_loss = (edge_diff * fill_mask).sum() / (fill_mask.sum() + 1e-8)
            return edge_loss
        except Exception as e:
            print(f"[AFT_ERROR] 에지 보존 손실 계산 중 오류: {e}")
            return torch.tensor(0.0, device=pred_image.device)

    def compute_color_consistency_loss(self, pred_image, gt_image, mask): 
        """색상 일관성 손실 - 픽셀 간 색상 조화""" 
        if not hasattr(self, 'color_consistency_weight') or self.color_consistency_weight <= 0: 
            return torch.tensor(0.0, device=pred_image.device) 
         
        try:
            B, C, H, W = pred_image.shape 
            fill_mask = (1.0 - mask).expand_as(pred_image) 
             
            # 인접 픽셀 간 색상 차이 
            # 수평 방향 
            pred_h_diff = torch.abs(pred_image[:, :, :, 1:] - pred_image[:, :, :, :-1]) 
            gt_h_diff = torch.abs(gt_image[:, :, :, 1:] - gt_image[:, :, :, :-1]) 
            h_mask = fill_mask[:, :, :, 1:] * fill_mask[:, :, :, :-1] 
            h_loss = torch.abs(pred_h_diff - gt_h_diff) * h_mask 
            h_loss = h_loss.sum() / (h_mask.sum() + 1e-8) 
             
            # 수직 방향 
            pred_v_diff = torch.abs(pred_image[:, :, 1:, :] - pred_image[:, :, :-1, :]) 
            gt_v_diff = torch.abs(gt_image[:, :, 1:, :] - gt_image[:, :, :-1, :]) 
            v_mask = fill_mask[:, :, 1:, :] * fill_mask[:, :, :-1, :] 
            v_loss = torch.abs(pred_v_diff - gt_v_diff) * v_mask 
            v_loss = v_loss.sum() / (v_mask.sum() + 1e-8) 
             
            return (h_loss + v_loss) / 2.0
        except Exception as e:
            print(f"[AFT_ERROR] 색상 일관성 손실 계산 중 오류: {e}")
            return torch.tensor(0.0, device=pred_image.device)
    
    def compute_pixel_level_selection_consistency(self, selection_weights, mask):
        """픽셀 레벨 선택 일관성 - 더 세밀한 제약"""
        if selection_weights is None or self.selection_consistency_weight <= 0:
            return torch.tensor(0.0, device=mask.device)
        
        B, K, H, W = selection_weights.shape
        fill_mask = 1.0 - mask  # [B, 1, H, W]
        
        # 픽셀 레벨에서의 선택 부드러움
        total_loss = 0.0
        
        # 각 후보별로 선택 맵의 부드러움 평가
        for k in range(K):
            weight_map = selection_weights[:, k:k+1, :, :]  # [B, 1, H, W]
            masked_weights = weight_map * fill_mask
            
            # 수평 부드러움
            h_diff = torch.abs(masked_weights[:, :, :, 1:] - masked_weights[:, :, :, :-1])
            h_loss = h_diff.sum() / (h_diff.numel() + 1e-8)
            
            # 수직 부드러움
            v_diff = torch.abs(masked_weights[:, :, 1:, :] - masked_weights[:, :, :-1, :])
            v_loss = v_diff.sum() / (v_diff.numel() + 1e-8)
            
            total_loss += (h_loss + v_loss) / 2.0
        
        return total_loss / K
    
    def forward(self, predicted_pixels_logits, confidence_map_logits, ground_truth_image, original_mask, 
                model_info=None):
        """개선된 픽셀 레벨 손실 계산"""
        
        # 기본 손실들 계산
        loss_dict = super().forward(
            predicted_pixels_logits, confidence_map_logits, 
            ground_truth_image, original_mask, model_info
        )
        
        predicted_pixels_sigmoid = torch.sigmoid(predicted_pixels_logits)
        
        # 픽셀 레벨 전용 추가 손실들
        
        # 에지 보존 손실
        if self.edge_preservation_weight > 0:
            edge_loss = self.compute_edge_preservation_loss(
                predicted_pixels_sigmoid, ground_truth_image, original_mask
            )
            loss_dict['edge_preservation'] = self.edge_preservation_weight * edge_loss
            loss_dict['total_loss'] += loss_dict['edge_preservation']
        else:
            loss_dict['edge_preservation'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        
        # 색상 일관성 손실
        if self.color_consistency_weight > 0:
            color_loss = self.compute_color_consistency_loss(
                predicted_pixels_sigmoid, ground_truth_image, original_mask
            )
            loss_dict['color_consistency'] = self.color_consistency_weight * color_loss
            loss_dict['total_loss'] += loss_dict['color_consistency']
        else:
            loss_dict['color_consistency'] = torch.tensor(0.0, device=predicted_pixels_logits.device)
        
        # 픽셀 레벨 선택 일관성 (기존 selection_consistency 대체)
        if model_info and 'selection_weights' in model_info:
            pixel_selection_loss = self.compute_pixel_level_selection_consistency(
                model_info['selection_weights'], original_mask
            )
            
            # 기존 selection_consistency를 픽셀 레벨 버전으로 교체
            loss_dict['total_loss'] -= loss_dict['selection_consistency']
            loss_dict['selection_consistency'] = self.selection_consistency_weight * pixel_selection_loss
            loss_dict['total_loss'] += loss_dict['selection_consistency']
        
        return loss_dict


# 픽셀 레벨 최적화를 위한 추가 유틸리티 함수들

def create_pixel_level_aft_loss(**kwargs):
    """픽셀 레벨 선택에 최적화된 손실 함수 생성"""
    return EnhancedRegionBasedCSFLoss(**kwargs)

def create_adaptive_aft_loss(region_size=1, **kwargs):
    """영역 크기에 따라 적응적으로 손실 함수 생성"""
    if region_size <= 2:  # 픽셀 레벨에 가까운 경우
        return EnhancedRegionBasedCSFLoss(**kwargs)
    else:  # 일반적인 영역 레벨
        return RegionBasedCSFLoss(**kwargs)