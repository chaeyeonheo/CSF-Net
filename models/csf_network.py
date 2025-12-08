# models/csf_network.py
# Region-based Selection이 추가된 AFT-Net (기존 클래스명 유지)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .components.swin_transformer_modules import PatchEmbed, BasicLayer, PatchMerging
from .components.fusion_attention import FusionAttentionModule
from .components.pixel_fusion_module import PixelLevelFusionModule
from .components.region_based_selector import RegionBasedSelector  # 새로 추가
from timm.layers import trunc_normal_

# ============================================================
# ==================== DEBUGGING PRINTS ====================
# ============================================================
DEBUG_AFT_NETWORK = False 
def print_shape_aft(tensor, name, enabled=DEBUG_AFT_NETWORK, context="CSFNetwork"):
    if enabled and tensor is not None:
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            if tensor: 
                shapes = [item.shape if hasattr(item, 'shape') else type(item) for item in tensor]
                print(f"[DEBUG {context}] {name}: List of {len(tensor)} items, shapes: {shapes}")
            else:
                print(f"[DEBUG {context}] {name}: Empty list/tuple")
        elif hasattr(tensor, 'shape'):
            print(f"[DEBUG {context}] {name}: {tensor.shape}")
        else:
            print(f"[DEBUG {context}] {name}: Type: {type(tensor)}, Value: {str(tensor)[:100]}")

DEBUG_AFT_DECODER_STAGE = False
def print_shape_decoder(tensor, name, enabled=DEBUG_AFT_DECODER_STAGE):
    print_shape_aft(tensor, name, enabled, context="CSFDecoderStage")
# ============================================================
# ================= END DEBUGGING PRINTS ===================
# ============================================================

class CSFDecoderStage(nn.Module):
    def __init__(self, 
                 input_spatial_dim,
                 actual_skip_ctx_dim,
                 fusion_module_dim,
                 dec_out_dim, 
                 num_fusion_heads, Kmax, 
                 mlp_ratio=4., norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 drop_path_rate_fusion=0.):
        super().__init__()
        self.Kmax = Kmax
        self.fusion_module_dim = fusion_module_dim

        if input_spatial_dim != fusion_module_dim:
            self.proj_input_to_fusion_dim_seq = nn.Linear(input_spatial_dim, fusion_module_dim)
        else:
            self.proj_input_to_fusion_dim_seq = nn.Identity()

        if actual_skip_ctx_dim != fusion_module_dim:
            self.proj_skip_ctx_to_fusion_dim_seq = nn.Linear(actual_skip_ctx_dim, fusion_module_dim)
        else:
            self.proj_skip_ctx_to_fusion_dim_seq = nn.Identity()
        
        self.norm_before_fusion_ctx = norm_layer(fusion_module_dim)
        self.act_before_fusion_ctx = act_layer()

        self.fusion_module = FusionAttentionModule(
            dim=fusion_module_dim, num_heads=num_fusion_heads, Kmax=Kmax,
            mlp_ratio=mlp_ratio, norm_layer=norm_layer, act_layer=act_layer,
            drop_path=drop_path_rate_fusion
        )
        
        self.final_conv = nn.Conv2d(fusion_module_dim, dec_out_dim, kernel_size=3, padding=1)
        self.norm_final = norm_layer(dec_out_dim) 
        self.act_final = act_layer()

    def _to_sequence(self, x_spatial):
        B, D, H, W = x_spatial.shape; return x_spatial.flatten(2).transpose(1,2).contiguous()

    def _to_spatial(self, x_seq, H, W):
        B, L, D = x_seq.shape; return x_seq.transpose(1,2).contiguous().view(B, D, H, W)

    def forward(self, x_input_spatial, 
                skip_ctx_seq, skip_cand_kmax_seq, 
                target_H_patch, target_W_patch): 
        
        print_shape_decoder(x_input_spatial, "Input x_input_spatial to CSFDecoderStage (should be at target_res, input_spatial_dim)")
        print_shape_decoder(skip_ctx_seq, f"Input skip_ctx_seq to CSFDecoderStage (L={target_H_patch*target_W_patch})")
        
        B, C_in_spatial, H_in_spatial, W_in_spatial = x_input_spatial.shape
        assert H_in_spatial == target_H_patch and W_in_spatial == target_W_patch, \
            f"CSFDecoderStage Input spatial resolution mismatch: expect ({target_H_patch},{target_W_patch}), got ({H_in_spatial},{W_in_spatial})"
        
        x_input_seq = self._to_sequence(x_input_spatial) 
        x_input_projected_seq = self.proj_input_to_fusion_dim_seq(x_input_seq)
        print_shape_decoder(x_input_projected_seq, "AFTDS x_input_projected_seq for fusion context")

        skip_ctx_projected_seq = self.proj_skip_ctx_to_fusion_dim_seq(skip_ctx_seq)
        print_shape_decoder(skip_ctx_projected_seq, "AFTDS skip_ctx_projected_seq for fusion context")
        
        assert x_input_projected_seq.shape[1] == skip_ctx_projected_seq.shape[1], \
            f"AFTDS Seq length mismatch for context combine: x_input L={x_input_projected_seq.shape[1]}, skip_ctx L={skip_ctx_projected_seq.shape[1]}"
        assert x_input_projected_seq.shape[-1] == self.fusion_module_dim, f"AFTDS x_input_projected_seq.dim ({x_input_projected_seq.shape[-1]}) != fusion_dim ({self.fusion_module_dim})"
        assert skip_ctx_projected_seq.shape[-1] == self.fusion_module_dim, f"AFTDS skip_ctx_projected_seq.dim ({skip_ctx_projected_seq.shape[-1]}) != fusion_dim ({self.fusion_module_dim})"

        context_for_fusion_seq = x_input_projected_seq + skip_ctx_projected_seq
        if isinstance(self.norm_before_fusion_ctx, nn.LayerNorm):
            context_for_fusion_seq = self.norm_before_fusion_ctx(context_for_fusion_seq)
        context_for_fusion_seq = self.act_before_fusion_ctx(context_for_fusion_seq)
        print_shape_decoder(context_for_fusion_seq, "AFTDS context_for_fusion_seq (combined, normed, acted)")

        assert skip_cand_kmax_seq.shape[2] == (target_H_patch * target_W_patch), \
            f"AFTDS Skip Cand Seq Length L={skip_cand_kmax_seq.shape[2]} != target L={target_H_patch*target_W_patch}"
        if skip_cand_kmax_seq.shape[-1] != self.fusion_module_dim:
            projection_layer_name = f"proj_skip_cand_to_fusion_dim_{skip_cand_kmax_seq.shape[-1]}"
            if not hasattr(self, projection_layer_name):
                setattr(self, projection_layer_name, 
                        nn.Linear(skip_cand_kmax_seq.shape[-1], self.fusion_module_dim).to(skip_cand_kmax_seq.device))
            proj_layer = getattr(self, projection_layer_name)
            B_sc, K_sc, L_sc, D_sc = skip_cand_kmax_seq.shape
            skip_cand_kmax_seq_projected = proj_layer(skip_cand_kmax_seq.reshape(B_sc*K_sc*L_sc, D_sc)).view(B_sc, K_sc, L_sc, self.fusion_module_dim)
        else:
            skip_cand_kmax_seq_projected = skip_cand_kmax_seq
        print_shape_decoder(skip_cand_kmax_seq_projected, "AFTDS skip_cand_kmax_seq_projected for fusion")
            
        fused_features_seq = self.fusion_module(context_for_fusion_seq, skip_cand_kmax_seq_projected)
        print_shape_decoder(fused_features_seq, "AFTDS fused_features_seq (output of fusion_module)")
        
        fused_features_spatial = self._to_spatial(fused_features_seq, target_H_patch, target_W_patch)
        output_spatial = self.final_conv(fused_features_spatial)
        
        B_out, D_out, H_out, W_out = output_spatial.shape
        output_seq = self._to_sequence(output_spatial)
        if isinstance(self.norm_final, nn.LayerNorm): output_seq = self.norm_final(output_seq)
        output_seq = self.act_final(output_seq)
        
        final_output_spatial = self._to_spatial(output_seq, H_out, W_out)
        print_shape_decoder(final_output_spatial, "AFTDS final_output_spatial of stage")
        return final_output_spatial


class CSFNetwork(nn.Module):
    def __init__(self, config_all):
        super().__init__()
        self.config_model = config_all.model
        self.config_data = config_all.data
        self.Kmax = self.config_data.Kmax
        self.norm_layer = getattr(nn, self.config_model.norm_layer)
        act_layer_name = getattr(self.config_model, 'act_layer', 'GELU')
        self.act_layer = getattr(nn, act_layer_name)
        self.num_encoder_stages = len(self.config_model.context_depths)

        # ===== 융합 방식 결정 (새로 추가) =====
        self.fusion_method = getattr(self.config_model, 'fusion_method', 'pixel_fusion')
        print(f"AFT-Net 융합 방식: {self.fusion_method}")

        # 기존 픽셀 융합 관련 설정 (호환성 유지)
        self.use_pixel_fusion = getattr(self.config_model, 'use_pixel_fusion', True)
        self.pixel_fusion_hidden_dim = getattr(self.config_model, 'pixel_fusion_hidden_dim', 128)

        print_shape_aft(None, f"--- Initializing CSFNetwork (act_layer: {self.act_layer.__name__}, fusion_method: {self.fusion_method}) ---")

        # ===== 수정 후 코드 =====
        # 1. Context 인코더 정의
        # 1-1. PatchEmbed 정의 (이미지용과 마스크용으로 분리)
        ctx_embed_dim = self.config_model.context_embed_dim  # 예: 96

        self.context_image_patch_embed = PatchEmbed(
            img_size=self.config_data.img_size,
            patch_size=self.config_model.patch_size,
            in_chans=3,  # 이미지 채널 수
            embed_dim=ctx_embed_dim,
            norm_layer=self.norm_layer if self.config_model.patch_norm else None
        )

        self.context_mask_patch_embed = PatchEmbed(
            img_size=self.config_data.img_size,
            patch_size=self.config_model.patch_size,
            in_chans=1,  # 마스크 채널 수
            embed_dim=ctx_embed_dim,
            norm_layer=self.norm_layer if self.config_model.patch_norm else None
        )

        # 두 임베딩을 합친 후의 차원을 맞추기 위한 프로젝션 레이어 (선택적이지만 권장)
        self.context_fusion_proj = nn.Linear(ctx_embed_dim * 2, ctx_embed_dim)

        # self.context_patch_embed 대신 분리된 embedder의 해상도를 사용
        self.context_pos_drop = nn.Dropout(p=self.config_model.drop_rate)
        self.context_encoder_layers = nn.ModuleList()
        self.ctx_encoder_stage_output_dims = []
        self.ctx_encoder_stage_output_resolutions = []
        current_dim_ctx = self.config_model.context_embed_dim
        # 해상도는 이미지/마스크 임베더가 동일하므로 하나만 사용
        current_res_ctx = self.context_image_patch_embed.patches_resolution 
        # dpr_ctx: 각 Swin Transformer 블록에 적용될 Stochastic Depth의 드롭 확률 리스트
        dpr_ctx = [x.item() for x in torch.linspace(0, self.config_model.drop_path_rate, sum(self.config_model.context_depths))]
        # sum(self.config.model.context_depths) = 2+2+6+2 = 12
        # drop_path_rate는 config 값 (예: 0.1)
        for i_layer in range(self.num_encoder_stages):
            is_last_enc_stage = (i_layer == self.num_encoder_stages - 1)
            layer = BasicLayer( # 현재 스테이지의 SwinT 블록
                dim=current_dim_ctx,                      # 현재 스테이지 입력 차원
                input_resolution=current_res_ctx,         # 현재 스테이지 입력 해상도 (패치 수 기준)
                depth=self.config_model.context_depths[i_layer],        # 현재 스테이지의 SwinT 블록 수 (2, 2, 6, 2 순차적용)
                num_heads=self.config_model.context_num_heads[i_layer], # 현재 스테이지의 어텐션 헤드 수 (3, 6, 12, 24 순차적용)
                window_size=self.config_model.window_size, # (예: 7)
                mlp_ratio=self.config_model.mlp_ratio,     # (예: 4.0)
                qkv_bias=self.config_model.qkv_bias,       # (예: True)
                drop=self.config_model.drop_rate,          # (예: 0.0)
                attn_drop=self.config_model.attn_drop_rate, # (예: 0.0)
                drop_path=dpr_ctx[sum(self.config_model.context_depths[:i_layer]):sum(self.config_model.context_depths[:i_layer + 1])], # 현재 스테이지 블록들의 drop path 확률
                norm_layer=self.norm_layer,               # nn.LayerNorm
                act_layer=self.act_layer,                 # nn.GELU (가정)
                downsample=PatchMerging if not is_last_enc_stage else None, # 마지막 스테이지 아니면 PatchMerging으로 다운샘플링
                # 인접한 2x2 패치들을 그룹화하고, 이를 새로운 하나의 패치로 만듭니다. 이 과정에서 차원은 4배가 되고(2x2=4), 이를 선형 계층을 통해 2배로 줄여 최종적으로 채널 수가 2배가 됩니다. 해상도는 가로, 세로 각각 절반이 됩니다
                use_checkpoint=self.config_model.use_checkpoint # (예: False)
            )
            self.context_encoder_layers.append(layer)
            # 현재 스테이지를 거친 후의 차원과 해상도 계산
            # 마지막 스테이지가 아니고, 다운샘플링이 필요한 경우(PatchMerging) 차원은 2배가 되고, 해상도는 절반이 됩니다. 
            # 마지막 스테이지이거나 다운샘플링이 필요하지 않은 경우(PatchMerging이 None) 차원과 해상도는 변하지 않습니다.
            dim_after_stage = current_dim_ctx * 2 if not is_last_enc_stage and layer.downsample is not None else current_dim_ctx
            # 마지막 스테이지가 아니고, 다운샘플링이 필요한 경우(PatchMerging) 해상도는 절반이 됩니다. 
            # 마지막 스테이지이거나 다운샘플링이 필요하지 않은 경우(PatchMerging이 None) 해상도는 변하지 않습니다.
            res_after_stage = (current_res_ctx[0] // 2, current_res_ctx[1] // 2) if not is_last_enc_stage and layer.downsample is not None else current_res_ctx
            # ctx_encoder_stage_output_dims와 ctx_encoder_stage_output_resolutions 리스트에 저장하여 나중에 디코더의 skip connection에서 사용합니다.
            # 다음 스테이지의 입력으로 사용하기 위해 현재 차원과 해상도 업데이트
            self.ctx_encoder_stage_output_dims.append(dim_after_stage)
            self.ctx_encoder_stage_output_resolutions.append(res_after_stage)
            # 다음 스테이지를 위해 현재 차원과 해상도 업데이트
            current_dim_ctx = dim_after_stage
            current_res_ctx = res_after_stage
        self.norm_ctx_encoder = self.norm_layer(current_dim_ctx)
        self.encoder_bottleneck_dim = current_dim_ctx 

        # 2. Candidate 인코더 정의
        self.candidate_patch_embed = PatchEmbed(
            img_size=self.config_data.img_size,    # 256
            patch_size=self.config_model.patch_size, # 4
            in_chans=self.config_model.candidate_in_channels, # 3 (후보 이미지는 RGB)
            embed_dim=self.config_model.candidate_embed_dim,  # 96
            norm_layer=self.norm_layer if self.config_model.patch_norm else None
        )
        self.candidate_pos_drop = nn.Dropout(p=self.config_model.drop_rate)
        self.candidate_encoder_layers = nn.ModuleList()
        self.cand_encoder_stage_output_dims = []
        self.cand_encoder_stage_output_resolutions = []
        current_dim_cand = self.config_model.candidate_embed_dim      # 초기 차원: 96
        current_res_cand = self.candidate_patch_embed.patches_resolution # 초기 해상도: (64, 64)
        # dpr_cand: Candidate 인코더용 drop path 확률 리스트
        dpr_cand = [x.item() for x in torch.linspace(0, self.config_model.drop_path_rate, sum(self.config_model.candidate_depths))]
        # sum(self.config.model.candidate_depths) = 2+2+6+2 = 12

        for i_layer in range(self.num_encoder_stages): # self.num_encoder_stages = 4
            is_last_enc_stage_cand = (i_layer == self.num_encoder_stages - 1) # 현재가 마지막 스테이지인지
            layer = BasicLayer( # 현재 스테이지의 SwinT 블록
                dim=current_dim_cand, # 96
                input_resolution=current_res_cand, # (64, 64)
                depth=self.config_model.candidate_depths[i_layer],         # Candidate용 depths 사용 (2,2,6,2)
                num_heads=self.config_model.candidate_num_heads[i_layer],  # Candidate용 num_heads 사용 (3,6,12,24)
                window_size=self.config_model.window_size, # 7
                mlp_ratio=self.config_model.mlp_ratio, # 4.0
                qkv_bias=self.config_model.qkv_bias, # True
                drop=self.config_model.drop_rate, # 0.0
                attn_drop=self.config_model.attn_drop_rate, # 0.0
                drop_path=dpr_cand[sum(self.config_model.candidate_depths[:i_layer]):sum(self.config_model.candidate_depths[:i_layer + 1])],
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
                downsample=PatchMerging if not is_last_enc_stage_cand else None, # 마지막 스테이지 아니면 다운샘플링
                use_checkpoint=self.config_model.use_checkpoint
            )
            self.candidate_encoder_layers.append(layer)
            # 현재 스테이지를 거친 후의 차원과 해상도 계산
            dim_after_stage_cand = current_dim_cand * 2 if not is_last_enc_stage_cand and layer.downsample is not None else current_dim_cand
            res_after_stage_cand = (current_res_cand[0] // 2, current_res_cand[1] // 2) if not is_last_enc_stage_cand and layer.downsample is not None else current_res_cand
            self.cand_encoder_stage_output_dims.append(dim_after_stage_cand)
            self.cand_encoder_stage_output_resolutions.append(res_after_stage_cand)
            # 다음 스테이지를 위해 현재 차원과 해상도 업데이트
            current_dim_cand = dim_after_stage_cand
            current_res_cand = res_after_stage_cand
        self.norm_cand_encoder = self.norm_layer(current_dim_cand) # self.norm_layer(768)
        assert self.encoder_bottleneck_dim == current_dim_cand # Context 인코더의 최종 차원과 Candidate 인코더의 최종 차원이 같은지 확인 (768 == 768)

        # 3. 디코더 업샘플러 및 퓨전 스테이지 정의
        self.decoder_upsamplers = nn.ModuleList()
        self.decoder_fusion_stages = nn.ModuleList()
        current_dec_input_channels_for_upsampler = self.encoder_bottleneck_dim # <--- 이 위치로 이동 또는 확인!
        # 디코더가 목표로 하는 해상도들 (인코더 출력 해상도의 역순)
        # self.ctx_encoder_stage_output_resolutions = [(32,32), (16,16), (8,8), (8,8)]
        # reversed -> [(8,8), (8,8), (16,16), (32,32)](패치 수 기준 해상도)
        self.decoder_target_operating_resolutions = list(reversed(self.ctx_encoder_stage_output_resolutions))

        # 디코더 각 스테이지에서 skip connection으로 받을 Context 특징의 차원들 (인코더 출력 차원의 역순)
        # self.ctx_encoder_stage_output_dims = [192, 384, 768, 768]
        # reversed -> [768, 768, 384, 192] Context 인코더의 각 스테이지 출력 차원을 역순으로 배열
        actual_skip_ctx_dims_rev = list(reversed(self.ctx_encoder_stage_output_dims))
        for i in range(self.num_encoder_stages): # self.num_encoder_stages = 4. i는 0, 1, 2, 3 순으로 반복
            # 현재 디코더 스테이지(i)가 목표로 하는 패치 해상도 (H_patch, W_patch)
            # 
            target_res_H_current_stage, target_res_W_current_stage = self.decoder_target_operating_resolutions[i]
            # 현재 디코더 스테이지(i)에서 skip connection으로 받을 Context 특징의 실제 차원
            actual_skip_ctx_dim_for_stage = actual_skip_ctx_dims_rev[i]
            # 현재 디코더 스테이지의 FusionAttentionModule이 사용할 내부 연산 차원 (skip connection 차원과 동일하게 설정)
            current_stage_fusion_module_dim = actual_skip_ctx_dim_for_stage

            # 현재 디코더 스테이지의 최종 출력 차원 결정
            if i < self.num_encoder_stages - 1: # 마지막 스테이지가 아니면
                # 다음 스테이지의 skip connection 차원을 현재 스테이지의 출력 차원으로 사용
                current_stage_dec_out_dim = actual_skip_ctx_dims_rev[i+1]
            else: # 마지막 디코더 스테이지(i=3)이면
                # config에 정의된 decoder_base_dim (예: 96)을 사용
                current_stage_dec_out_dim = self.config_model.decoder_base_dim
            
            # 업샘플러/프로젝터 로직 
            if i == 0: # 첫 번째 디코더 스테이지 (가장 깊은 곳에서 시작)
                # 입력 채널(current_dec_input_channels_for_upsampler=768)과
                # 현재 스테이지 퓨전 모듈 차원(current_stage_fusion_module_dim=768)이 다르면 Conv2d로 차원 맞춤
                if current_dec_input_channels_for_upsampler != current_stage_fusion_module_dim:
                    upsampler = nn.Conv2d(current_dec_input_channels_for_upsampler, current_stage_fusion_module_dim, kernel_size=1)
                else: # 같으면 아무것도 안함 (Identity)
                    upsampler = nn.Identity() # 현재 설정에서는 768 == 768 이므로 Identity
            else: # 두 번째 디코더 스테이지부터
                # 이전 스테이지의 출력 해상도
                prev_stage_output_res_H, prev_stage_output_res_W = self.decoder_target_operating_resolutions[i-1]
                # 현재 스테이지 목표 해상도가 이전 스테이지 출력 해상도보다 크면 업샘플링 필요
                if target_res_H_current_stage > prev_stage_output_res_H:
                    scale_factor = target_res_H_current_stage // prev_stage_output_res_H
                    # 정확히 2배 업샘플링이고, 채널 수도 맞춰야 하면 ConvTranspose2d 사용
                    if scale_factor == 2 and target_res_H_current_stage % prev_stage_output_res_H == 0:
                         upsampler = nn.ConvTranspose2d(current_dec_input_channels_for_upsampler, current_stage_fusion_module_dim, kernel_size=2, stride=2)
                    else: # 그 외의 경우 (일반적이지 않음) Upsample + Conv1x1
                         upsampler = nn.Sequential(
                             nn.Upsample(size=(target_res_H_current_stage, target_res_W_current_stage), mode='bilinear', align_corners=False),
                             nn.Conv2d(current_dec_input_channels_for_upsampler, current_stage_fusion_module_dim, kernel_size=1)
                         )
                else: # 해상도 변경이 없거나 줄어드는 경우 (일반적인 U-Net 디코더에서는 잘 없음, 여기서는 차원만 맞춤)
                    if current_dec_input_channels_for_upsampler != current_stage_fusion_module_dim:
                        upsampler = nn.Conv2d(current_dec_input_channels_for_upsampler, current_stage_fusion_module_dim, kernel_size=1)
                    else:
                        upsampler = nn.Identity()
            self.decoder_upsamplers.append(upsampler)

            # 현재 디코더 스테이지의 FusionAttentionModule에 사용될 어텐션 헤드 수
            fusion_heads = self.config_model.decoder_fusion_num_heads[i] # [24, 12, 6, 3] 순차 적용
            dpr_fusion = self.config_model.drop_path_rate # (예: 0.1)

            self.decoder_fusion_stages.append(
                CSFDecoderStage( # CSFDecoderStage는 내부에 FusionAttentionModule을 가짐
                    input_spatial_dim=current_stage_fusion_module_dim, # 업샘플러를 거친 입력의 차원
                    actual_skip_ctx_dim=actual_skip_ctx_dim_for_stage, # Context skip connection의 차원
                    fusion_module_dim=current_stage_fusion_module_dim, # FusionAttentionModule 내부 연산 차원
                    dec_out_dim=current_stage_dec_out_dim,           # 현재 스테이지의 최종 출력 차원
                    num_fusion_heads=fusion_heads, Kmax=self.Kmax,   # 어텐션 헤드 수, 후보 수
                    mlp_ratio=self.config_model.mlp_ratio, norm_layer=self.norm_layer,
                    act_layer=self.act_layer, drop_path_rate_fusion=dpr_fusion
                )
            )
            # 다음 디코더 스테이지의 업샘플러 입력 채널을 현재 스테이지의 출력 차원으로 업데이트
            current_dec_input_channels_for_upsampler = current_stage_dec_out_dim
            
        self.final_features_norm = self.norm_layer(current_dec_input_channels_for_upsampler) if self.config_model.final_norm_output else nn.Identity()
        
        # ===== 4. 융합 모듈 초기화 (수정된 부분) =====
        self._initialize_fusion_modules(current_dec_input_channels_for_upsampler)
        
        # 5. 최종 업샘플링 레이어 (기존과 동일)
        self._initialize_final_upsampling()
        
        self.final_target_img_size = (self.config_data.img_size, self.config_data.img_size)
        self._init_weights()


            # 🚨 핵심 추가: PatchEmbed 긴급 안정화 (이 두 줄 추가!)
        self._emergency_patch_embed_stabilization()
        if hasattr(self, 'context_image_patch_embed'):
            self._emergency_separated_patch_embed_stabilization()



# models/csf_network.py의 CSFNetwork 클래스 내부에 새로운 메서드로 추가 (forward 메서드 위)



    def _emergency_patch_embed_stabilization(self):
        """
        🎯 역할: PatchEmbed 레이어의 weight/bias 긴급 안정화 (분리된 임베더 지원)
        📍 위치: CSFNetwork._emergency_patch_embed_stabilization 메서드를 이것으로 완전 교체
        
        🚨 핵심: context_patch_embed.proj.bias의 초기값과 범위를 강제로 안정화
        """
        print("🔧 [INIT] Emergency PatchEmbed stabilization starting...")
        
        with torch.no_grad():
            # 1. 분리된 임베더들 안정화 (현재 코드 구조)
            if hasattr(self, 'context_image_patch_embed'):
                print("🔧 [INIT] Found separated embedders - stabilizing...")
                
                # context_image_patch_embed 안정화
                if hasattr(self.context_image_patch_embed, 'proj'):
                    proj_layer = self.context_image_patch_embed.proj
                    
                    if hasattr(proj_layer, 'weight'):
                        fan_in = proj_layer.weight.size(1) * proj_layer.weight.size(2) * proj_layer.weight.size(3)
                        std = (1.0 / fan_in) ** 0.5
                        proj_layer.weight.data.uniform_(-std, std)
                        proj_layer.weight.data = torch.clamp(proj_layer.weight.data, -0.1, 0.1)
                        print(f"🔧 [INIT] context_image_patch_embed.proj.weight stabilized")
                    
                    if hasattr(proj_layer, 'bias') and proj_layer.bias is not None:
                        proj_layer.bias.data.zero_()
                        proj_layer.bias.data += torch.randn_like(proj_layer.bias.data) * 0.001
                        print(f"🔧 [INIT] context_image_patch_embed.proj.bias stabilized")
                
                # context_mask_patch_embed 안정화
                if hasattr(self, 'context_mask_patch_embed') and hasattr(self.context_mask_patch_embed, 'proj'):
                    proj_layer = self.context_mask_patch_embed.proj
                    
                    if hasattr(proj_layer, 'weight'):
                        fan_in = proj_layer.weight.size(1) * proj_layer.weight.size(2) * proj_layer.weight.size(3)
                        std = (1.0 / fan_in) ** 0.5
                        proj_layer.weight.data.uniform_(-std, std)
                        proj_layer.weight.data = torch.clamp(proj_layer.weight.data, -0.1, 0.1)
                        print(f"🔧 [INIT] context_mask_patch_embed.proj.weight stabilized")
                    
                    if hasattr(proj_layer, 'bias') and proj_layer.bias is not None:
                        proj_layer.bias.data.zero_()
                        proj_layer.bias.data += torch.randn_like(proj_layer.bias.data) * 0.001
                        print(f"🔧 [INIT] context_mask_patch_embed.proj.bias stabilized")
            
            # 2. 통합 임베더 안정화 (호환성 지원)
            elif hasattr(self, 'context_patch_embed'):
                print("🔧 [INIT] Found unified embedder - stabilizing...")
                
                if hasattr(self.context_patch_embed, 'proj'):
                    proj_layer = self.context_patch_embed.proj
                    
                    if hasattr(proj_layer, 'weight'):
                        fan_in = proj_layer.weight.size(1) * proj_layer.weight.size(2) * proj_layer.weight.size(3)
                        std = (1.0 / fan_in) ** 0.5
                        proj_layer.weight.data.uniform_(-std, std)
                        proj_layer.weight.data = torch.clamp(proj_layer.weight.data, -0.1, 0.1)
                        print(f"🔧 [INIT] context_patch_embed.proj.weight stabilized")
                    
                    if hasattr(proj_layer, 'bias') and proj_layer.bias is not None:
                        proj_layer.bias.data.zero_()
                        proj_layer.bias.data += torch.randn_like(proj_layer.bias.data) * 0.001
                        print(f"🔧 [INIT] context_patch_embed.proj.bias stabilized")
            else:
                print("⚠️ [INIT] No context patch embed found - skipping stabilization")
            
            # 3. candidate_patch_embed 안정화 (항상 실행)
            if hasattr(self, 'candidate_patch_embed') and hasattr(self.candidate_patch_embed, 'proj'):
                proj_layer = self.candidate_patch_embed.proj
                
                if hasattr(proj_layer, 'weight'):
                    fan_in = proj_layer.weight.size(1) * proj_layer.weight.size(2) * proj_layer.weight.size(3)
                    std = (1.0 / fan_in) ** 0.5
                    proj_layer.weight.data.uniform_(-std, std)
                    proj_layer.weight.data = torch.clamp(proj_layer.weight.data, -0.1, 0.1)
                    print(f"🔧 [INIT] candidate_patch_embed.proj.weight stabilized")
                
                if hasattr(proj_layer, 'bias') and proj_layer.bias is not None:
                    proj_layer.bias.data.zero_()
                    proj_layer.bias.data += torch.randn_like(proj_layer.bias.data) * 0.001
                    print(f"🔧 [INIT] candidate_patch_embed.proj.bias stabilized")
            
            # 4. context_fusion_proj 안정화 (분리된 임베더 사용 시)
            if hasattr(self, 'context_fusion_proj'):
                if hasattr(self.context_fusion_proj, 'weight'):
                    std = (1.0 / self.context_fusion_proj.weight.size(1)) ** 0.5
                    self.context_fusion_proj.weight.data.uniform_(-std, std)
                    self.context_fusion_proj.weight.data = torch.clamp(self.context_fusion_proj.weight.data, -0.1, 0.1)
                    print(f"🔧 [INIT] context_fusion_proj.weight stabilized")
                
                if hasattr(self.context_fusion_proj, 'bias') and self.context_fusion_proj.bias is not None:
                    self.context_fusion_proj.bias.data.zero_()
                    self.context_fusion_proj.bias.data += torch.randn_like(self.context_fusion_proj.bias.data) * 0.001
                    print(f"🔧 [INIT] context_fusion_proj.bias stabilized")
        
        print("✅ [INIT] Emergency PatchEmbed stabilization completed successfully!")

    def _emergency_separated_patch_embed_stabilization(self):
        """
        🎯 역할: 분리된 이미지/마스크 PatchEmbed 레이어들 안정화
        📍 위치: CSFNetwork 클래스에 새로 추가할 메서드
        
        🚨 핵심: context_image_patch_embed, context_mask_patch_embed bias 안정화
        """
        with torch.no_grad():
            # context_image_patch_embed 안정화
            if hasattr(self, 'context_image_patch_embed') and hasattr(self.context_image_patch_embed, 'proj'):
                proj_layer = self.context_image_patch_embed.proj
                
                if hasattr(proj_layer, 'weight'):
                    fan_in = proj_layer.weight.size(1) * proj_layer.weight.size(2) * proj_layer.weight.size(3)
                    std = (1.0 / fan_in) ** 0.5
                    proj_layer.weight.data.uniform_(-std, std)
                    proj_layer.weight.data = torch.clamp(proj_layer.weight.data, -0.1, 0.1)
                
                if hasattr(proj_layer, 'bias') and proj_layer.bias is not None:
                    proj_layer.bias.data.zero_()
                    proj_layer.bias.data += torch.randn_like(proj_layer.bias.data) * 0.001
                    print(f"🔧 [INIT] context_image_patch_embed.proj.bias stabilized")
            
            # context_mask_patch_embed 안정화
            if hasattr(self, 'context_mask_patch_embed') and hasattr(self.context_mask_patch_embed, 'proj'):
                proj_layer = self.context_mask_patch_embed.proj
                
                if hasattr(proj_layer, 'weight'):
                    fan_in = proj_layer.weight.size(1) * proj_layer.weight.size(2) * proj_layer.weight.size(3)
                    std = (1.0 / fan_in) ** 0.5
                    proj_layer.weight.data.uniform_(-std, std)
                    proj_layer.weight.data = torch.clamp(proj_layer.weight.data, -0.1, 0.1)
                
                if hasattr(proj_layer, 'bias') and proj_layer.bias is not None:
                    proj_layer.bias.data.zero_()
                    proj_layer.bias.data += torch.randn_like(proj_layer.bias.data) * 0.001
                    print(f"🔧 [INIT] context_mask_patch_embed.proj.bias stabilized")
            
            print("✅ [INIT] Separated patch embed stabilization completed")

    def _initialize_fusion_modules(self, feature_dim):
        """융합 방식에 따른 모듈 초기화"""
        
        if self.fusion_method == "pixel_fusion":
            # 기존 픽셀 융합 방식
            print("픽셀 융합 모듈 초기화...")
            self.pixel_fusion_module = PixelLevelFusionModule(
                num_candidates=self.Kmax,
                feature_dim=feature_dim,
                hidden_dim=self.pixel_fusion_hidden_dim,
                fusion_strategy=getattr(self.config_model, 'pixel_fusion_strategy', 'adaptive')
            )
            self.region_based_selector = None
            
            # 픽셀 융합용 최종 예측 헤드
            enhanced_pred_in_channels = feature_dim + 3 + 1  # 기존특징 + 융합픽셀 + 신뢰도
            self.final_pred_head = nn.Sequential(
                nn.Conv2d(enhanced_pred_in_channels, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim, 3 + 1, 3, padding=1)  # RGB + confidence
            )
            
        elif self.fusion_method == "region_based":
            # 새로운 Region-based Selection 방식 (픽셀 레벨 지원)
            print("Region-based Selection 모듈 초기화 (픽셀 레벨 지원)...")
            region_config = getattr(self.config_model, 'region_based_selector', None)
            
            # ConfigNamespace 객체에서 안전하게 값 추출
            if region_config is not None:
                hidden_dim = getattr(region_config, 'hidden_dim', 128)
                region_scales = getattr(region_config, 'region_scales', [32, 16, 8, 4])
                selection_strategy = getattr(region_config, 'selection_strategy', 'hierarchical')
                final_region_size = getattr(region_config, 'final_region_size', 1)  # 1=픽셀 레벨
            else:
                hidden_dim = 128
                region_scales = [32, 16, 8, 4]  # 더 세밀한 스케일 추가
                selection_strategy = 'hierarchical'
                final_region_size = 1  # 기본값: 픽셀 레벨
            
            print(f"Region-based config: scales={region_scales}, final_size={final_region_size}")
            if final_region_size == 1:
                print("✅ 픽셀 레벨 선택 모드 활성화")
            
            self.region_based_selector = RegionBasedSelector(
                num_candidates=self.Kmax,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                region_scales=region_scales,
                selection_strategy=selection_strategy,
                final_region_size=final_region_size
            )
            self.pixel_fusion_module = None
            
            # Region-based용 최종 예측 헤드 (간단한 구조)
            self.final_pred_head = nn.Conv2d(feature_dim, 3 + 1, kernel_size=3, padding=1)
            
        elif self.fusion_method == "hybrid":
            # 하이브리드 방식 (둘 다 사용)
            print("하이브리드 융합 모듈 초기화...")
            
            # 픽셀 융합 모듈
            self.pixel_fusion_module = PixelLevelFusionModule(
                num_candidates=self.Kmax,
                feature_dim=feature_dim,
                hidden_dim=self.pixel_fusion_hidden_dim,
                fusion_strategy=getattr(self.config_model, 'pixel_fusion_strategy', 'adaptive')
            )
            
            # Region-based 모듈
            region_config = getattr(self.config_model, 'region_based_selector', None)
            if region_config is not None:
                hidden_dim = getattr(region_config, 'hidden_dim', 128)
                region_scales = getattr(region_config, 'region_scales', [32, 16, 8])
                selection_strategy = getattr(region_config, 'selection_strategy', 'hierarchical')
            else:
                hidden_dim = 128
                region_scales = [32, 16, 8]
                selection_strategy = 'hierarchical'
                
            self.region_based_selector = RegionBasedSelector(
                num_candidates=self.Kmax,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                region_scales=region_scales,
                selection_strategy=selection_strategy,
                final_region_size=final_region_size  # 추가
            )
            
            # 하이브리드 융합 가중치
            hybrid_config = getattr(self.config_model, 'hybrid_fusion', None)
            if hybrid_config is not None:
                self.region_weight = getattr(hybrid_config, 'region_weight', 0.7)
                self.pixel_weight = getattr(hybrid_config, 'pixel_weight', 0.3)
            else:
                self.region_weight = 0.7
                self.pixel_weight = 0.3
            
            # 하이브리드용 최종 예측 헤드
            self.final_pred_head = nn.Conv2d(feature_dim, 3 + 1, kernel_size=3, padding=1)
            
        else:
            # 기본값: 기존 방식 (호환성)
            print("기본 방식 (특징 기반 예측)")
            self.pixel_fusion_module = None
            self.region_based_selector = None
            self.final_pred_head = nn.Conv2d(feature_dim, 3 + 1, kernel_size=3, padding=1)

    def _initialize_final_upsampling(self):
        """최종 업샘플링 레이어 초기화 (기존과 동일)"""

        self.final_pixel_upsampler = nn.Sequential() # 최종 픽셀 업샘플링 레이어
        last_decoder_stage_output_res_H, last_decoder_stage_output_res_W = self.decoder_target_operating_resolutions[-1] # (8, 8)
        num_final_2x_upsamples = 0 # 최종 업샘플링 횟수
        if self.config_data.img_size > last_decoder_stage_output_res_H: # 256 > 8
            scale_needed = self.config_data.img_size / last_decoder_stage_output_res_H # 256 / 8 = 32
            if scale_needed > 1 and scale_needed.is_integer(): # 32 > 1 and 32.is_integer() -> True
                scale_needed_int = int(scale_needed) # 32
                if (scale_needed_int > 0) and (scale_needed_int & (scale_needed_int - 1) == 0): # 32 > 0 and 32 & (32 - 1) == 0 -> True
                    num_final_2x_upsamples = int(np.log2(scale_needed_int)) # 5
                else:
                    num_final_2x_upsamples = -1
            elif scale_needed > 1: # 32 > 1 -> False
                 num_final_2x_upsamples = -1
        
        upsample_channels_final = 3 + 1 # 4
        if num_final_2x_upsamples > 0: # 5 > 0 -> True
            current_upsample_C = upsample_channels_final # 4
            for k_up in range(num_final_2x_upsamples): # 0, 1, 2, 3, 4
                self.final_pixel_upsampler.add_module( # 최종 픽셀 업샘플링 레이어에 업샘플링 레이어 추가
                    f"final_upsample_{k_up}", # final_upsample_0, final_upsample_1, final_upsample_2, final_upsample_3, final_upsample_4
                    nn.ConvTranspose2d(current_upsample_C, current_upsample_C, kernel_size=2, stride=2) # 2x2 업샘플링
                )
                if k_up < num_final_2x_upsamples - 1:
                     self.final_pixel_upsampler.add_module(f"final_relu_{k_up}", self.act_layer())

    def _init_weights(self): 
        """가중치 초기화 (기존과 동일)"""
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm): 
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def _seq_to_spatial(self, x_seq, H_patch, W_patch):
        B_in, L, D_in = x_seq.shape
        if L != H_patch * W_patch: 
            raise AssertionError(f"L={L} != H*W={H_patch*W_patch} for _seq_to_spatial. Shape: {x_seq.shape}, Target: {H_patch}x{W_patch}")
        return x_seq.transpose(1,2).contiguous().view(B_in, D_in, H_patch, W_patch)

    def _kmax_seq_to_spatial(self, x_seq_kmax, H_patch, W_patch):
        B_in, K_in, L, D_in = x_seq_kmax.shape
        if L != H_patch * W_patch: 
            raise AssertionError(f"L={L} != H*W={H_patch*W_patch} for _kmax_seq_to_spatial. Shape: {x_seq_kmax.shape}, Target: {H_patch}x{W_patch}")
        return x_seq_kmax.permute(0,1,3,2).contiguous().view(B_in, K_in, D_in, H_patch, W_patch)

    def _resize_candidates_to_resolution(self, candidate_images_kmax, target_h, target_w):
        """후보 이미지들을 특정 해상도로 리사이즈"""
        B, K, C, H, W = candidate_images_kmax.shape
        if H == target_h and W == target_w:
            return candidate_images_kmax
            
        # 각 후보를 개별적으로 리사이즈
        resized_candidates = []
        for k in range(K):
            resized_cand = F.interpolate(
                candidate_images_kmax[:, k], 
                size=(target_h, target_w), 
                mode='bilinear', 
                align_corners=False
            )
            resized_candidates.append(resized_cand.unsqueeze(1))
        
        return torch.cat(resized_candidates, dim=1)  # [B, K, C, target_h, target_w]
    
    def _smooth_mask_for_context(self, mask):
        """
        🎯 역할: 이진 마스크(0과 1)의 급격한 경계를 부드럽게 만들어 초기 컨볼루션 레이어의 안정성을 높입니다.
        이는 그래디언트 폭주를 막는 매우 효과적인 방법입니다.
        """
        # 간단한 평균 필터(Average Pooling)를 사용하여 경계를 부드럽게 합니다.
        # kernel_size=3, stride=1, padding=1은 이미지 크기를 유지하면서 주변 픽셀과 값을 섞는 효과를 줍니다.
        smoothed_mask = F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1)
        
        # 원래 마스크와 부드러워진 마스크를 약간 섞어 원래 형태를 너무 많이 잃지 않도록 합니다.
        return torch.clamp(mask * 0.5 + smoothed_mask * 0.5, 0.0, 1.0)





    def forward(self, partial_image, original_mask, candidate_images_kmax):
        B, K, C_cand, H_img, W_img = candidate_images_kmax.shape
        
        # ======================================================================
        # 🎯 1. 입력 안정화 (그래디언트 폭주 방지)
        # 모든 입력 이미지와 마스크를 [0, 1] 범위로 클램핑하여 안정성을 확보합니다.
        partial_image_stable = torch.clamp(partial_image, 0.0, 1.0)
        original_mask_stable = torch.clamp(original_mask, 0.0, 1.0)
        
        # 🎯 2. 마스크 스무딩 (핵심 안정화 기법)
        # 0과 1의 날카로운 경계를 가진 마스크를 부드럽게 변환하여 초기 레이어의 그래디언트 폭주를 방지합니다.
        mask_smoothed = self._smooth_mask_for_context(original_mask_stable)
        # ======================================================================

        # Context 인코더 입력 생성
        if hasattr(self, 'context_image_patch_embed'):
            # 분리된 임베더 사용 시, 안정화된 입력을 각각 전달합니다.
            image_patches_seq = self.context_image_patch_embed(partial_image_stable)
            mask_patches_seq = self.context_mask_patch_embed(mask_smoothed)
            
            combined_patches_seq = torch.cat([image_patches_seq, mask_patches_seq], dim=-1)
            ctx_patches_seq = self.context_fusion_proj(combined_patches_seq)
        else:
            # 통합 임베더 사용 시 (호환성), 안정화된 입력들을 결합하여 전달합니다.
            context_input = torch.cat([partial_image_stable, mask_smoothed], dim=1)
            ctx_patches_seq = self.context_patch_embed(context_input)

        ctx_patches_seq = self.context_pos_drop(ctx_patches_seq)
        
        ctx_skip_features_seq_list = [] 
        current_ctx_seq = ctx_patches_seq
        for i, layer_module in enumerate(self.context_encoder_layers):
            current_ctx_seq = layer_module(current_ctx_seq)
            ctx_skip_features_seq_list.append(current_ctx_seq)
            # print_shape_aft(current_ctx_seq, f"ctx_skip_features_seq_list[{i}] (Actual Output of Enc Stage {i})")
        
        # 2. Candidate 인코딩 (기존과 동일)
        candidate_images_flat = candidate_images_kmax.reshape(B * K, C_cand, H_img, W_img)
        cand_patches_seq_flat = self.candidate_patch_embed(candidate_images_flat)
        cand_patches_seq_flat = self.candidate_pos_drop(cand_patches_seq_flat)
        cand_skip_features_seq_kmax_list = []
        current_cand_seq_flat = cand_patches_seq_flat
        for i, layer_module in enumerate(self.candidate_encoder_layers):
            current_cand_seq_flat = layer_module(current_cand_seq_flat)
            _, L_i, D_i = current_cand_seq_flat.shape
            cand_skip_features_seq_kmax_list.append(current_cand_seq_flat.view(B, K, L_i, D_i))
            # print_shape_aft(cand_skip_features_seq_kmax_list[-1], f"cand_skip_features_seq_kmax_list[{i}] (Actual Output of Cand Enc Stage {i})")

        # 3. 디코더 및 특징 레벨 융합 (기존과 동일)
        deepest_ctx_seq_norm = self.norm_ctx_encoder(ctx_skip_features_seq_list[-1])
        H_p_bottleneck, W_p_bottleneck = self.ctx_encoder_stage_output_resolutions[-1] 
        current_decoded_spatial = self._seq_to_spatial(deepest_ctx_seq_norm, H_p_bottleneck, W_p_bottleneck)
        # print_shape_aft(current_decoded_spatial, "Initial current_decoded_spatial for decoder (bottleneck)")
        
        for i in range(len(self.decoder_fusion_stages)):
            H_p_target_for_stage, W_p_target_for_stage = self.decoder_target_operating_resolutions[i]
            x_input_for_stage_spatial = self.decoder_upsamplers[i](current_decoded_spatial)
            
            skip_encoder_idx = self.num_encoder_stages - 1 - i
            skip_ctx_seq_current = ctx_skip_features_seq_list[skip_encoder_idx]
            skip_cand_seq_kmax_current = cand_skip_features_seq_kmax_list[skip_encoder_idx]
            
            current_decoded_spatial = self.decoder_fusion_stages[i](
                x_input_for_stage_spatial, 
                skip_ctx_seq_current,
                skip_cand_seq_kmax_current,
                H_p_target_for_stage, 
                W_p_target_for_stage
            )
            
        # 4. 최종 특징 정규화 (기존과 동일)
        final_features_to_pred = current_decoded_spatial # 이 변수가 최종 디코더 특징맵
        if isinstance(self.final_features_norm, nn.LayerNorm):
            B_f, C_f, H_f, W_f = final_features_to_pred.shape
            final_features_seq = final_features_to_pred.flatten(2).transpose(1,2)
            final_features_seq = self.final_features_norm(final_features_seq)
            final_features_to_pred = final_features_seq.transpose(1,2).view(B_f, C_f, H_f, W_f)
        elif not isinstance(self.final_features_norm, nn.Identity): # Identity가 아니면 적용
            final_features_to_pred = self.final_features_norm(final_features_to_pred)
        
        # 5. 융합 방식에 따른 최종 처리 (중요: 반환 형식을 일관되게 맞춰야 함)
        # _apply_fusion_method의 각 브랜치가 (image_tensor, confidence_tensor, info_dict)를 반환하도록 하는 것이 이상적입니다.
        # image_tensor: RegionBased일 경우 [-2.118, 2.640] 범위, 그 외엔 로짓 (visualize_and_save_batch_aft에서 처리)
        # confidence_tensor: RegionBased일 경우 [0,1] 범위, 그 외엔 로짓 (visualize_and_save_batch_aft에서 처리)
        
        if self.fusion_method == "region_based":
            # candidate_scores 정보 추출 (새로 추가)
            candidate_scores = getattr(self, '_current_candidate_scores', None)
            
            # _apply_region_based_selection은 이미 (final_selected_image_norm_range, final_confidence_01, selection_details)를 반환합니다.
            return self._apply_region_based_selection(
                final_features_to_pred, candidate_images_kmax, partial_image, original_mask, candidate_scores
            )
        
        elif self.fusion_method == "pixel_fusion" and self.pixel_fusion_module is not None:
            # _apply_pixel_fusion은 (final_pixels_logits, final_confidence_logits)를 반환합니다.
            # visualize_and_save_batch_aft에서 3개 요소를 기대하므로, 빈 딕셔너리를 추가합니다.
            pred_logits, conf_logits = self._apply_pixel_fusion(final_features_to_pred, candidate_images_kmax, partial_image, original_mask)
            return pred_logits, conf_logits, {"fusion_method": "pixel_fusion"} # 3번째 요소로 info 추가
            
        elif self.fusion_method == "hybrid" and self.pixel_fusion_module is not None and self.region_based_selector is not None:
            # _apply_hybrid_fusion은 (final_pixels_logits, final_confidence_logits)를 반환합니다.
            # visualize_and_save_batch_aft에서 3개 요소를 기대하므로, 빈 딕셔너리를 추가합니다.
            pred_logits, conf_logits = self._apply_hybrid_fusion(final_features_to_pred, candidate_images_kmax, partial_image, original_mask)
            return pred_logits, conf_logits, {"fusion_method": "hybrid"} # 3번째 요소로 info 추가
            
        else: # "feature_based" (기본) 또는 fusion_method 설정이 잘못된 경우
            # _apply_feature_based_prediction은 (predicted_pixels_logits, confidence_map_logits)를 반환합니다.
            # visualize_and_save_batch_aft에서 3개 요소를 기대하므로, 빈 딕셔너리를 추가합니다.
            if self.fusion_method != "feature_based":
                 print_shape_aft(None, f"Warning: fusion_method '{self.fusion_method}' not fully handled, falling back to feature_based prediction return format.")
            pred_logits, conf_logits = self._apply_feature_based_prediction(final_features_to_pred)
            return pred_logits, conf_logits, {"fusion_method": "feature_based"} # 3번째 요소로 info 추가

    def _apply_fusion_method(self, final_features, candidate_images_kmax, partial_image, original_mask):
        """융합 방식에 따른 최종 처리"""
        
        if self.fusion_method == "pixel_fusion" and self.pixel_fusion_module is not None:
            return self._apply_pixel_fusion(final_features, candidate_images_kmax, partial_image, original_mask)
            
        elif self.fusion_method == "region_based" and self.region_based_selector is not None:
            return self._apply_region_based_selection(final_features, candidate_images_kmax, partial_image, original_mask)
            
        elif self.fusion_method == "hybrid" and self.pixel_fusion_module is not None and self.region_based_selector is not None:
            return self._apply_hybrid_fusion(final_features, candidate_images_kmax, partial_image, original_mask)
            
        else:
            # 기본값: 특징 기반 예측만
            return self._apply_feature_based_prediction(final_features)

    def _apply_pixel_fusion(self, final_features, candidate_images_kmax, partial_image, original_mask):
        """픽셀 융합 방식 적용 (기존 코드 유지)"""
        target_fusion_resolution = 128
        current_h, current_w = final_features.shape[2:]
        
        if current_h < target_fusion_resolution:
            upsampled_features = F.interpolate(
                final_features, 
                size=(target_fusion_resolution, target_fusion_resolution), 
                mode='bilinear', align_corners=False
            )
        else:
            upsampled_features = final_features
            
        candidates_at_fusion_res = self._resize_candidates_to_resolution(
            candidate_images_kmax, target_fusion_resolution, target_fusion_resolution
        )
        partial_at_fusion_res = F.interpolate(
            partial_image, size=(target_fusion_resolution, target_fusion_resolution), 
            mode='bilinear', align_corners=False
        )
        mask_at_fusion_res = F.interpolate(
            original_mask, size=(target_fusion_resolution, target_fusion_resolution), 
            mode='nearest'
        )
        
        fused_pixels_high_res, pixel_confidence_high_res, candidate_weights = self.pixel_fusion_module(
            candidates_at_fusion_res,
            partial_at_fusion_res,
            mask_at_fusion_res,
            upsampled_features
        )
        
        if target_fusion_resolution > current_h:
            fused_pixels_for_head = F.interpolate(
                fused_pixels_high_res, size=(current_h, current_w), 
                mode='bilinear', align_corners=False
            )
            pixel_confidence_for_head = F.interpolate(
                pixel_confidence_high_res, size=(current_h, current_w), 
                mode='bilinear', align_corners=False
            )
        else:
            fused_pixels_for_head = fused_pixels_high_res
            pixel_confidence_for_head = pixel_confidence_high_res
        
        combined_features = torch.cat([
            final_features,
            fused_pixels_for_head,
            pixel_confidence_for_head
        ], dim=1)
        
        output_logits_low_res = self.final_pred_head(combined_features)
        output_logits = self._apply_final_upsampling(output_logits_low_res)
        
        # 고해상도 융합 결과와 결합
        if fused_pixels_high_res.shape[2:] != self.final_target_img_size:
            fused_pixels_final_res = F.interpolate(
                fused_pixels_high_res,
                size=self.final_target_img_size,
                mode='bilinear',
                align_corners=False
            )
            pixel_confidence_final_res = F.interpolate(
                pixel_confidence_high_res,
                size=self.final_target_img_size,
                mode='bilinear',
                align_corners=False
            )
        else:
            fused_pixels_final_res = fused_pixels_high_res
            pixel_confidence_final_res = pixel_confidence_high_res
            
        cnn_predicted_pixels = torch.sigmoid(output_logits[:, :3, :, :])
        cnn_confidence = torch.sigmoid(output_logits[:, 3:4, :, :])
        
        fusion_weight = pixel_confidence_final_res
        final_pixels = (fusion_weight * fused_pixels_final_res + 
                       (1 - fusion_weight) * cnn_predicted_pixels)
        final_confidence = torch.max(cnn_confidence, pixel_confidence_final_res)
        
        final_pixels_logits = torch.logit(torch.clamp(final_pixels, 1e-7, 1-1e-7))
        final_confidence_logits = torch.logit(torch.clamp(final_confidence, 1e-7, 1-1e-7))
        
        return final_pixels_logits, final_confidence_logits



    # models/csf_network.py의 _apply_region_based_selection 메서드 수정
    # 기존 파일의 해당 메서드만 교체하세요

    def _apply_region_based_selection(self, final_features_from_decoder, candidate_images_kmax, partial_image, original_mask, candidate_scores=None):
        """개선된 Region-based Selection - 노이즈 제거"""
        
        if self.region_based_selector is None:
            pred_logits, conf_logits = self._apply_feature_based_prediction(final_features_from_decoder)
            return pred_logits, conf_logits, {"error": "RegionBasedSelector not initialized", "fallback_active": True}

        try:
            # 🔧 입력 정규화 확인 및 통일
            # candidate_images는 이미 정규화된 상태여야 함
            candidate_normalized = torch.clamp(candidate_images_kmax, -3.0, 3.0)
            partial_normalized = torch.clamp(partial_image, -3.0, 3.0)
            
            # RegionBasedSelector 호출
            # 🔧 RegionBasedSelector 호출 시 candidate_scores 전달
            final_output_image, output_confidence_01, output_selection_info = self.region_based_selector(
                candidate_normalized,
                partial_normalized,
                original_mask,
                final_features_from_decoder,
                candidate_scores=candidate_scores  # 새로 추가
            )
            
            # 🔧 출력 범위 안정화
            final_output_image = torch.clamp(final_output_image, -3.0, 3.0)
            output_confidence_01 = torch.clamp(output_confidence_01, 0.0, 1.0)
            
            # 🔧 마스크 영역만 처리했는지 재확인
            visible_mask = original_mask.expand_as(final_output_image)
            fill_mask = 1.0 - visible_mask
            
            # visible 영역은 무조건 원본 유지
            final_output_image = partial_normalized * visible_mask + final_output_image * fill_mask
            
            return final_output_image, output_confidence_01, output_selection_info
            
        except Exception as e:
            print(f"ERROR in _apply_region_based_selection: {e}")
            pred_logits, conf_logits = self._apply_feature_based_prediction(final_features_from_decoder)
            return pred_logits, conf_logits, {"error": str(e), "fallback_active": True}

    def _enhance_decoder_features(self, features):
        """디코더 특징 향상 (픽셀 레벨 최적화)"""
        if not hasattr(self, 'feature_enhancer'):
            # 동적으로 특징 개선기 생성 (더 세밀한 구조)
            self.feature_enhancer = nn.Sequential(
                nn.Conv2d(features.shape[1], features.shape[1], 3, padding=1),
                nn.BatchNorm2d(features.shape[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(features.shape[1], features.shape[1], 1),  # 1x1 conv for pixel refinement
                nn.BatchNorm2d(features.shape[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(features.shape[1], features.shape[1], 3, padding=1),
                nn.BatchNorm2d(features.shape[1])
            ).to(features.device)
        
        enhanced = self.feature_enhancer(features)
        return features + enhanced * 0.3  # 더 부드러운 residual connection

    def _enhance_candidate_quality(self, candidate_images, partial_image, mask):
        """후보 이미지 품질 개선 (픽셀 레벨 최적화)"""
        B, K, C, H, W = candidate_images.shape
        enhanced_candidates = candidate_images.clone()
        
        # 각 후보에 대해 더 정교한 품질 보정 적용
        for k in range(K):
            candidate = candidate_images[:, k]  # [B, C, H, W]
            
            # 1. 향상된 색상 보정
            candidate_corrected = self._advanced_color_correction(candidate, partial_image, mask)
            
            # 2. 픽셀 레벨 경계 부드럽게 처리
            candidate_smoothed = self._pixel_level_boundary_smoothing(candidate_corrected, mask)
            
            # 3. 노이즈 제거 (픽셀 레벨에서 중요)
            candidate_denoised = self._gentle_denoising(candidate_smoothed)
            
            enhanced_candidates[:, k] = candidate_denoised
        
        return enhanced_candidates

    def _advanced_color_correction(self, candidate, partial_image, mask):
        """향상된 색상 보정"""
        visible_mask = mask.expand_as(partial_image)
        if visible_mask.sum() > 0:
            # 채널별 히스토그램 매칭 근사
            partial_mean = (partial_image * visible_mask).sum(dim=(2, 3), keepdim=True) / (visible_mask.sum(dim=(2, 3), keepdim=True) + 1e-6)
            partial_std = ((partial_image - partial_mean)**2 * visible_mask).sum(dim=(2, 3), keepdim=True) / (visible_mask.sum(dim=(2, 3), keepdim=True) + 1e-6)
            partial_std = torch.sqrt(partial_std + 1e-6)
            
            candidate_mean = candidate.mean(dim=(2, 3), keepdim=True)
            candidate_std = candidate.std(dim=(2, 3), keepdim=True) + 1e-6
            
            # 표준화 후 목표 분포로 변환
            normalized = (candidate - candidate_mean) / candidate_std
            corrected = normalized * partial_std * 0.8 + partial_mean * 0.8 + candidate_mean * 0.2
            
            return torch.clamp(corrected, -3, 3)
        
        return candidate

    def _pixel_level_boundary_smoothing(self, candidate, mask):
        """픽셀 레벨 경계 부드럽게 처리"""
        # 더 작은 커널로 부드럽게 처리
        kernel_size = 3
        padding = kernel_size // 2
        
        # 적응적 블러링 (마스크 경계 근처에서만)
        # 마스크 경계 감지
        mask_float = mask.float()
        mask_grad = torch.abs(F.conv2d(mask_float, 
                                      torch.ones(1, 1, 3, 3).to(mask.device) / 9, 
                                      padding=1) - mask_float)
        boundary_strength = torch.clamp(mask_grad * 5, 0, 1)  # 경계 강도
        
        # 가우시안 블러 근사
        blurred = F.avg_pool2d(
            F.pad(candidate, (padding, padding, padding, padding), mode='reflect'),
            kernel_size, stride=1
        )
        
        # 경계에서만 적응적으로 블러링 적용
        boundary_mask = boundary_strength.expand_as(candidate)
        result = candidate * (1 - boundary_mask * 0.4) + blurred * (boundary_mask * 0.4)
        
        return result

    def _gentle_denoising(self, candidate):
        """부드러운 노이즈 제거"""
        # 간단한 bilateral filter 근사
        kernel = torch.ones(1, 1, 3, 3).to(candidate.device) / 9
        denoised = F.conv2d(
            F.pad(candidate, (1, 1, 1, 1), mode='reflect'),
            kernel.expand(candidate.shape[1], 1, 3, 3),
            groups=candidate.shape[1]
        )
        
        # 원본과 부드럽게 블렌딩
        return candidate * 0.85 + denoised * 0.15

    def _post_process_region_output(self, output_image, partial_image, mask, confidence, selection_info):
        """Region 출력 후처리 (픽셀 레벨 최적화)"""
        # 1. 신뢰도 기반 적응적 후처리
        high_conf_mask = (confidence > 0.8).float()
        medium_conf_mask = ((confidence > 0.4) & (confidence <= 0.8)).float()
        low_conf_mask = (confidence <= 0.4).float()
        
        # 2. 낮은 신뢰도 영역은 더 부드럽게 처리
        if low_conf_mask.sum() > 0:
            smoothed_output = F.avg_pool2d(
                F.pad(output_image, (1, 1, 1, 1), mode='reflect'),
                3, stride=1
            )
            output_image = (output_image * (1 - low_conf_mask) + 
                           smoothed_output * low_conf_mask * 0.7 + 
                           output_image * low_conf_mask * 0.3)
        
        # 3. 중간 신뢰도 영역은 살짝 부드럽게
        if medium_conf_mask.sum() > 0:
            gentle_smooth = F.avg_pool2d(
                F.pad(output_image, (1, 1, 1, 1), mode='reflect'),
                3, stride=1
            )
            output_image = (output_image * 0.8 + gentle_smooth * 0.2) * medium_conf_mask + \
                          output_image * (1 - medium_conf_mask)
        
        # 4. 최종 일관성 보장
        visible_mask = mask.expand_as(output_image)
        fill_mask = 1.0 - visible_mask
        
        # 보이는 영역은 항상 원본 사용
        final_output = partial_image * visible_mask + output_image * fill_mask
        
        return final_output


    

    def _apply_hybrid_fusion(self, final_features, candidate_images_kmax, partial_image, original_mask):
        """하이브리드 융합 방식 적용 (새로 추가)"""
        print("=== 하이브리드 융합 시작 ===")
        
        # 1. 픽셀 융합 결과
        pixel_result_logits, pixel_conf_logits = self._apply_pixel_fusion(
            final_features, candidate_images_kmax, partial_image, original_mask
        )
        pixel_result = torch.sigmoid(pixel_result_logits)
        pixel_conf = torch.sigmoid(pixel_conf_logits)
        
        # 2. Region-based 결과
        region_result_logits, region_conf_logits = self._apply_region_based_selection(
            final_features, candidate_images_kmax, partial_image, original_mask
        )
        region_result = torch.sigmoid(region_result_logits)
        region_conf = torch.sigmoid(region_conf_logits)
        
        # 3. 하이브리드 조합
        # 신뢰도 기반 적응적 가중치
        total_conf = pixel_conf + region_conf + 1e-8
        adaptive_region_weight = region_conf / total_conf * self.region_weight
        adaptive_pixel_weight = pixel_conf / total_conf * self.pixel_weight
        
        # 정규화
        total_weight = adaptive_region_weight + adaptive_pixel_weight
        adaptive_region_weight = adaptive_region_weight / total_weight
        adaptive_pixel_weight = adaptive_pixel_weight / total_weight
        
        # 최종 결합
        final_pixels = (adaptive_region_weight * region_result + 
                       adaptive_pixel_weight * pixel_result)
        final_confidence = torch.max(pixel_conf, region_conf)
        
        # logit으로 변환
        final_pixels_logits = torch.logit(torch.clamp(final_pixels, 1e-7, 1-1e-7))
        final_confidence_logits = torch.logit(torch.clamp(final_confidence, 1e-7, 1-1e-7))
        
        print("=== 하이브리드 융합 완료 ===")
        
        return final_pixels_logits, final_confidence_logits

    def _apply_feature_based_prediction(self, final_features):
        """기본 특징 기반 예측 (기존 방식)"""
        output_logits_low_res = self.final_pred_head(final_features)
        output_logits = self._apply_final_upsampling(output_logits_low_res)
        
        predicted_pixels_logits = output_logits[:, :3, :, :]
        confidence_map_logits = output_logits[:, 3:4, :, :]

        return predicted_pixels_logits, confidence_map_logits

    def _apply_final_upsampling(self, output_logits):
        """최종 업샘플링 적용"""
        if len(self.final_pixel_upsampler) > 0:
            return self.final_pixel_upsampler(output_logits)
        elif output_logits.shape[2:] != self.final_target_img_size:
            return F.interpolate(
                output_logits, 
                size=self.final_target_img_size, 
                mode='bilinear', 
                align_corners=False
            )
        else:
            return output_logits