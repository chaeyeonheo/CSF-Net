# models/components/fusion_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath

# ============================================================
# ==================== DEBUGGING PRINTS ====================
# ============================================================
# 나중에 쉽게 제거하려면 이 블록 주석을 검색하세요.
# 예: DEBUG_FUSION_ATTENTION = os.environ.get('DEBUG_FUSION_ATTENTION', 'False') == 'True'
DEBUG_FUSION_ATTENTION = False # True로 설정하면 프린트 활성화
def print_shape(tensor, name, enabled=DEBUG_FUSION_ATTENTION):
    if enabled and tensor is not None:
        print(f"[DEBUG FusionAttention] {name}: {tensor.shape}")
# ============================================================
# ================= END DEBUGGING PRINTS ===================
# ============================================================

class FusionAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, Kmax, mlp_ratio=4., qkv_bias=True,
                 attn_drop=0., proj_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_meta_mlp=True, meta_mlp_hidden_ratio=0.25):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.Kmax = Kmax
        self.scale = (dim // num_heads) ** -0.5

        # 개선된 입력 정규화 (LayerNorm + BatchNorm 결합)
        self.norm_q = nn.Sequential(
            norm_layer(dim),
            nn.BatchNorm1d(dim) if dim > 1 else nn.Identity()
        )
        self.norm_kv = nn.Sequential(
            norm_layer(dim),
            nn.BatchNorm1d(dim) if dim > 1 else nn.Identity()
        )

        # 개선된 Query, Key, Value 프로젝션 (Dropout 추가)
        self.to_q = nn.Sequential(
            nn.Linear(dim, dim, bias=qkv_bias),
            nn.Dropout(proj_drop * 0.5)  # 절반 강도로 Dropout
        )
        self.to_k = nn.Sequential(
            nn.Linear(dim, dim, bias=qkv_bias),
            nn.Dropout(proj_drop * 0.5)
        )
        self.to_v = nn.Sequential(
            nn.Linear(dim, dim, bias=qkv_bias),
            nn.Dropout(proj_drop * 0.5)
        )

        self.attn_drop = nn.Dropout(attn_drop)

        # 개선된 어텐션 결과 출력 프로젝션
        self.proj_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

        # DropPath (Stochastic Depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 개선된 MLP (더 깊은 구조)
        self.norm_mlp = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(int(dim * mlp_ratio), int(dim * mlp_ratio * 0.5)),
            act_layer(),
            nn.Dropout(proj_drop * 0.5),
            nn.Linear(int(dim * mlp_ratio * 0.5), dim),
            nn.Dropout(proj_drop)
        )

        # 개선된 Meta-Weighting Network
        self.use_meta_mlp = use_meta_mlp
        if self.use_meta_mlp:
            self.meta_weight_predictor = nn.Sequential(
                nn.Linear(self.num_heads, max(self.num_heads, 8)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(max(self.num_heads, 8), max(self.num_heads // 2, 4)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(max(self.num_heads // 2, 4), 1)
            )
            
        # 개선된 온도 파라미터 (학습 가능한 범위 제한)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # 추가: 어텐션 안정화를 위한 스케일링 파라미터
        self.attention_scale = nn.Parameter(torch.ones(1))

    def forward(self, context_features_seq, candidate_features_seq_kmax):
        B, L, D = context_features_seq.shape
        _B, K, _L, _D = candidate_features_seq_kmax.shape
        
        # ==================== DEBUG START: FusionAttentionModule Inputs ====================
        print_shape(context_features_seq, "Input context_features_seq")
        print_shape(candidate_features_seq_kmax, "Input candidate_features_seq_kmax")
        # ===================== DEBUG END: FusionAttentionModule Inputs =====================
        
        assert L == _L and D == _D and B == _B and K == self.Kmax, \
            f"Dim mismatch: ctx {context_features_seq.shape}, cand {candidate_features_seq_kmax.shape}"

        shortcut = context_features_seq

        # 1. 개선된 입력 정규화
        q_src = self._apply_norm_sequence(context_features_seq, self.norm_q)
        kv_src_flat = candidate_features_seq_kmax.reshape(B * K * L, D)
        kv_src_flat_normalized = self._apply_norm_sequence(kv_src_flat, self.norm_kv)
        kv_src = kv_src_flat_normalized.view(B, K, L, D)
        
        # ==================== DEBUG START: Normalized Q, KV Src ====================
        print_shape(q_src, "Normalized q_src")
        print_shape(kv_src, "Normalized kv_src (reshaped)")
        # ===================== DEBUG END: Normalized Q, KV Src =====================

        # 2. 개선된 Query, Key, Value 프로젝션
        q = self.to_q(q_src).reshape(B, L, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        # ==================== DEBUG START: Projected Q ====================
        print_shape(q, "Projected q (B, num_heads, L, head_dim)")
        # ===================== DEBUG END: Projected Q =====================

        # Key, Value 생성 (배치 처리 최적화)
        kv_src_for_projection = kv_src.permute(0,2,1,3).reshape(B*L, K, D)
        k_flat = self.to_k(kv_src_for_projection)  # [B*L, K, D]
        v_flat = self.to_v(kv_src_for_projection)  # [B*L, K, D]
        
        # ==================== DEBUG START: Projected K, V flat ====================
        print_shape(k_flat, "Projected k_flat (B*L, K, D)")
        print_shape(v_flat, "Projected v_flat (B*L, K, D)")
        # ===================== DEBUG END: Projected K, V flat =====================
        
        # 멀티헤드 형태로 변환
        k = k_flat.reshape(B*L, K, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        v = v_flat.reshape(B*L, K, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        
        # ==================== DEBUG START: Reshaped K, V for Attention ====================
        print_shape(k, "Reshaped k (B*L, num_heads, K, head_dim)")
        print_shape(v, "Reshaped v (B*L, num_heads, K, head_dim)")
        # ===================== DEBUG END: Reshaped K, V for Attention =====================

        # 3. 개선된 어텐션 스코어 계산
        q_for_batch_attn = q.permute(0,2,1,3).reshape(B*L, self.num_heads, 1, D // self.num_heads)
        # ==================== DEBUG START: Q for Batch Attention ====================
        print_shape(q_for_batch_attn, "q_for_batch_attn (B*L, num_heads, 1, head_dim)")
        # ===================== DEBUG END: Q for Batch Attention =====================

        # 어텐션 스코어 계산 (온도 및 스케일링 적용)
        attn_scores = (q_for_batch_attn @ k.transpose(-2, -1)) * self.scale * self.attention_scale
        attn_scores = attn_scores.squeeze(2)  # [B*L, num_heads, K]
        # ==================== DEBUG START: Attention Scores ====================
        print_shape(attn_scores, "attn_scores (B*L, num_heads, K)")
        # ===================== DEBUG END: Attention Scores =====================

        # 4. 개선된 Meta-Weights 계산
        if self.use_meta_mlp:
            # 어텐션 스코어를 정규화하여 안정성 향상
            normalized_attn_scores = F.softmax(attn_scores, dim=-1)
            meta_input = normalized_attn_scores.permute(0,2,1).reshape(B*L*K, self.num_heads)
            meta_logits_flat = self.meta_weight_predictor(meta_input)
            meta_logits = meta_logits_flat.view(B*L, K)
        else:
            meta_logits = attn_scores.mean(dim=1)
        # ==================== DEBUG START: Meta Logits ====================
        print_shape(meta_logits, "meta_logits (B*L, K)")
        # ===================== DEBUG END: Meta Logits =====================
            
        # 온도 제한을 통한 안정성 향상
        clamped_temperature = torch.clamp(self.temperature, min=0.01, max=2.0)
        meta_weights = F.softmax(meta_logits / torch.exp(clamped_temperature), dim=-1)
        # ==================== DEBUG START: Meta Weights ====================
        print_shape(meta_weights, "meta_weights (B*L, K) after softmax & temp")
        # ===================== DEBUG END: Meta Weights =====================
        
        # 5. 개선된 어텐션 확률 계산 및 Value 융합
        attn_probs = F.softmax(attn_scores / torch.exp(clamped_temperature), dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        # ==================== DEBUG START: Attention Probabilities ====================
        print_shape(attn_probs, "attn_probs (B*L, num_heads, K) after softmax & temp & drop")
        # ===================== DEBUG END: Attention Probabilities =====================

        # Value 가중합 (개선된 수치 안정성)
        fused_v_per_head = (attn_probs.unsqueeze(2) @ v)  # [B*L, num_heads, 1, head_dim]
        fused_v_seq = fused_v_per_head.squeeze(2).transpose(1,2).contiguous().view(B*L, D)

        # 6. 개선된 최종 출력 프로젝션
        fused_output_seq = self.proj_out(fused_v_seq)
        fused_output_seq = fused_output_seq.view(B, L, D)
        # ==================== DEBUG START: Projected Fused Output ====================
        print_shape(fused_output_seq, "fused_output_seq (B, L, D) after projection")
        # ===================== DEBUG END: Projected Fused Output =====================

        # 7. 첫 번째 Skip Connection (Residual Connection 강화)
        x = shortcut + self.drop_path(fused_output_seq)

        # 8. 개선된 MLP 파트 및 두 번째 Skip Connection
        mlp_input = self.norm_mlp(x)
        mlp_output = self.mlp(mlp_input)
        x = x + self.drop_path(mlp_output)
        # ==================== DEBUG START: Final Output of FusionAttentionModule ====================
        print_shape(x, "Final output x (B, L, D)")
        # ===================== DEBUG END: Final Output of FusionAttentionModule =====================
        
        return x

    def _apply_norm_sequence(self, x, norm_module):
        """시퀀스 데이터에 정규화 적용"""
        if isinstance(norm_module, nn.Sequential):
            # LayerNorm 먼저 적용
            x = norm_module[0](x)
            # BatchNorm이 있는 경우 적용 (차원 변환 필요)
            if len(norm_module) > 1 and not isinstance(norm_module[1], nn.Identity):
                original_shape = x.shape
                x = x.view(-1, original_shape[-1])  # [B*L, D]
                x = norm_module[1](x)
                x = x.view(original_shape)  # [B, L, D]
        else:
            x = norm_module(x)
        return x
    


    