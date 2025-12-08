# models/components/cross_attention_scorer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionScorer(nn.Module):
    """
    부분 이미지 컨텍스트 특징(Query)과 후보의 전역 특징(Key/Value) 간의
    교차 어텐션을 수행하여 관련성 점수를 계산합니다.
    """
    def __init__(self, feature_dim, num_heads=4, ff_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        if ff_dim is None:
            ff_dim = feature_dim * 2
            
        # 간단한 단일 교차 어텐션 레이어
        # Query: 컨텍스트 특징 (1개 토큰으로 가정 또는 평균 풀링된 특징)
        # Key/Value: 후보 특징 (1개 토큰으로 가정 또는 평균 풀링된 특징)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 점수 예측을 위한 MLP
        self.score_mlp = nn.Sequential(
            nn.Linear(feature_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, 1)  # 최종 스칼라 점수 (로짓)
        )
        self.norm_q = nn.LayerNorm(feature_dim)
        self.norm_kv = nn.LayerNorm(feature_dim)
        
    def forward(self, context_feature, candidate_feature):
        """
        Args:
            context_feature (torch.Tensor): 부분 이미지 컨텍스트의 특징 벡터. [B, D_feat]
                                           또는 [B, 1, D_feat] (시퀀스 길이 1)
            candidate_feature (torch.Tensor): 후보 이미지의 전역 특징 벡터. [B, D_feat]
                                            또는 [B, 1, D_feat] (시퀀스 길이 1)
        Returns:
            torch.Tensor: 관련성 점수 (로짓). [B, 1]
        """
        if context_feature.dim() == 2:  # [B, D] -> [B, 1, D]
            context_feature_seq = context_feature.unsqueeze(1)
        else:
            context_feature_seq = context_feature
            
        if candidate_feature.dim() == 2:  # [B, D] -> [B, 1, D]
            candidate_feature_seq = candidate_feature.unsqueeze(1)
        else:
            candidate_feature_seq = candidate_feature
            
        q = self.norm_q(context_feature_seq)
        k = v = self.norm_kv(candidate_feature_seq)
        
        # attn_output: [B, 1, D_feat], attn_weights는 사용 안함
        attn_output, _ = self.cross_attn(query=q, key=k, value=v)
        
        # 어텐션 출력을 사용하여 점수 계산
        score_logit = self.score_mlp(attn_output.squeeze(1))  # [B, 1]
        
        return score_logit

if __name__ == '__main__':
    # 테스트
    feat_dim = 128
    scorer = CrossAttentionScorer(feature_dim=feat_dim)
    
    batch_size = 4
    dummy_context_feat = torch.randn(batch_size, feat_dim)
    dummy_candidate_feat = torch.randn(batch_size, feat_dim)
    
    scores = scorer(dummy_context_feat, dummy_candidate_feat)
    print("Cross Attention Scores shape:", scores.shape)  # 예상: [4, 1]
    print("Scores (logits):", scores)