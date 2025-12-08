# models/components/simple_feature_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFeatureExtractor(nn.Module):
    """
    교차 주의 점수 계산 시 컨텍스트 및 후보 영역에서 특징을 추출하기 위한 간단한 CNN.
    입력 이미지 (또는 패치)를 받아 고정된 차원의 특징 벡터를 출력합니다.
    """
    def __init__(self, in_channels=3, output_dim=256, img_size=64):
        super().__init__()
        self.output_dim = output_dim
        
        # 입력 이미지 크기에 따라 레이어 구성 조절 가능
        # 여기서는 img_size에 비교적 덜 민감한 구조 (AdaptiveAvgPool 사용)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 16 -> 8
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, output_dim, kernel_size=3, stride=2, padding=1)  # 8 -> 4
        self.bn4 = nn.BatchNorm2d(output_dim)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.adaptive_pool(x)  # [B, output_dim, 1, 1]
        x = x.view(x.size(0), -1)  # [B, output_dim]
        return x

if __name__ == '__main__':
    # 테스트
    extractor = SimpleFeatureExtractor(output_dim=128, img_size=256)
    dummy_input = torch.randn(4, 3, 256, 256)  # B, C, H, W
    features = extractor(dummy_input)
    print("Extracted features shape:", features.shape)  # 예상: [4, 128]
    
    extractor_small = SimpleFeatureExtractor(output_dim=64, img_size=64)
    dummy_input_small = torch.randn(4, 3, 64, 64)
    features_small = extractor_small(dummy_input_small)
    print("Extracted features (small) shape:", features_small.shape)  # 예상: [4, 64]