# CSF-Net 업로드 가이드

## ✅ 코드 준비 완료!

모든 파일이 **CSF-Net** 이름으로 변경되고 커밋되었습니다.

## 📦 현재 상황

```
✅ 파일 이름 변경 완료
   - train_aft.py → train_csf.py
   - inference_aft.py → inference_csf.py
   - models/aft_network.py → models/csf_network.py
   - data_utils/aft_dataset.py → data_utils/csf_dataset.py
   - losses/aft_losses.py → losses/csf_losses.py
   - configs/aft_config.yaml → configs/csf_config.yaml

✅ 코드 내용 변경 완료
   - AFTNetwork → CSFNetwork
   - AFTDataset → CSFDataset
   - AFTLoss → CSFLoss
   - 모든 import 문 수정 완료

✅ Git 커밋 완료
   - Commit: "Add CSF-Net implementation (WACV 2026)"
   - 28 files changed, 8779 insertions(+)
```

## 🚀 GitHub에 업로드하는 방법

### 방법 1: 로컬에서 push (권장)

```bash
# 1. 현재 레포 클론 (로컬 PC에서)
git clone https://github.com/chaeyeonheo/CSF-Net.git
cd CSF-Net

# 2. 원격 브랜치에서 pull
git pull origin main

# 3. 확인
ls -la
```

현재 서버에 커밋은 완료되어 있지만, push는 인증 문제로 실패했습니다.
서버의 `/home/user/CSF-Net` 디렉토리를 다운로드해서 로컬에서 push하세요.

### 방법 2: 압축 파일 다운로드 후 업로드

서버에 `CSF-Net-WACV2026.tar.gz` 파일이 생성되었습니다.

```bash
# 로컬 PC에서:
# 1. 압축 파일 다운로드 (서버에서 전송)

# 2. 압축 해제
tar -xzf CSF-Net-WACV2026.tar.gz
cd CSF-Net

# 3. Git 초기화 및 푸시
git init
git remote add origin https://github.com/chaeyeonheo/CSF-Net.git
git add .
git commit -m "Add CSF-Net implementation (WACV 2026)

Official PyTorch implementation of:
'CSF-Net: Context-Semantic Fusion Network for Large Mask Inpainting'
Chae-Yeon Heo and Yeong-Jun Cho
WACV 2026"

git push -u origin main
```

## 📂 업로드될 파일 목록

```
CSF-Net/
├── README.md                          (논문 기반)
├── LICENSE
├── WACV26 논문 PDF
├── requirements.txt
│
├── models/
│   ├── csf_network.py                 ✅ 이름 변경됨
│   └── components/
│       ├── fusion_attention.py
│       ├── region_based_selector.py   (SSN + PSN)
│       ├── swin_transformer_modules.py
│       └── ...
│
├── data_utils/
│   ├── csf_dataset.py                 ✅ 이름 변경됨
│   └── candidate_processing.py
│
├── losses/
│   ├── csf_losses.py                  ✅ 이름 변경됨
│   └── vgg_perceptual_loss.py
│
├── utils/
│   ├── checkpoint.py
│   ├── logger.py
│   └── visualization.py
│
├── configs/
│   └── csf_config.yaml                ✅ 이름 변경됨
│
├── scripts/
│   └── prepare_candidate_info.py
│
├── train_csf.py                       ✅ 이름 변경됨
└── inference_csf.py                   ✅ 이름 변경됨
```

## ⚠️ 업로드 후 해야 할 일

### 1. VGG 가중치 안내 추가
README에 VGG 가중치 다운로드 링크가 포함되어 있습니다:
```bash
wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth -P pretrained_model/
```

### 2. 데이터 경로 설명
`configs/csf_config.yaml`의 데이터 경로는 예시입니다.
사용자가 직접 수정해야 함을 README에 명시했습니다.

### 3. 사용 예시
README에 학습/추론 명령어가 포함되어 있습니다:
```bash
# Training
python train_csf.py --config_path ./configs/csf_config.yaml

# Inference
python inference_csf.py --config_path ./configs/csf_config.yaml --checkpoint_path /path/to/model.pth
```

## 🎯 완료 체크리스트

- ✅ 모든 파일 이름이 CSF-Net으로 변경됨
- ✅ 모든 클래스 이름이 CSF-Net으로 변경됨
- ✅ README가 논문 기반으로 작성됨
- ✅ requirements.txt 포함
- ✅ 논문 PDF 포함
- ✅ Git 커밋 완료
- ⏳ GitHub push 대기 중 (로컬에서 진행)

## 📧 문의

업로드 과정에서 문제가 있으면:
1. Git 이력 확인: `git log`
2. 파일 목록 확인: `ls -R`
3. import 오류 확인: `python -c "from models import CSFNetwork"`
