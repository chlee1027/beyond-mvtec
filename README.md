# beyond-mvtec

**MVTec AD에서 고른 이상 탐지 설정이 다른 산업 데이터셋에서도 같은 순서로 통하는가.**

공장 외관 검사용 이상 탐지 모델은 정상 이미지만으로 학습한다. 백본, 특징을 꺼낼 층, 메모리 뱅크 크기 같은 설정은 보통 MVTec AD 결과를 보고 고른다. 그런데 세 데이터셋은 촬영 조건이 다르다.

- **MVTec AD**는 촬영 장비와 조명을 통제해 찍었다. 물체 부류 가운데 Toothbrush·Capsule·Pill은 원문 표현으로 「roughly aligned」이고, Metal Nut·Screw·Hazelnut은 무작위로 회전되어 있다(MVTec AD 논문 3절).
- **VisA**에는 한 이미지에 물체가 여러 개 든 부류가 4개 있다. Capsules는 약 20개, Candle·Macaroni1·Macaroni2는 4개씩이다. 논문이 물체 위치·자세 차이가 크다고 적은 부류는 Capsules·Macaroni2다.
- **Real-IAD**는 30 부류 151,050장 규모다. 한 물체를 다섯 방향에서 찍었고 원본 해상도가 2,000~5,000px이다. 생산 라인에서 찍은 이미지는 아니다. 실물 제품을 사서 결함을 손으로 만들고, 시제품 촬영 장치로 찍었다(Real-IAD 논문).

이 레포는 MVTec AD에서 매긴 설정 순위가 VisA와 Real-IAD에서도 유지되는지 잰다.

> **상태 (2026-11-06).** 구현과 재현 노트북까지 끝났다. MVTec 15 부류 재현은 아직 돌리지 않았고 결과 표는 비어 있다. 지금 있는 값은 CPU에서 두 부류, 시드 하나로 코드를 확인한 것이다(아래 「구현 확인」). 관련 논문 여섯 편의 분석 노트는 [`papers/notes/`](papers/notes/)에 있다.

---

## 질문

| 번호 | 질문 | 재는 것 |
|---|---|---|
| Q1 (주) | MVTec AD에서 정한 설정 순위가 VisA, Real-IAD에서도 같은가 | 설정 12개의 이미지 AUROC 순위, 데이터셋 쌍마다 Kendall 순위 상관 |
| Q2 (보조) | 정상 학습 이미지를 줄일 때 떨어지는 폭이 데이터셋마다 다른가 | 학습 장수(전체 / 50 / 10)별 이미지 AUROC |

Q2는 새 제품을 라인에 막 넣어 정상 샘플이 적은 상황을 흉내 낸다. MVTec AD는 부류마다 학습 장수가 60장(Toothbrush)에서 391장(Hazelnut)까지다. 그래서 「50장」은 Toothbrush에서는 전체의 83%이고 Hazelnut에서는 13%다.

## 데이터셋

| 데이터셋 | 내용 | 이 레포에서 쓰는 범위 | 받는 곳 · 라이선스 |
|---|---|---|---|
| MVTec AD | 15 부류. 학습 3,629장(정상만), 시험 1,725장. 정사각 700~1024px. Grid·Screw·Zipper는 흑백 | 15 부류 전부 | MVTec 홈페이지. CC BY-NC-SA 4.0(홈페이지 기준이고 논문에는 적혀 있지 않다) |
| VisA | 12 부류. 정상 9,621장, 이상 1,200장. 4,000 x 6,000 센서로 촬영 | 12 부류 전부, 공식 1cls 분할(부류마다 정상 90%로 학습하고, 남은 정상 10%와 이상 전부로 시험) | `amazon-science/spot-diff`. 저장소 코드는 Apache-2.0이고 데이터 라이선스는 확인 필요 |
| Real-IAD | 30 부류. 정상 99,721장, 이상 51,329장. 시점 5개 | 부류 일부와 위 시점 하나. 범위는 받은 뒤 확정 | Hugging Face `Real-IAD/Real-IAD`. 저자 승인 필요, 라이선스 확인 필요 |

데이터는 레포에 올리지 않는다. 받는 순서는 [`data/README.md`](data/README.md)에 있다.

**Real-IAD를 쓰는 방식.** 논문의 single-view 설정은 30 부류 모두에서 위(top view) 한 시점만 쓴다. 이 레포도 같은 시점을 쓰지만 부류는 일부만 쓴다. 그래서 30 부류 평균이 아니라 부록 표 5에서 해당 부류 행과 맞댄다. 두 가지를 더 지킨다.
- 시점 이름 C1~C5는 anomalib 문서의 표기이고 논문에는 없다(논문은 No.1~No.5). C1이 위 시점인지는 받은 뒤 사진으로 확인한다.
- 이상 표본 안에서도 결함이 안 보이는 시점의 사진은 정상이다. 그래서 표본 라벨이 아니라 이미지 라벨로 채점한다.

## 방법과 설정

방법은 두 가지다. 둘 다 ImageNet으로 사전학습한 CNN을 고정한 채 특징만 꺼내 쓰고, 새로 학습하는 파라미터는 없다.

- **PaDiM**: 조각(patch) 위치마다 정상 특징의 평균과 공분산(가우시안)을 저장하고, 시험 조각은 마할라노비스 거리로 점수를 매긴다.
- **PatchCore**: 정상 조각 특징을 메모리 뱅크에 모으고 탐욕 코어셋으로 일부만 남긴다. 시험 조각은 뱅크에서 가장 가까운 거리로 점수를 매긴다.

**무엇을 직접 짰는가.**

| 부분 | 출처 |
|---|---|
| PatchCore | 원 저자 공개 코드 [`amazon-science/patchcore-inspection`](https://github.com/amazon-science/patchcore-inspection)(Apache-2.0)을 커밋 `fcaa92f`로 고정해 쓴다. 원본 파일은 고치지 않고 [`patchcore_adapter.py`](src/beyond_mvtec/patchcore_adapter.py)에서 바깥으로 끼운다. 바꾸는 것은 다섯이다. (1) 백본 목록에 resnet18 추가, (2) 이미지 점수와 분할 지도 저장(원본은 `results.csv`만 남긴다), (3) Q2용 50장·10장 표본 고르기(원본 `--train_val_split`은 정렬된 파일 앞에서부터 자른다), (4) VisA·Real-IAD 데이터 읽기, (5) 아래 「전처리」의 정사각 줄이기 |
| PaDiM | 저자 공식 코드가 없어서 직접 구현했다([`padim.py`](src/beyond_mvtec/padim.py)). 원 논문이 정한 값은 그대로 쓰고, 정하지 않은 값은 아래 「PaDiM에서 이 레포가 고른 값」에 적었다 |
| 지표 (이미지·픽셀 AUROC, PRO) | 두 방법이 같은 코드로 재도록 [`metrics.py`](src/beyond_mvtec/metrics.py) 하나를 쓴다. 원본 PatchCore 코드는 PRO를 계산하지 않는다 |
| 데이터 읽기 | 세 데이터셋의 파일 목록과 전처리는 [`data.py`](src/beyond_mvtec/data.py), 설정 목록·실행·판정은 [`experiment.py`](src/beyond_mvtec/experiment.py)에 있다 |

Q1에서 비교하는 설정은 12개다.

| 방법 | 백본 | 층 | 그 밖 | 개수 |
|---|---|---|---|---:|
| PatchCore | ResNet18, WRN-50-2 | (2,3), (1,2,3) | 코어셋 1%, 10% | 8 |
| PaDiM | ResNet18, WRN-50-2 | (1,2,3), (2,3) | 줄인 차원 R18 100, WRN 550 | 4 |

- **원 논문이 직접 잰 칸은 일부다.** PatchCore 논문이 잰 것은 WRN-50-2 (2,3)의 10%와 1%뿐이다. ResNet18은 논문에도 원본 코드의 백본 목록에도 없다. PaDiM 논문은 층 (1,2,3) 하나만 쟀다. 줄인 차원 100·550도 그 조합에서 정한 값이다. (2,3)에 같은 차원 수를 쓰는 것은 이 레포의 선택이다.
- **시드.** 코어셋 시작점과 PaDiM 차원 고르기에 무작위성이 있다. 그래서 설정마다 시드 3개로 돌리고 평균과 표준편차를 적는다. 원본 PatchCore 코드는 시드 하나로 numpy·torch·cuda를 모두 고정한다.
- **PaDiM 공분산과 적은 학습 장수.** 공분산 정규화 항은 원 논문 값 ε = 0.01을 고정해 쓴다. 학습이 N장이면 공분산 계수(rank)는 N−1을 넘지 못하고, 나머지 방향은 ε만으로 채워진다. MVTec 부류당 학습은 약 240장이다. 그래서 WRN-550은 학습 장수를 줄이지 않아도 이 상태다. Q2의 50장·10장에서는 R18-100도 그렇다. 원 논문은 이 상황을 다루지 않는다. 그래서 결과 표에 ε를 함께 적는다.

정확한 값은 [`configs/experiments.yaml`](configs/experiments.yaml)에 있다.

**PaDiM에서 이 레포가 고른 값.** 원 논문에 적혀 있지 않은 자리다. 재현이 어긋나면 여기부터 다시 본다.

| 자리 | 고른 값 |
|---|---|
| 가중치 | torchvision `IMAGENET1K_V1`. 원본 PatchCore의 `pretrained=True`와 같은 가중치다 |
| 입력 정규화 | ImageNet 평균·표준편차 |
| 작은 특징 지도를 격자에 맞추는 법 | 최근접 복제. 격자는 쓴 층 가운데 가장 큰 지도다 |
| 무작위 차원 | 시드 하나에 한 벌을 전 위치·전 부류에 쓴다. 중복 없이 뽑는다 |
| 이미지 점수 | 쌍삼차로 입력 크기까지 키우고 sigma=4로 평활한 지도의 최댓값 |
| 평활 | `scipy.ndimage.gaussian_filter` 기본값. 원본 PatchCore의 평활과 같은 함수다 |

**전처리.** PatchCore 원 논문은 256으로 줄인 뒤 가운데 224를 자른다. 이렇게 자르면 가장자리 결함이 잘릴 수 있다. 원 논문 부록 C.4는 거짓 음성 23장 가운데 1장을 그런 사례로 든다. 그래서 두 단계로 나눈다.
- 재현 단계(MVTec만): 원 논문과 같이 256 -> 224 가운데 자르기.
- 비교 단계(세 데이터셋): 256 x 256으로 줄이기만 하고 자르지 않는다. torchvision의 `Resize(256)`은 짧은 변만 맞추고 비율을 지키므로, 정사각이 아닌 이미지에 `CenterCrop`을 붙이면 긴 변이 잘린다. 그래서 `Resize((256, 256))`으로 두 변을 다 맞춘다. VisA처럼 정사각이 아닌 이미지는 가로세로비가 바뀐다. 이 선택은 VisA 해상도를 잰 뒤 다시 본다.

두 단계에 공통으로 적용하는 것도 있다.
- 흑백 이미지(MVTec Grid·Screw·Zipper)는 채널을 세 번 복사한다.
- 정답 마스크는 최근접 보간으로 줄인다. 원본 PatchCore 코드는 양선형 보간 뒤 정수로 바꾸는데, 그러면 마스크 가장자리가 줄어든다.

## 진행 순서

1. **재현.** MVTec AD에서 원 논문 값과 맞는지 확인한다. PatchCore는 원본 코드가 이 레포의 데이터 읽기·지표 코드를 거쳐도 같은 값을 내는지, PaDiM은 직접 구현이 맞는지를 본다. 맞지 않으면 다음 단계로 가지 않는다.
2. **Q1.** 비교 단계 전처리로 설정 12개 x 데이터셋 3개 x 시드 3개를 돌린다.
3. **Q2.** 방법마다 Q1에서 가장 좋았던 설정 하나로 학습 장수를 줄여 본다.
4. **정리.** 결과 표, 그림, 한계를 적는다.

### 재현 목표 (MVTec AD 15 부류 평균)

| 방법 | 이미지 AUROC | 픽셀 AUROC | PRO | 출처 |
|---|---:|---:|---:|---|
| PatchCore-10%, WRN-50-2 (2,3) | 99.0 | 98.1 | 93.5 | PatchCore 논문 표 1 · 2 · 3 |
| PatchCore-1%, WRN-50-2 (2,3) | 99.0 | 98.0 | 93.1 | 같은 곳 |
| PaDiM, WRN-50-2 | 95.3 | 97.5 | 92.1 | PaDiM 논문 표 V(이미지) · 표 III(픽셀, PRO) |
| PaDiM, ResNet18 (차원 100) | 논문에 없음 | 96.7 | 90.1 | PaDiM 논문 표 III |

판정 기준: 이미지 AUROC는 목표값 ±0.5 안, 픽셀 AUROC와 PRO는 ±1.0 안이면 재현된 것으로 본다. 이 폭은 시드 3개의 흩어짐을 본 뒤 다시 정한다. 판정할 때 지킬 것이 둘 있다.
- **PatchCore 원본 코드는 논문과 값이 다르다.** 원본 코드 README가 보고한 WRN-50-2 10%(시드 0) 값은 99.2 · 98.1이다. 부류 값도 논문과 다르고, Screw에서는 1.77점 차이가 난다. 코드 README의 PRO 열(94.4)은 논문 93.5와 0.9 다르다. 코드의 PRO는 이 레포 정의와 다른 계산일 수 있다. 그래서 PRO가 어긋나면 정의부터 맞춰 본다.
- **부류별로 대조할 때.** Toothbrush는 시험이 정상 12장, 결함 30장이라 (정상, 결함) 쌍 하나가 뒤집히면 이미지 AUROC가 0.28점 움직인다.

### 구현 확인 (2026-11-06)

재현 판정이 아니다. 조건은 CPU, MVTec의 Bottle·Screw 두 부류, 시드 0이다. 값은 [`results/check_two_classes_cpu.csv`](results/check_two_classes_cpu.csv)에 있다.

**원본 실행기와 대조.** 같은 조건에서 원본 `bin/run_patchcore.py`(장치 선택 함수만 CPU로 바꿔 끼움)와 이 레포 경로의 이미지 AUROC를 맞댔다. 세 칸 모두 CSV에 찍힌 자릿수까지 같다.

| 설정 | 부류 | 원본 실행기 | 이 레포 |
|---|---|---:|---:|
| WRN-50-2 (2,3) 1% | Screw | 0.9739700758352122 | 0.9739700758352122 |
| WRN-50-2 (2,3) 1% | Bottle | 1.0 | 1.0 |
| WRN-50-2 (2,3) 10% | Screw | 0.9877023980323837 | 0.9877023980323837 |

10% Screw 값은 원본 레포에 공개된 모델의 `results.csv`(시드 0) 값과도 같다.

**논문 부류 값과 대조.** 왼쪽이 이 레포, 오른쪽이 논문이다(PatchCore 표 S1·S2·S3, PaDiM 표 III). 단위는 %다.

| 설정 | 부류 | 이미지 AUROC | 픽셀 AUROC | PRO |
|---|---|---:|---:|---:|
| PatchCore-10% | Bottle | 100 / 100 | 98.59 / 98.6 | 94.68 / 96.1 |
| PatchCore-10% | Screw | 98.77 / 97.0 | 99.42 / 99.4 | 97.05 / 97.9 |
| PatchCore-1% | Bottle | 100 / 100 | 98.58 / 98.5 | 94.65 / 95.9 |
| PatchCore-1% | Screw | 97.40 / 96.4 | 99.23 / 99.2 | 95.97 / 97.1 |
| PaDiM WRN-50-2 | Bottle | 99.92 / 없음 | 98.15 / 98.3 | 94.87 / 94.8 |
| PaDiM WRN-50-2 | Screw | 86.14 / 없음 | 98.45 / 98.5 | 93.89 / 94.4 |
| PaDiM ResNet18 | Bottle | 99.60 / 없음 | 98.08 / 98.1 | 94.27 / 93.9 |
| PaDiM ResNet18 | Screw | 78.21 / 없음 | 97.33 / 97.4 | 90.88 / 91.3 |

- **PatchCore PRO가 논문보다 0.85~1.42점 낮다. 원인은 찾지 못했다.** 같은 지표 코드에서 PaDiM은 논문과 -0.51~+0.37점 차이다. 저장한 지도로 마스크 처리(최근접, 원본식)와 이미지 범위(전체, 결함 이미지만)를 조합한 네 가지를 다시 재 봤지만 네 칸 모두 논문보다 낮았다. 15 부류 평균이 판정 폭 ±1.0을 넘을 수 있으므로 재현 때 먼저 본다.
- **Screw 이미지 AUROC는 논문보다 1.0~1.77점 높다.** 원본 실행기도 같은 값을 내므로 이 레포 경로의 차이가 아니다.
- **마스크 처리가 픽셀 AUROC를 바꾼다.** 원본은 Screw 99.41·99.52, Bottle 98.47·98.48(1%·10%)이다. 이 레포 마스크(최근접)로는 위 표의 값이 된다. 이 레포 지도에 원본식 마스크를 씌워 다시 재면 원본 값이 나와 원인이 마스크임을 확인했다. 원본식에서는 결함 픽셀 수가 Screw 68.7%, Bottle 92.9%로 준다. 논문과의 차이는 최근접이 0.08점 이내, 원본식이 0.21점 이내다.

**아직 확인하지 못한 것.** GPU와 faiss GPU 경로, VisA·Real-IAD 실제 파일. Real-IAD 읽기는 가짜 파일로만 시험했다. 시험은 `tests/`에 58개 있고 모두 통과한다. PRO는 문턱마다 직접 세는 계산과 맞댄다.

### 다른 데이터셋의 참고값 (범위 확인용)

아래 값은 백본·층·코어셋 비율·해상도가 논문에 적혀 있지 않거나 이 레포와 다르다. 그래서 재현 판정에는 쓰지 않고, 이 레포 값이 같은 범위에 드는지만 본다.

| 데이터셋 | 방법 | 이미지 AUROC | 픽셀 지표 | 출처와 조건 |
|---|---|---:|---:|---|
| VisA (1cls, 12 부류 평균) | PatchCore, WRN-50 | 92.4 | 98.4 (픽셀 AUROC) | VisA 논문 표 6. 같은 표의 MVTec 값이 99.8이라 PatchCore 원 논문(99.0~99.1)과 설정이 다르다 |
| VisA (1cls, 12 부류 평균) | PaDiM, ResNet-50 | 87.8 | 93.1 (픽셀 AUROC) | VisA 논문 표 2. 백본이 이 레포와 다르다 |
| Real-IAD (single-view, 30 부류 평균) | PatchCore | 90.4 | 92.5 | Real-IAD 논문 표 2, anomalib으로 돌림. 픽셀 지표 이름이 표 2에서는 P-AUPRO, 표 5에서는 P-AUROC로 갈린다 |
| Real-IAD (single-view, 30 부류 평균) | PaDiM | 86.6 | 90.0 | 같은 곳 |

Q2의 참고값(MVTec AD, PatchCore 논문 표 S5)도 있다. 학습 10장에서 이미지 AUROC는 PatchCore-10이 93.6, 다시 구현한 PaDiM이 83.2다. 학습 50장에서는 전체 학습 때보다 각각 1.5, 5.3 떨어진다.

## 지표

- **이미지 AUROC**: 이미지 점수로 정상과 이상을 가를 때의 ROC 곡선 아래 넓이. 부류마다 재서 평균한다.
- **픽셀 AUROC**: 같은 넓이를 픽셀 하나하나를 표본으로 잰 것.
- **PRO (per-region overlap)**: 정답 마스크를 연결 성분(결함 영역)으로 나눈다. 영역마다 예측과 겹친 비율을 재고, 영역 크기와 상관없이 평균한다. 문턱을 훑어 거짓 양성률(FPR) 0~0.3 구간의 곡선 넓이를 구하고 최댓값이 1이 되게 정규화한다. 출처는 Uninformed Students 논문 4.2절이다. MVTec AD 논문(CVPR 2019)에는 이 정의가 없다.
  - 원 논문이 정하지 않은 자리는 MVTec 평가 코드(`pro_curve_util.py`, `generic_util.py`)의 선택을 따랐다. 평균 단위 하나만 바꿔도 같은 예측이 0.25와 0.50으로 갈리는 예가 있어서(Uninformed Students 노트 7절) 아래에 다 적는다.

    | 자리 | 이 레포의 선택 |
    |---|---|
    | 연결 방식 | 8-연결 |
    | 문턱 | 점수의 서로 다른 값 전부. 개수를 따로 정하지 않는다 |
    | FPR | 부류의 정상 픽셀을 모두 모아 한 번에 나눈다 |
    | 결함 없는 시험 이미지 | 넣는다(정상 픽셀에 들어간다) |
    | 영역 평균 | 부류의 모든 영역에 같은 무게. 이미지별로 먼저 평균하지 않는다 |
    | FPR 0.3 | 양옆 점을 선형 보간 |
    | 적분 | 사다리꼴. 곡선은 (0, 0)에서 시작하고 0.3으로 나눈다 |
    | 해상도 | 입력 크기(재현 224, 비교 256). 마스크는 최근접으로 줄인다 |
    | 평균 단위 | 부류마다 재서 단순 평균 |

  - 이 선택의 근거는 MVTec 평가 코드의 사본(`TArdelean/AnomalyLocalizationFCA`의 `evaluation/`)을 읽은 것이다. MVTec 배포본을 직접 받아 대조하지 못했고, 사본을 실행해 값을 맞대 보지도 않았다.
- **Kendall 순위 상관 (Q1)**: 두 데이터셋에서 설정 12개의 순위가 얼마나 같은지 나타내는 값. 1이면 순위가 똑같고 0이면 관계가 없다. 설정이 12개뿐이라 이 값 하나로는 불확실성이 크다. 그래서 순위 표를 같이 싣는다.
  - 예로, Real-IAD 논문 표 2의 방법 일곱을 이미지 AUROC 순위로 비교하면 MVTec–Real-IAD 사이 Kendall 순위 상관이 0.62다(21쌍 중 4쌍이 뒤집힘, 내가 계산한 값). 이것은 방법 사이의 순위이고, 이 레포가 재는 설정 사이의 순위가 아니다.

## 결과

아직 없다.

## 한계

아직 없다. 실험을 돌린 뒤 채운다.

## 논문 노트

원문을 처음부터 끝까지 읽고 쓴 분석 노트다. 표·그림마다 무엇을 바꾸고 무엇을 고정했는지, 반복 수는 몇인지, 주장이 결과보다 센지를 적었다. 부류별 표가 있으면 평균을 다시 계산해 인쇄된 값과 대조했다.

| 노트 | 이 레포에서 쓰는 곳 |
|---|---|
| [PatchCore (CVPR 2022)](papers/notes/CVPR22_PatchCore_Towards_Total_Recall_in_Industrial_Anomaly_Detection.md) | 비교 방법, 원본 코드 사용법, 재현 목표 |
| [PaDiM (ICPR 2021 워크숍)](papers/notes/ICPRW21_PaDiM_Patch_Distribution_Modeling_Framework_for_Anomaly_Detection_and_Localization.md) | 비교 방법, 직접 구현할 때의 구현 명세 |
| [MVTec AD (CVPR 2019)](papers/notes/CVPR19_MVTecAD_Comprehensive_Real_World_Dataset_for_Unsupervised_Anomaly_Detection.md) | 데이터셋 구성, 부류별 장수와 정렬 상태 |
| [Uninformed Students (CVPR 2020)](papers/notes/CVPR20_Uninformed_Students_Student_Teacher_Anomaly_Detection_with_Discriminative_Latent_Embeddings.md) | PRO 지표의 원 출처, PRO 구현 명세 |
| [VisA (ECCV 2022)](papers/notes/ECCV22_VisA_SPot_the_Difference_Self_Supervised_Pretraining_for_Anomaly_Detection_and_Segmentation.md) | 데이터셋 구성, 공식 분할, 참고값 |
| [Real-IAD (CVPR 2024)](papers/notes/CVPR24_RealIAD_Real_World_Multi_View_Dataset_for_Benchmarking_Versatile_Industrial_Anomaly_Detection.md) | 데이터셋 구성, single-view 정의, 참고값 |

논문 PDF는 저작권 때문에 레포에 올리지 않는다. `bash papers/download.sh`로 `papers/pdf/`에 받는다.

## 폴더

```
configs/experiments.yaml   실험 범위(설정, 시드, 학습 장수)
data/README.md             데이터 받는 법 (데이터 자체는 올리지 않음)
papers/notes/              논문 분석 노트 여섯 편
papers/download.sh         논문 PDF 받기 (PDF는 올리지 않음)
src/beyond_mvtec/          구현: data, metrics, padim, patchcore_adapter, experiment
tests/                     pytest. 원본 PatchCore를 쓰는 시험은 PATCHCORE_SRC가 없으면 건너뛴다
notebooks/00_check_visa_realiad.ipynb   Drive의 VisA·Real-IAD 점검 (학습 없음)
notebooks/01_reproduce_mvtec.ipynb      1단계 재현 (Colab)
results/                   결과 CSV와 그림
```

## 실행 환경

Google Colab Pro, NVIDIA L4 GPU. 패키지는 `requirements.txt`. faiss는 환경마다 판이 달라 목록에 넣지 않았다(Colab GPU는 `faiss-gpu-cu12`, CPU는 `faiss-cpu`).

시험은 CPU에서 돈다.

```
git clone https://github.com/amazon-science/patchcore-inspection.git
git -C patchcore-inspection checkout fcaa92f
PATCHCORE_SRC=patchcore-inspection/src python -m pytest tests
```

## 참고 문헌

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
- Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, ICPR Workshops 2021.
- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Bergmann et al., *Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings*, CVPR 2020.
- Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation* (VisA), ECCV 2022.
- Wang et al., *Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection*, CVPR 2024.
