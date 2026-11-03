# beyond-mvtec

**정리된 데이터에서 고른 이상 탐지 설정이 현장형 데이터에서도 그대로 통하는가.**

공장 외관 검사용 이상 탐지 모델은 정상 이미지만으로 학습한다. 백본, 특징을 꺼낼 층, 메모리 뱅크 크기 같은 설정은 보통 MVTec AD 결과를 보고 고른다. MVTec AD는 촬영 조건을 맞춰 정리한 데이터다(물체가 가운데 있고 조명이 같다). 실제 라인에서 찍은 이미지는 부품이 여러 개 들어 있거나 정렬이 어긋나 있다. 이 레포는 MVTec AD에서 매긴 설정 순위가 VisA와 Real-IAD에서도 유지되는지 잰다.

> **상태 (2026-11-03).** 뼈대만 있다. 아직 아무 실험도 돌리지 않았고 결과 표는 비어 있다.

---

## 질문

| 번호 | 질문 | 재는 것 |
|---|---|---|
| Q1 (주) | MVTec AD에서 정한 설정 순위가 VisA, Real-IAD에서도 같은가 | 설정 12개의 이미지 AUROC 순위, 데이터셋 쌍마다 Kendall 순위 상관 |
| Q2 (보조) | 정상 학습 이미지를 줄일 때 떨어지는 폭이 데이터셋마다 다른가 | 학습 장수(전체 / 50 / 10)별 이미지 AUROC |

Q2는 새 제품을 라인에 막 넣어 정상 샘플이 적은 상황을 흉내 낸다.

## 데이터셋

| 데이터셋 | 내용 | 이 레포에서 쓰는 범위 | 받는 법 · 라이선스 |
|---|---|---|---|
| MVTec AD | 15 부류, 학습 3,629장 (정상만), 시험 1,725장 | 15 부류 전부 | MVTec 홈페이지 신청 후 내려받기, CC BY-NC-SA 4.0 |
| VisA | 12 부류, 부품 여러 개가 들어간 이미지 포함 | 12 부류 전부, 공식 1cls 분할 | amazon-science/spot-diff, 라이선스 확인 필요 |
| Real-IAD | 30 부류, 약 15만 장, 한 물체를 5개 시점(C1~C5)에서 촬영 | 부류 일부와 한 시점, 범위는 내려받은 뒤 확정 | Hugging Face `Real-IAD/Real-IAD`, 저자 승인 필요, 라이선스 확인 필요 |

데이터는 레포에 올리지 않는다. 받는 순서는 [`data/README.md`](data/README.md)에 있다.

## 방법과 설정

방법은 두 가지다. 둘 다 ImageNet으로 사전학습한 CNN을 고정한 채 특징만 꺼내 쓰고, 새로 학습하는 파라미터는 없다.

- **PaDiM**: 조각(patch) 위치마다 정상 특징의 평균과 공분산(가우시안)을 저장하고, 시험 조각은 마할라노비스 거리로 점수를 매긴다.
- **PatchCore**: 정상 조각 특징을 메모리 뱅크에 모으고 탐욕 코어셋으로 일부만 남긴다. 시험 조각은 뱅크에서 가장 가까운 거리로 점수를 매긴다.

Q1에서 비교하는 설정은 12개다.

| 방법 | 백본 | 층 | 그 밖 | 개수 |
|---|---|---|---|---:|
| PatchCore | ResNet18, WRN-50-2 | (2,3), (1,2,3) | 코어셋 1%, 10% | 8 |
| PaDiM | ResNet18, WRN-50-2 | (1,2,3), (2,3) | 차원 수는 원 논문대로 (R18 100, WRN 550) | 4 |

코어셋 시작점과 PaDiM 차원 고르기에 무작위성이 있으므로 설정마다 시드 3개로 돌리고 평균과 표준편차를 적는다. 정확한 값은 [`configs/experiments.yaml`](configs/experiments.yaml)에 있다.

**전처리.** PatchCore 원 논문은 256으로 줄인 뒤 가운데 224를 자른다. 이렇게 자르면 가장자리에 있는 결함이 잘려 나갈 수 있다(원 논문 부록 C.4에 그런 시험 이미지가 하나 적혀 있다). VisA와 Real-IAD에서는 물체가 가운데 있지 않은 경우가 있어서 이 문제가 더 커질 수 있다. 그래서 두 단계로 나눈다.
- 재현 단계(MVTec만): 원 논문과 같이 256 -> 224 가운데 자르기
- 비교 단계(세 데이터셋): 셋 다 256 x 256으로 줄이기만 하고 자르지 않음

## 진행 순서

1. **재현.** MVTec AD에서 직접 구현한 결과가 원 논문 값과 맞는지 확인한다. 맞지 않으면 다음 단계로 가지 않는다.
2. **Q1.** 비교 단계 전처리로 설정 12개 x 데이터셋 3개 x 시드 3개를 돌린다.
3. **Q2.** 방법마다 Q1에서 가장 좋았던 설정 하나로 학습 장수를 줄여 본다.
4. **정리.** 결과 표, 그림, 한계를 적는다.

### 재현 목표 (MVTec AD 15 부류 평균)

| 방법 | 이미지 AUROC | 픽셀 AUROC | PRO | 출처 |
|---|---:|---:|---:|---|
| PatchCore-10%, WRN-50-2 (2,3) | 99.0 | 98.1 | 93.5 | PatchCore 논문 표 1 · 2 · 3 |
| PatchCore-1%, WRN-50-2 (2,3) | 99.0 | 98.0 | 93.1 | 같은 곳 |
| PaDiM, WRN-50-2 | 95.3 | 97.5 | 92.1 | PatchCore 논문 표 1 · 2 · 3에 옮겨 적힌 값 |

판정 기준은 이미지 AUROC가 목표값에서 ±0.5 안, 픽셀 AUROC와 PRO는 ±1.0 안이면 재현된 것으로 본다. 이 폭은 시드 3개의 흩어짐을 본 뒤 다시 정한다.

## 지표

- **이미지 AUROC**: 이미지 점수로 정상과 이상을 가를 때의 ROC 곡선 아래 넓이. 부류마다 재서 평균한다.
- **픽셀 AUROC**: 같은 넓이를 픽셀 하나하나를 표본으로 잰 것.
- **PRO**: 결함 영역을 연결 성분으로 나눠 영역마다 겹침을 재는 분할 지표. MVTec AD 평가 관례대로 거짓 양성률 0.3까지 적분하고 0~1로 정규화한다.
- **Kendall 순위 상관 (Q1)**: 두 데이터셋에서 설정 12개의 순위가 얼마나 같은지 나타내는 값. 1이면 순위가 똑같고 0이면 관계가 없다. 설정이 12개뿐이라 이 값 하나로는 불확실성이 크다. 그래서 순위 표를 같이 싣는다.

## 결과

아직 없다.

## 한계

아직 없다. 실험을 돌린 뒤 채운다.

## 폴더

```
configs/experiments.yaml   실험 범위(설정, 시드, 학습 장수)
data/README.md             데이터 받는 법 (데이터 자체는 올리지 않음)
src/                       구현 (만들 예정)
notebooks/                 Colab 실행 노트북 (만들 예정)
results/                   결과 CSV와 그림
```

## 실행 환경

Google Colab Pro, NVIDIA L4 GPU. 패키지는 `requirements.txt`.

## 참고 문헌

- Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (PatchCore), CVPR 2022.
- Defard et al., *PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization*, ICPR Workshops 2021.
- Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- Zou et al., *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation* (VisA), ECCV 2022.
- Wang et al., *Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection*, CVPR 2024.
