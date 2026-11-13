# beyond-mvtec

**MVTec AD에서 고른 이상 탐지 설정이 다른 산업 데이터셋에서도 같은 순서로 통하는가.**

공장 외관 검사용 이상 탐지 모델은 정상 이미지만으로 학습한다. 백본, 특징을 꺼낼 층, 메모리 뱅크 크기 같은 설정은 보통 MVTec AD 결과를 보고 고른다. 그런데 세 데이터셋은 촬영 조건이 다르다.

- **MVTec AD**는 촬영 장비와 조명을 통제해 찍었다. 물체 부류 가운데 Toothbrush·Capsule·Pill은 원문 표현으로 「roughly aligned」이고, Metal Nut·Screw·Hazelnut은 무작위로 회전되어 있다(MVTec AD 논문 3절).
- **VisA**에는 한 이미지에 물체가 여러 개 든 부류가 4개 있다. Capsules는 약 20개, Candle·Macaroni1·Macaroni2는 4개씩이다. 논문이 물체 위치·자세 차이가 크다고 적은 부류는 Capsules·Macaroni2다.
- **Real-IAD**는 30 부류 151,050장 규모다. 한 물체를 다섯 방향에서 찍었고 원본 해상도가 2,000~5,000px이다. 생산 라인에서 찍은 이미지는 아니다. 실물 제품을 사서 결함을 손으로 만들고, 시제품 촬영 장치로 찍었다(Real-IAD 논문).

이 레포는 MVTec AD에서 매긴 설정 순위가 VisA와 Real-IAD에서도 유지되는지 잰다.

> **상태 (2026-11-13).** Q1이 끝났다. 설정 10개 x 세 데이터셋(37 부류) x 시드 3개 = 1,110번을 빈칸 없이 돌렸다. 이미지 AUROC 순위의 Kendall 순위 상관은 데이터셋 쌍마다 0.51~0.78이고, MVTec 1위 설정이 VisA에서는 5위다. Q2도 끝났다. 설정 넷 x 37 부류 x 시드 3개 x 학습 50장·10장 = 888번이다. PatchCore 두 설정의 순서는 데이터셋마다 학습 장수를 줄여도 그대로였고, 10장에서 이미지 AUROC가 떨어진 폭은 데이터셋마다 달랐다(PatchCore (2,3) 10%: MVTec 5.59, VisA 6.87, Real-IAD 8.69점). 결과는 아래 「결과」에 있다. 관련 논문 여섯 편의 분석 노트는 [`papers/notes/`](papers/notes/)에 있다.
>
> > (2026-11-13 오전 상태) Q1이 끝났다. Q2는 아직 돌리지 않았다.
>
> > (2026-11-08 상태) 1단계 재현이 끝났다. MVTec 15 부류 x 시드 3개에서 판정한 11칸이 모두 판정 폭 안이다(아래 「재현 결과」). VisA·Real-IAD 파일은 읽기 코드로 다 읽혔다. Q1·Q2는 아직 돌리지 않았다.
>
> > (2026-11-06 상태) 구현과 재현 노트북까지 끝났다. MVTec 15 부류 재현은 아직 돌리지 않았고 결과 표는 비어 있다.

---

## 질문

| 번호 | 질문 | 재는 것 |
|---|---|---|
| Q1 (주) | MVTec AD에서 정한 설정 순위가 VisA, Real-IAD에서도 같은가 | 설정 10개의 이미지 AUROC 순위, 데이터셋 쌍마다 Kendall 순위 상관 |
| Q2 (보조) | 정상 학습 이미지를 줄일 때 떨어지는 폭이 데이터셋마다 다른가 | 학습 장수(전체 / 50 / 10)별 이미지 AUROC |

Q2는 새 제품을 라인에 막 넣어 정상 샘플이 적은 상황을 흉내 낸다. MVTec AD는 부류마다 학습 장수가 60장(Toothbrush)에서 391장(Hazelnut)까지다. 그래서 「50장」은 Toothbrush에서는 전체의 83%이고 Hazelnut에서는 13%다.

## 데이터셋

| 데이터셋 | 내용 | 이 레포에서 쓰는 범위 | 받는 곳 · 라이선스 |
|---|---|---|---|
| MVTec AD | 15 부류. 학습 3,629장(정상만), 시험 1,725장. 정사각 700~1024px. Grid·Screw·Zipper는 흑백 | 15 부류 전부 | MVTec 홈페이지. CC BY-NC-SA 4.0(홈페이지 기준이고 논문에는 적혀 있지 않다) |
| VisA | 12 부류. 정상 9,621장, 이상 1,200장. 4,000 x 6,000 센서로 촬영 | 12 부류 전부, 공식 1cls 분할(부류마다 정상 90%로 학습하고, 남은 정상 10%와 이상 전부로 시험) | `amazon-science/spot-diff`. 저장소 코드는 Apache-2.0이고 데이터 라이선스는 확인 필요 |
| Real-IAD | 30 부류. 정상 99,721장, 이상 51,329장. 시점 5개 | 10 부류(시드 0으로 뽑음), 위 시점(C1) 하나, 단일 시점 분할 `realiad_jsons_sv` | Hugging Face `Real-IAD/Real-IAD`. 저자 승인 필요, 라이선스 확인 필요 |

데이터는 레포에 올리지 않는다. 받는 순서는 [`data/README.md`](data/README.md)에 있다.

**Real-IAD를 쓰는 방식.** 논문의 single-view 설정은 30 부류 모두에서 위(top view) 한 시점만 쓴다. 이 레포도 같은 시점을 쓴다. 부류는 10개만 쓴다(Q1 계산 시간, 아래 「Q1 설정」). 30 부류 이름을 정렬해 `random.Random(0).sample(이름, 10)`으로 결과를 보기 전에 뽑았다: Bottle Cap, PCB, Phone Battery, Porcelain Doll, Regulator, SIM Card Set, Switch, USB, Wooden Beads, Woodstick. 그래서 30 부류 평균이 아니라 부록 표 5에서 이 열 부류 행과 맞댄다. 세 가지를 더 지킨다.
- 시점 이름 C1~C5는 anomalib 문서의 표기이고 논문에는 없다(논문은 No.1~No.5). Audiojack 표본 하나의 다섯 시점 사진에서 C1이 위에서 내려다본 사진이었다. 다른 부류는 사진으로 보지 않았다.
- 분할은 `realiad_jsons_sv`를 쓴다. anomalib 문서는 이 폴더를 「Single-view metadata」, `realiad_jsons`를 「Base metadata (multi-view)」라 적는다. 30 부류 모두 `realiad_jsons_sv`의 항목 수가 C1 사진 수와 같다(합 30,210). 두 분할은 같은 C1 사진을 학습·시험에 다르게 나눈다. 학습 장수 합이 단일 시점 분할 10,408장, 다중 시점 분할에서 C1만 고르면 7,293장이다. 단일 시점 분할에 C1 아닌 항목이 있으면 읽기 코드가 오류를 낸다.
- 이상 표본 안에서도 결함이 안 보이는 시점의 사진은 정상이다. 그래서 표본 라벨이 아니라 이미지 라벨로 채점한다.

## 방법과 설정

방법은 두 가지다. 둘 다 ImageNet으로 사전학습한 CNN을 고정한 채 특징만 꺼내 쓰고, 새로 학습하는 파라미터는 없다.

- **PaDiM**: 조각(patch) 위치마다 정상 특징의 평균과 공분산(가우시안)을 저장하고, 시험 조각은 마할라노비스 거리로 점수를 매긴다.
- **PatchCore**: 정상 조각 특징을 메모리 뱅크에 모으고 탐욕 코어셋으로 일부만 남긴다. 시험 조각은 뱅크에서 가장 가까운 거리로 점수를 매긴다.

**무엇을 직접 짰는가.**

| 부분 | 출처 |
|---|---|
| PatchCore | 원 저자 공개 코드 [`amazon-science/patchcore-inspection`](https://github.com/amazon-science/patchcore-inspection)(Apache-2.0)을 커밋 `fcaa92f`로 고정해 쓴다. 원본 파일은 고치지 않고 [`patchcore_adapter.py`](src/beyond_mvtec/patchcore_adapter.py)에서 바깥으로 끼운다. 바꾸는 것은 다섯이다. (1) 백본 목록에 resnet18 추가, (2) 이미지 점수와 분할 지도 저장(원본은 `results.csv`만 남긴다), (3) Q2용 50장·10장 표본 고르기(원본 `--train_val_split`은 정렬된 파일 앞에서부터 자른다), (4) VisA·Real-IAD 데이터 읽기, (5) 아래 「전처리」의 정사각 줄이기. 계산 값은 그대로 두고 GPU 메모리 쓰는 법만 하나 바꾼다. 코어셋 샘플러가 128차원 사영을 기울기 계산을 끄지 않고 해서, 코어셋 전 뱅크(VisA pcb3, 층 (1,2,3)에서 15.2 GB)가 GPU에 묶여 남았다. Q1 첫 실행에서 바로 뒤 FAISS의 GPU 메모리 요청이 이것 때문에 실패해서, 사영을 기울기 없이 하고 코어셋 뒤 PyTorch 캐시를 비운다. 같은 조각을 고르는 것은 시험으로 확인했다 |
| PaDiM | 저자 공식 코드가 없어서 직접 구현했다([`padim.py`](src/beyond_mvtec/padim.py)). 원 논문이 정한 값은 그대로 쓰고, 정하지 않은 값은 아래 「PaDiM에서 이 레포가 고른 값」에 적었다 |
| 지표 (이미지·픽셀 AUROC, PRO) | 두 방법이 같은 코드로 재도록 [`metrics.py`](src/beyond_mvtec/metrics.py) 하나를 쓴다. 원본 PatchCore 코드는 PRO를 계산하지 않는다 |
| 데이터 읽기 | 세 데이터셋의 파일 목록과 전처리는 [`data.py`](src/beyond_mvtec/data.py), 설정 목록·실행·판정은 [`experiment.py`](src/beyond_mvtec/experiment.py)에 있다 |

### Q1 설정

Q1에서 비교하는 설정은 10개다.

| 방법 | 백본 | 층 | 그 밖 | 개수 |
|---|---|---|---|---:|
| PatchCore | ResNet18, WRN-50-2 | (2,3) | 코어셋 1%, 10% | 4 |
| PatchCore | ResNet18, WRN-50-2 | (1,2,3) | 코어셋 1% | 2 |
| PaDiM | ResNet18, WRN-50-2 | (1,2,3), (2,3) | 줄인 차원 R18 100, WRN 550 | 4 |

> (2026-11-06 계획) 설정 12개, PatchCore는 백본 2 x 층 2 x 코어셋 2 = 8개.

- **층 (1,2,3)의 10%를 뺀 이유 (2026-11-08).** 계산 시간이다. 원본 PatchCore는 조각 격자를 첫 층에 맞춘다. 그래서 256 입력에서 (2,3)은 이미지 하나에 32 x 32 = 1,024 조각이고 (1,2,3)은 64 x 64 = 4,096 조각이다. 탐욕 코어셋은 한 번 고를 때마다 뱅크의 조각 N개 전부와 거리를 재고 (비율 x N)번 고르므로, 시간이 N의 제곱에 비례한다(원본 `sampler.py`를 읽은 것).
  - 재현 CSV에서 PatchCore-10%와 1%의 학습 시간 차를 (0.10 − 0.01) x N²으로 나눈 계수가 14 부류에서 5.3~5.6 x 10⁻⁹초였다(N = 163,856~306,544). 뱅크가 가장 작은 Toothbrush(N = 47,040)만 7.0 x 10⁻⁹초다. 아래 추정은 15 부류의 중앙값 5.5 x 10⁻⁹초를 쓴다.
  - 이 계수로 늘려 잡으면 VisA `pcb3`(학습 905장, N = 370만)에서 (1,2,3) 10% 한 번이 약 2.1시간이고, 12개 설정 x 세 데이터셋(Real-IAD 30 부류) x 시드 3개가 약 230시간이다. 10개 설정에 Real-IAD 10 부류로는 약 35시간이다. 잰 범위(N 31만까지)보다 최대 12배 큰 뱅크로 늘려 잡은 추정이라, [`02_q1_compare.ipynb`](notebooks/02_q1_compare.ipynb)에서 가장 무거운 칸을 먼저 잰다.
    > (2026-11-08 추정) 10개 설정에 Real-IAD 10 부류로 약 35시간. -> 2026-11-13 실제로 잰 합은 38.9시간이다(학습·예측·지표, 모델 적재 제외). 먼저 잰 가장 무거운 칸(VisA `pcb3`, WRN-50-2 (1,2,3) 1%, 시드 0)은 학습 843초, GPU 메모리 최대 17.4 GB였다. 뺀 (1,2,3) 10%의 2.1시간은 돌리지 않아 추정으로 남는다.
  - 그래서 PatchCore의 층 비교((2,3) 대 (1,2,3))는 코어셋 1%에서만 한다.

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
- 비교 단계(세 데이터셋): 256 x 256으로 줄이기만 하고 자르지 않는다. torchvision의 `Resize(256)`은 짧은 변만 맞추고 비율을 지키므로, 정사각이 아닌 이미지에 `CenterCrop`을 붙이면 긴 변이 잘린다. 그래서 `Resize((256, 256))`으로 두 변을 다 맞춘다. VisA처럼 정사각이 아닌 이미지는 가로세로비가 바뀐다. VisA 부류별 해상도는 1,274 x 1,176에서 1,562 x 960까지이고 가로/세로 비가 1.08~1.63이다([`data/README.md`](data/README.md)). Real-IAD 받아 둔 판은 1,024 x 1,024 정사각이다.

두 단계에 공통으로 적용하는 것도 있다.
- 흑백 이미지(MVTec Grid·Screw·Zipper)는 채널을 세 번 복사한다.
- 정답 마스크는 최근접 보간으로 줄인다. 원본 PatchCore 코드는 양선형 보간 뒤 정수로 바꾸는데, 그러면 마스크 가장자리가 줄어든다.

## 진행 순서

1. **재현.** MVTec AD에서 원 논문 값과 맞는지 확인한다. PatchCore는 원본 코드가 이 레포의 데이터 읽기·지표 코드를 거쳐도 같은 값을 내는지, PaDiM은 직접 구현이 맞는지를 본다. 맞지 않으면 다음 단계로 가지 않는다.
2. **Q1.** 비교 단계 전처리로 설정 10개 x 데이터셋 3개 x 시드 3개를 돌린다.
3. **Q2.** Q1 1위 설정을 데이터셋마다 모은 넷으로 학습 장수를 50장, 10장으로 줄여 본다. PatchCore는 WRN-50-2 (2,3) 10%(MVTec 1위)와 (1,2,3) 1%(Real-IAD·VisA 1위), PaDiM은 WRN-50-2 (2,3)(MVTec·Real-IAD 1위)과 (1,2,3)(VisA 1위)이다. 전체 장수 칸은 Q1 결과를 쓴다. 노트북은 [`03_q2_few_shot.ipynb`](notebooks/03_q2_few_shot.ipynb)다.
   > (2026-11-08 계획) 방법마다 Q1에서 가장 좋았던 설정 하나로 학습 장수를 줄여 본다. -> Q1에서 1위가 데이터셋마다 갈려서(위 「Q1 결과」) 방법마다 둘씩 쓴다.
4. **정리.** 결과 표, 그림, 한계를 적는다.

### 재현 목표 (MVTec AD 15 부류 평균)

| 방법 | 이미지 AUROC | 픽셀 AUROC | PRO | 출처 |
|---|---:|---:|---:|---|
| PatchCore-10%, WRN-50-2 (2,3) | 99.0 | 98.1 | 93.5 | PatchCore 논문 표 1 · 2 · 3 |
| PatchCore-1%, WRN-50-2 (2,3) | 99.0 | 98.0 | 93.1 | 같은 곳 |
| PaDiM, WRN-50-2 | 95.3 | 97.5 | 92.1 | PaDiM 논문 표 V(이미지) · 표 III(픽셀, PRO) |
| PaDiM, ResNet18 (차원 100) | 논문에 없음 | 96.7 | 90.1 | PaDiM 논문 표 III |

판정 기준: 이미지 AUROC는 목표값 ±0.5 안, 픽셀 AUROC와 PRO는 ±1.0 안이면 재현된 것으로 본다. 판정은 결과를 보기 전에 정한 이 폭으로 했고, 결과를 본 뒤 바꾸지 않았다. 판정할 때 지킬 것이 둘 있다.
- **PatchCore 원본 코드는 논문과 값이 다르다.** 원본 코드 README가 보고한 WRN-50-2 10%(시드 0) 값은 99.2 · 98.1이다. 부류 값도 논문과 다르고, Screw에서는 1.77점 차이가 난다. 코드 README의 PRO 열(94.4)은 논문 93.5와 0.9 다르다. 코드의 PRO는 이 레포 정의와 다른 계산일 수 있다. 그래서 PRO가 어긋나면 정의부터 맞춰 본다.
- **부류별로 대조할 때.** Toothbrush는 시험이 정상 12장, 결함 30장이라 (정상, 결함) 쌍 하나가 뒤집히면 이미지 AUROC가 0.28점 움직인다.

### 재현 결과 (2026-11-08)

Colab L4 GPU, faiss GPU 검색. MVTec 15 부류 x 시드 3개 = 설정마다 45번. 시드마다 15 부류 평균을 내고, 그 세 값의 평균과 표준편차(n−1)를 적는다. 단위는 %다. 부류·시드별 값은 [`results/reproduce_mvtec.csv`](results/reproduce_mvtec.csv)에 있다.

| 설정 | 이미지 AUROC | 픽셀 AUROC | PRO | 판정 |
|---|---:|---:|---:|---|
| PatchCore-10%, WRN-50-2 (2,3) | 99.00 ± 0.06 (0.00) | 98.11 ± 0.01 (+0.01) | 92.97 ± 0.03 (−0.53) | 셋 다 폭 안 |
| PatchCore-1%, WRN-50-2 (2,3) | 98.87 ± 0.10 (−0.13) | 97.95 ± 0.02 (−0.05) | 92.31 ± 0.07 (−0.79) | 셋 다 폭 안 |
| PaDiM, WRN-50-2 (차원 550) | 95.32 ± 0.63 (+0.02) | 97.38 ± 0.09 (−0.12) | 92.39 ± 0.19 (+0.29) | 셋 다 폭 안 |
| PaDiM, ResNet18 (차원 100) | 92.33 ± 0.37 (목표 없음) | 96.34 ± 0.34 (−0.36) | 90.25 ± 0.40 (+0.15) | 판정한 둘 다 폭 안 |

괄호 안은 목표값과의 차이다.

- **PatchCore PRO는 판정 폭 안이지만 두 설정 모두 논문보다 낮다(−0.53, −0.79).** 이 차이는 시드 간 표준편차(0.03, 0.07)의 10배가 넘어 시드 흩어짐으로 설명되지 않는다. 같은 지표 코드에서 PaDiM PRO는 +0.29, +0.15다. 원인은 찾지 못했다.
- **GPU와 CPU의 차이.** 시드 0의 Bottle·Screw를 「구현 확인」의 CPU 값과 맞대면 PaDiM은 여덟 칸 모두 0.001점 이내다. PatchCore는 열한 칸이 0.1점 이내이고 PatchCore-10% Screw 이미지 AUROC 한 칸만 −0.29점(GPU 98.48, CPU 98.77)이다. PatchCore 쪽 차이가 어디서 오는지는 확인하지 않았다.
- **시간.** 학습·예측·지표 계산 시간을 CSV에서 더하면 PatchCore-10% 29.1분, PatchCore-1% 14.6분, PaDiM WRN 12.7분, PaDiM R18 5.7분, 합 62.1분이다(설정마다 45번). 모델 적재와 데이터 읽기 준비는 들어 있지 않다. GPU 메모리 최대는 PaDiM WRN이 14.3 GB, 나머지는 1.8 GB 이하다.
- 실행 환경은 Python 3.13, torch 2.14.1+cu130, torchvision 0.29.1+cu130, faiss-gpu-cu12 1.14.1, 레포 커밋 `981e1f6`이다.

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

- **PatchCore PRO가 논문보다 0.85~1.42점 낮다. 원인은 찾지 못했다.** 같은 지표 코드에서 PaDiM은 논문과 -0.51~+0.37점 차이다. 저장한 지도로 마스크 처리(최근접, 원본식)와 이미지 범위(전체, 결함 이미지만)를 조합한 네 가지를 다시 재 봤지만 네 칸 모두 논문보다 낮았다.
  > (2026-11-06에 적은 할 일) 15 부류 평균이 판정 폭 ±1.0을 넘을 수 있으므로 재현 때 먼저 본다. -> 2026-11-08 재현에서 15 부류 평균은 −0.53, −0.79로 폭 안이었다(위 「재현 결과」).
- **Screw 이미지 AUROC는 논문보다 1.0~1.77점 높다.** 원본 실행기도 같은 값을 내므로 이 레포 경로의 차이가 아니다.
- **마스크 처리가 픽셀 AUROC를 바꾼다.** 원본은 Screw 99.41·99.52, Bottle 98.47·98.48(1%·10%)이다. 이 레포 마스크(최근접)로는 위 표의 값이 된다. 이 레포 지도에 원본식 마스크를 씌워 다시 재면 원본 값이 나와 원인이 마스크임을 확인했다. 원본식에서는 결함 픽셀 수가 Screw 68.7%, Bottle 92.9%로 준다. 논문과의 차이는 최근접이 0.08점 이내, 원본식이 0.21점 이내다.

시험은 `tests/`에 66개 있고 모두 통과한다. PRO는 문턱마다 직접 세는 계산과 맞댄다.

> (2026-11-06에 적은 미확인 목록) GPU와 faiss GPU 경로, VisA·Real-IAD 실제 파일. -> GPU와 faiss GPU 경로는 2026-11-08 재현에서 돌았다. VisA 12 부류와 Real-IAD 30 부류(C1)는 [`notebooks/00_check_visa_realiad.ipynb`](notebooks/00_check_visa_realiad.ipynb)에서 읽기 코드로 다 읽혔고, JSON에 있는 파일 가운데 빠진 것이 없었다. 학습은 아직 돌리지 않았다. -> 2026-11-13 Q1에서 VisA 12 부류와 Real-IAD 10 부류의 학습·평가까지 돌았다(아래 「결과」).

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
- **Kendall 순위 상관 (Q1)**: 두 데이터셋에서 설정 10개의 순위가 얼마나 같은지 나타내는 값. 1이면 순위가 똑같고 0이면 관계가 없다. 동률을 고려하는 tau-b를 쓴다. 설정 10개면 비교하는 쌍이 45개뿐이라 이 값 하나로는 불확실성이 크다. 그래서 순위 표와 뒤집힌 쌍 수, 시드 하나씩 따로 낸 값을 같이 싣는다.
  - 예로, Real-IAD 논문 표 2의 방법 일곱을 이미지 AUROC 순위로 비교하면 MVTec–Real-IAD 사이 Kendall 순위 상관이 0.62다(21쌍 중 4쌍이 뒤집힘, 내가 계산한 값). 이것은 방법 사이의 순위이고, 이 레포가 재는 설정 사이의 순위가 아니다.

## 결과

1단계 재현은 위 「재현 결과」에 있다. Q1·Q2 결과는 아래에 있다.

> (2026-11-13 오전에 적은 문장) Q2 결과는 아직 없다. -> 같은 날 Q2 결과를 아래 「Q2 결과」에 적는다.

> (2026-11-08에 적은 문장) Q1·Q2 결과는 아직 없다. -> 2026-11-13 Q1 결과를 아래에 적는다.

### Q1 결과 (2026-11-13)

Colab L4 GPU, 비교 단계 전처리(256 x 256, 자르지 않음), 레포 커밋 `88fd4dc`. 설정 10개 x 37 부류(MVTec 15, VisA 12, Real-IAD 10) x 시드 3개 = 1,110번이다. 시드마다 부류 평균을 내고, 그 세 값의 평균과 표준편차(n−1)를 적는다. 단위는 %다. 부류·시드별 값은 [`results/q1_mvtec_ad.csv`](results/q1_mvtec_ad.csv), [`results/q1_real_iad.csv`](results/q1_real_iad.csv), [`results/q1_visa.csv`](results/q1_visa.csv)에 있고, 픽셀 AUROC·PRO·뱅크 크기·시간도 같은 파일에 있다.

**이미지 AUROC와 데이터셋 안 순위** (괄호 안이 순위, 1 = 가장 높음)

| 설정 | MVTec AD | Real-IAD | VisA |
|---|---:|---:|---:|
| PatchCore WRN-50-2 (2,3) 10% | 98.55 ± 0.04 (1) | 92.56 ± 0.03 (2) | 93.14 ± 0.25 (5) |
| PatchCore WRN-50-2 (2,3) 1% | 98.37 ± 0.13 (2) | 91.83 ± 0.06 (4) | 93.64 ± 0.22 (3) |
| PatchCore WRN-50-2 (1,2,3) 1% | 98.20 ± 0.05 (3) | 93.55 ± 0.06 (1) | 96.60 ± 0.09 (1) |
| PatchCore ResNet18 (2,3) 1% | 97.01 ± 0.07 (4) | 89.03 ± 0.26 (6) | 93.23 ± 0.03 (4) |
| PatchCore ResNet18 (2,3) 10% | 96.87 ± 0.08 (5) | 90.39 ± 0.20 (5) | 93.10 ± 0.07 (6) |
| PaDiM WRN-50-2 (2,3) 550 | 95.47 ± 0.22 (6) | 88.67 ± 0.26 (7) | 87.77 ± 0.61 (8) |
| PaDiM WRN-50-2 (1,2,3) 550 | 95.15 ± 0.99 (7) | 88.03 ± 0.42 (8) | 87.96 ± 0.52 (7) |
| PatchCore ResNet18 (1,2,3) 1% | 95.01 ± 0.25 (8) | 92.11 ± 0.00 (3) | 95.85 ± 0.19 (2) |
| PaDiM ResNet18 (2,3) 100 | 91.56 ± 1.39 (9) | 83.41 ± 0.45 (9) | 84.46 ± 0.76 (9) |
| PaDiM ResNet18 (1,2,3) 100 | 90.96 ± 0.42 (10) | 81.62 ± 0.56 (10) | 83.60 ± 0.13 (10) |

Real-IAD PatchCore ResNet18 (1,2,3) 1%의 표준편차 0.00은 반올림한 값이다. 부류별 값은 시드마다 달랐고(예: Switch 90.65·90.82·90.08), 시드 평균이 92.110·92.117·92.113으로 0.007점 안에 들었다.

**순위 상관** (이미지 AUROC, Kendall tau-b, 설정 10개 = 비교하는 쌍 45개, 동률 없음)

| 데이터셋 쌍 | 뒤집힌 쌍 | tau-b | p | 시드 하나씩 낸 tau-b (0 · 1 · 2) |
|---|---:|---:|---:|---|
| MVTec AD – Real-IAD | 8 | 0.644 | 0.009 | 0.689 · 0.600 · 0.600 |
| MVTec AD – VisA | 11 | 0.511 | 0.047 | 0.600 · 0.511 · 0.644 |
| Real-IAD – VisA | 5 | 0.778 | 0.001 | 0.822 · 0.733 · 0.867 |

쌍이 45개이고 동률이 없으면 쌍 하나가 뒤집힐 때마다 tau-b가 2/45 = 0.044씩 준다.

- **세 쌍 모두 순위가 같은 방향이지만, MVTec AD가 낀 두 쌍이 새 데이터셋끼리의 쌍보다 낮다.** 시드 하나씩 낸 값으로도 MVTec AD가 낀 쌍은 0.511~0.689, Real-IAD – VisA는 0.733~0.867이라 두 범위가 겹치지 않는다.
- **1위가 갈린다.** MVTec AD 1위인 PatchCore WRN-50-2 (2,3) 10%는 VisA에서 5위다. Real-IAD·VisA 1위는 PatchCore WRN-50-2 (1,2,3) 1%이고, MVTec AD에서는 3위다. MVTec AD에서 둘의 차이는 0.35점이다.
- **층 비교의 부호가 바뀐다.** PatchCore 코어셋 1%에서 (1,2,3)에서 (2,3)을 뺀 이미지 AUROC는 아래와 같다. PaDiM은 같은 비교에서 여섯 칸 가운데 다섯이 음수이고(VisA WRN-50-2만 +0.19), 이 부호 바뀜이 없다. 왜 바뀌는지는 재지 않았다.

  | 백본 | MVTec AD | Real-IAD | VisA |
  |---|---:|---:|---:|
  | ResNet18 | −2.00 | +3.08 | +2.62 |
  | WRN-50-2 | −0.17 | +1.72 | +2.96 |

  그래서 PatchCore ResNet18 (1,2,3) 1%가 MVTec AD 8위에서 Real-IAD 3위, VisA 2위로 오른다.
- **코어셋 10%와 1%의 차이(층 (2,3))는 부호가 데이터셋과 백본에 따라 갈린다.** 10%에서 1%를 빼면 MVTec AD −0.14 · +0.18, Real-IAD +1.36 · +0.73, VisA −0.13 · −0.50이다(ResNet18 · WRN-50-2 순서). 두 칸 모두 양수인 것은 Real-IAD뿐이다.
- **VisA의 PatchCore (2,3) 넷은 93.10~93.64, 폭 0.54점 안에 있다.** 시드 표준편차가 0.03~0.25라서, 이 넷 사이의 순위(3~6위)에는 시드 흩어짐과 같은 자릿수의 차이가 들어 있다.
- **PaDiM 네 설정은 Real-IAD·VisA에서 모든 PatchCore 설정보다 낮다.** MVTec AD에서만 PatchCore ResNet18 (1,2,3) 1%(95.01)가 PaDiM WRN-50-2 둘(95.47, 95.15)보다 낮다.

**다른 지표로 낸 순위 상관.** 같은 표를 픽셀 AUROC와 PRO로 매기면 tau-b가 이렇게 바뀐다. Q1의 주 지표는 이미지 AUROC로 미리 정했고, 아래는 참고로만 싣는다.

| 데이터셋 쌍 | 이미지 AUROC | 픽셀 AUROC | PRO |
|---|---:|---:|---:|
| MVTec AD – Real-IAD | 0.644 (p 0.009) | 0.467 (p 0.073) | 0.556 (p 0.029) |
| MVTec AD – VisA | 0.511 (p 0.047) | 0.644 (p 0.009) | 0.511 (p 0.047) |
| Real-IAD – VisA | 0.778 (p 0.001) | 0.556 (p 0.029) | 0.778 (p 0.001) |

픽셀 AUROC의 MVTec AD – Real-IAD만 p가 0.05를 넘는다. 어느 쌍이 가장 낮은지도 지표에 따라 갈린다.

**참고값과 대조.** VisA PatchCore WRN-50-2 (2,3)은 93.64(1%) · 93.14(10%)이고, VisA 논문 표 6의 PatchCore WRN-50은 92.4다. VisA PaDiM WRN-50-2는 87.77 · 87.96이고, VisA 논문 표 2의 PaDiM ResNet-50은 87.8이다. 두 값 모두 조건이 이 레포와 다르다(위 「다른 데이터셋의 참고값」). Real-IAD는 30 부류 평균만 참고값으로 적어 두어서, 부록 표 5의 열 부류 행과 맞대는 일은 아직 안 했다.

**MVTec AD 값이 재현 결과와 다른 이유.** Q1은 비교 단계 전처리(자르지 않음)를 쓴다. 그래서 같은 PatchCore WRN-50-2 (2,3) 10%가 재현 단계(256 -> 224 가운데 자르기)에서는 99.00, Q1에서는 98.55다.

**시간과 메모리.** 학습·예측·지표 시간을 더하면 MVTec AD 5.9시간, Real-IAD 10.2시간, VisA 22.8시간, 합 38.9시간이다. 모델 적재와 데이터 복사는 들어 있지 않다. 한 번에 가장 오래 걸린 설정은 VisA의 PatchCore (1,2,3) 1%로, 평균 11.0분(ResNet18 10.9분, WRN-50-2 11.0분)이다. PaDiM은 한 번에 0.2~0.9분이다. GPU 메모리 최대는 PaDiM WRN-50-2 (1,2,3)의 17.8 GB, PatchCore WRN-50-2 (1,2,3)의 17.4 GB다. 실행 환경은 Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, faiss 1.14.1이다. 실행 기록은 [`notebooks/02_q1_compare.ipynb`](notebooks/02_q1_compare.ipynb)의 출력에 있다.

### Q2 결과 (2026-11-13)

조건은 Q1과 같다(Colab L4 GPU, 비교 단계 전처리). 레포 커밋은 `dd716bc`다. Q1에서 1위가 데이터셋마다 갈려서, 방법마다 1위였던 설정 둘씩 모두 넷을 썼다(위 「진행 순서」). 설정 4 x 37 부류 x 시드 3 x 학습 장수 2(50, 10) = 888번이다. 전체 장수 칸은 Q1 결과를 그대로 쓴다. 시드마다 학습 목록에서 중복 없이 뽑았고, 같은 시드의 10장은 50장 안에 든다. 시드는 고른 사진과 코어셋 시작점(PaDiM은 무작위 차원)을 함께 바꾸므로, 표준편차에 둘이 섞여 있다. 부류·시드별 값은 [`results/q2_mvtec_ad.csv`](results/q2_mvtec_ad.csv), [`results/q2_real_iad.csv`](results/q2_real_iad.csv), [`results/q2_visa.csv`](results/q2_visa.csv)에 있다. 단위는 %다.

**이미지 AUROC** (시드 3개 평균 ± 표준편차)

| 설정 | 데이터셋 | 전체 | 50장 | 10장 |
|---|---|---:|---:|---:|
| PatchCore WRN-50-2 (2,3) 10% | MVTec AD | 98.55 ± 0.04 | 96.83 ± 0.41 | 92.95 ± 0.46 |
| | Real-IAD | 92.56 ± 0.03 | 88.98 ± 0.18 | 83.87 ± 0.37 |
| | VisA | 93.14 ± 0.25 | 90.33 ± 0.14 | 86.27 ± 0.86 |
| PatchCore WRN-50-2 (1,2,3) 1% | MVTec AD | 98.20 ± 0.05 | 95.96 ± 0.24 | 90.30 ± 0.49 |
| | Real-IAD | 93.55 ± 0.06 | 91.44 ± 0.19 | 84.46 ± 0.44 |
| | VisA | 96.60 ± 0.09 | 93.47 ± 0.22 | 87.48 ± 1.11 |
| PaDiM WRN-50-2 (2,3) 550 | MVTec AD | 95.47 ± 0.22 | 90.80 ± 0.95 | 83.90 ± 2.28 |
| | Real-IAD | 88.67 ± 0.26 | 82.69 ± 0.15 | 74.17 ± 0.72 |
| | VisA | 87.77 ± 0.61 | 84.12 ± 1.58 | 78.09 ± 0.28 |
| PaDiM WRN-50-2 (1,2,3) 550 | MVTec AD | 95.15 ± 0.99 | 90.24 ± 0.79 | 84.24 ± 1.67 |
| | Real-IAD | 88.03 ± 0.42 | 81.23 ± 0.59 | 72.45 ± 0.43 |
| | VisA | 87.96 ± 0.52 | 83.35 ± 1.40 | 76.45 ± 1.28 |

**떨어진 폭** (같은 시드의 전체 장수 값 − N장 값, 시드 3개 평균 ± 표준편차, %p)

| 설정 | MVTec 50 | MVTec 10 | Real-IAD 50 | Real-IAD 10 | VisA 50 | VisA 10 |
|---|---:|---:|---:|---:|---:|---:|
| PatchCore (2,3) 10% | 1.72 ± 0.43 | 5.59 ± 0.48 | 3.58 ± 0.21 | 8.69 ± 0.35 | 2.81 ± 0.27 | 6.87 ± 1.09 |
| PatchCore (1,2,3) 1% | 2.24 ± 0.25 | 7.90 ± 0.44 | 2.11 ± 0.20 | 9.08 ± 0.47 | 3.13 ± 0.14 | 9.12 ± 1.05 |
| PaDiM (2,3) | 4.67 ± 0.77 | 11.57 ± 2.13 | 5.98 ± 0.26 | 14.50 ± 0.97 | 3.65 ± 0.99 | 9.68 ± 0.49 |
| PaDiM (1,2,3) | 4.91 ± 0.24 | 10.91 ± 1.06 | 6.81 ± 0.22 | 15.59 ± 0.45 | 4.61 ± 0.90 | 11.51 ± 0.92 |

- **PatchCore 두 설정의 순서는 데이터셋마다 학습 장수를 줄여도 그대로다.** MVTec AD에서는 (2,3) 10%가, Real-IAD·VisA에서는 (1,2,3) 1%가 전체·50장·10장 모두에서 위다. 시드 하나씩 봐도 27칸(데이터셋 3 x 장수 3 x 시드 3) 모두 같은 부호다. 차이의 크기는 바뀐다. (2,3) 10%에서 (1,2,3) 1%를 빼면 MVTec AD +0.35 · +0.87 · +2.65, Real-IAD −0.99 · −2.46 · −0.59, VisA −3.46 · −3.14 · −1.21이다(전체 · 50장 · 10장 순서).
- **10장에서 떨어진 폭은 데이터셋마다 다르다.** PatchCore (2,3) 10%는 MVTec AD 5.59, VisA 6.87, Real-IAD 8.69점이다. 넷 가운데 셋이 Real-IAD에서 가장 많이 떨어진다. 나머지 PatchCore (1,2,3) 1%는 Real-IAD 9.08, VisA 9.12점으로 0.04점 차이이고, 시드 표준편차는 0.47, 1.05다. PaDiM은 10장에서 9.68~15.59점 떨어지고, PatchCore는 5.59~9.12점 떨어진다.
- **같은 「50장」이 데이터셋마다 전체의 다른 몫이다.** 부류당 학습 장수가 MVTec AD 60~391장, Real-IAD 267~477장, VisA 450~905장이라, 50장은 각각 전체의 13~83%, 10~19%, 6~11%이고 10장은 2.6~17%, 2.1~3.7%, 1.1~2.2%다. 그래서 떨어진 폭의 차이에는 데이터셋의 성질과 남긴 몫이 함께 들어 있다. 둘을 나눠 보지 않았다.
- **PaDiM 두 설정의 순서는 두 칸에서만 뒤집히고, 그 두 칸의 차이가 시드 표준편차보다 작다.** (2,3)에서 (1,2,3)을 빼면 MVTec AD +0.32 · +0.56 · −0.34, Real-IAD +0.64 · +1.46 · +1.72, VisA −0.19 · +0.77 · +1.64다(전체 · 50장 · 10장 순서). 음수인 MVTec AD 10장과 VisA 전체의 시드 표준편차는 1.67~2.28, 0.52~0.61이다.
- **픽셀 지표로는 PatchCore 순서가 바뀐다.** 10장에서 픽셀 AUROC는 세 데이터셋 모두 (2,3) 10%가 위다(MVTec AD 96.48 대 95.53, Real-IAD 97.94 대 97.65, VisA 97.61 대 95.90). 전체 장수에서는 Real-IAD·VisA에서 (1,2,3) 1%가 위였다. VisA PRO는 10장에서 (1,2,3) 1%가 11.77점, (2,3) 10%가 5.84점 떨어진다.

**참고값과 대조.** PatchCore 논문 표 S5(MVTec AD)는 학습 10장에서 PatchCore-10% 93.6, 다시 구현한 PaDiM 83.2이고, 50장에서 떨어진 폭이 1.5, 5.3이다. 이 레포는 PatchCore (2,3) 10%가 92.95와 1.72, PaDiM (2,3)이 83.90과 4.67이다. 논문은 가운데 224를 자르는 전처리라 조건이 같지 않다. 논문이 표본을 어떻게 골랐는지는 이 레포의 PatchCore 노트에서 확인하지 않았다.

**시간과 메모리.** 학습·예측·지표 시간을 더하면 MVTec AD 1.5시간, Real-IAD 4.2시간, VisA 1.6시간, 합 7.3시간이다. 시험 사진 수는 그대로라 예측·지표 시간이 학습 장수에 따라 줄지 않는다. Real-IAD는 시험이 6,501장이다. 코어셋 뒤 뱅크 크기는 PatchCore (1,2,3) 1%가 50장에서 2,048, 10장에서 409개이고, (2,3) 10%가 5,120, 1,024개다. GPU 메모리 최대는 PaDiM (1,2,3)의 17.7 GB다. 실행 기록은 [`notebooks/03_q2_few_shot.ipynb`](notebooks/03_q2_few_shot.ipynb)의 출력에 있다.

## 한계

Q1·Q2를 돌린 뒤 확인된 것이다.

> (2026-11-13 오전에 적은 문장) Q1을 돌린 뒤 확인된 것이다. Q2를 돌리면 더 적는다. -> 같은 날 Q2 몫을 아래 끝의 셋으로 더했다.

- **설정이 10개뿐이다.** 비교하는 쌍이 45개라 쌍 하나가 뒤집힐 때마다 tau-b가 0.044씩 움직인다. p값은 scipy `kendalltau`가 낸 것이다. 데이터셋 쌍 셋을 따로 검정했고 다중 비교 보정은 하지 않았다.
- **순위 일부는 시드 흩어짐과 같은 자릿수의 차이로 갈린다.** VisA의 PatchCore (2,3) 넷이 그렇다(위 「Q1 결과」). 그래서 시드 하나씩 낸 tau-b를 함께 싣는다.
- **결론이 지표에 따라 바뀐다.** 이미지 AUROC로는 세 쌍 모두 p < 0.05지만, 픽셀 AUROC로는 MVTec AD – Real-IAD가 p 0.073이다.
- **Real-IAD는 30 부류 가운데 10 부류, 위 시점(C1) 하나만 썼다.** 다른 부류나 시점에서도 같은 순위가 나오는지는 재지 않았다.
- **전처리를 하나로 고정했다.** 세 데이터셋 모두 256 x 256으로 줄였고, VisA는 가로세로비가 바뀐다(1.08~1.63). Real-IAD는 받아 둔 1,024px 판을 한 변 1/4로 줄인다. 해상도를 바꾸면 순위가 어떻게 되는지는 재지 않았다.
- **층 비교의 부호가 왜 바뀌는지 재지 않았다.** (1,2,3)은 조각 격자가 64 x 64이고 (2,3)은 32 x 32라서 결함 크기와 관련이 있을 수 있지만, 결함 크기별로 나눠 보지 않았다.
- **설정 공간이 일부다.** PatchCore (1,2,3) 10%는 계산 시간 때문에 뺐다(위 「Q1 설정」). 백본은 둘, 입력 크기는 하나다.
- **Q2는 학습 장수를 절대 장수(50, 10)로 정했다.** 데이터셋마다 전체에서 남기는 몫이 달라서(위 「Q2 결과」), 떨어진 폭의 차이를 데이터셋 성질 탓으로만 돌릴 수 없다. 같은 몫으로 줄이는 실험은 하지 않았다.
- **Q2의 설정은 Q1 결과를 보고 골랐다.** 방법마다 데이터셋별 1위를 모은 넷이다. 다른 설정이 적은 장수에서 더 나은지는 재지 않았다.
- **Q2 표본은 시드마다 한 벌이다.** 시드 셋이 고른 사진과 코어셋 시작점을 함께 바꾸므로, 어떤 사진을 고르느냐에 따른 흩어짐만 따로 재지 않았다.

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
notebooks/02_q1_compare.ipynb           2단계 Q1 (Colab)
notebooks/03_q2_few_shot.ipynb          3단계 Q2 (Colab)
results/                   결과 CSV와 그림 (reproduce_mvtec.csv: 1단계 재현 180줄,
                           q1_mvtec_ad.csv · q1_real_iad.csv · q1_visa.csv: Q1 450 · 300 · 360줄,
                           q2_mvtec_ad.csv · q2_real_iad.csv · q2_visa.csv: Q2 360 · 240 · 288줄)
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
