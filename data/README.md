# 데이터 받는 법

데이터는 레포에 올리지 않는다(`.gitignore`가 `data/` 아래 이 파일만 남긴다). 받은 뒤 아래 구조로 둔다.

```
data/
  mvtec_ad/<부류>/{train,test,ground_truth}/...
  visa/<부류>/...           + split_csv/1cls.csv
  real_iad/<부류>/...       + realiad_jsons/
  mvtec_ad_2/<부류>/{train,validation,test_public,test_private,test_private_mixed}/...
```

## MVTec AD

1. MVTec 홈페이지의 MVTec AD 페이지에서 이용 조건에 동의하고 내려받는다.
2. 라이선스는 홈페이지 기준 CC BY-NC-SA 4.0이다(비상업). 논문에는 라이선스가 적혀 있지 않다.
3. Grid·Screw·Zipper는 흑백(단일 채널)이다. 읽을 때 채널을 세 번 복사한다.

## VisA

1. GitHub `amazon-science/spot-diff` 의 안내대로 내려받는다. 논문에 적힌 `amazon-research/spot-diff` 는 이 저장소로 넘어간다.
2. 같은 저장소의 공식 분할 CSV(`1cls`)를 쓴다. 논문은 「부류마다 정상 90% 학습, 남은 정상 10%와 이상 전부 시험」이라고만 적고 반올림 방식과 시드는 적지 않았다. 그래서 장수는 CSV로 확정한다.
3. 저장소 코드의 라이선스는 Apache-2.0이다. 데이터 라이선스는 내려받을 때 확인해서 여기 적는다.
4. 부류별 해상도가 논문에 없다(「4,000 x 6,000 센서」뿐이다). 아래는 `notebooks/00_check_visa_realiad.ipynb`에서 잰 값이다(2026-11-08). 해상도는 부류마다 학습 이미지 앞 20장에서 가장 많은 크기다. 장수는 공식 1cls CSV를 따른 것이고 합이 학습 8,659장, 시험 2,162장이다.

   | 부류 | 학습 | 시험 (이상) | 가로 x 세로 |
   |---|---:|---:|---|
   | candle | 900 | 200 (100) | 1,284 x 1,168 |
   | capsules | 542 | 160 (100) | 1,500 x 1,000 |
   | cashew | 450 | 150 (100) | 1,274 x 1,176 |
   | chewinggum | 453 | 150 (100) | 1,342 x 1,118 |
   | fryum | 450 | 150 (100) | 1,500 x 1,000 |
   | macaroni1 | 900 | 200 (100) | 1,500 x 1,000 |
   | macaroni2 | 900 | 200 (100) | 1,500 x 1,000 |
   | pcb1 | 904 | 200 (100) | 1,404 x 1,070 |
   | pcb2 | 901 | 200 (100) | 1,404 x 1,070 |
   | pcb3 | 905 | 201 (100) | 1,562 x 960 |
   | pcb4 | 904 | 201 (100) | 1,358 x 1,104 |
   | pipe_fryum | 450 | 150 (100) | 1,300 x 1,154 |

5. 마스크는 0/255가 아니라 작은 정수 값이다. 부류마다 이상 이미지 하나의 마스크를 보니 여덟 부류는 {0, 1}이고, 네 부류(cashew·chewinggum·macaroni2·pcb4)에는 2·3·5·6·7 가운데 일부가 더 있었다. 읽기 코드는 0보다 큰 값을 모두 결함으로 본다.

## Real-IAD

1. Hugging Face 계정으로 `Real-IAD/Real-IAD` 데이터셋 페이지에서 접근을 신청한다. 저자 승인이 나야 내려받을 수 있다. 이 레포의 실행에서는 받아 둔 부류 zip(`MyDrive/Real_IAD/<부류>.zip`)과 분할 JSON(`MyDrive/realiad_jsons.zip`)을 쓴다. `data.prepare_real_iad`가 zip에서 한 시점만 푼다.
2. 원본은 한 변이 2,000~5,000px이다. 줄인 판(256·512·1024)과 시점 이름 C1~C5는 anomalib 문서의 표기이고, 논문에는 없다(논문은 원본을 직접 줄였고 시점을 No.1~No.5로 부른다). 받아 둔 zip은 1,024 x 1,024 판이다(30 부류마다 학습 첫 이미지를 잰 값, 2026-11-08). 비교 단계에서 `Resize((256, 256))`으로 줄인다.
3. 논문의 single-view 설정은 위(top view) 한 시점이다. 같은 노트북에서 Audiojack 표본 하나의 다섯 시점을 그려 보니 C1이 위에서 내려다본 사진이었다. 다른 부류는 사진으로 보지 않았다. 각 zip에는 시점마다 같은 수의 사진이 있다(Audiojack은 시점마다 1,010장).
4. 채점은 표본 라벨이 아니라 이미지 라벨로 한다. 이상 표본 안에도 결함이 안 보이는 시점의 사진이 있다.
5. 라이선스는 데이터셋 페이지에서 확인해서 여기 적는다.
6. 읽기 코드는 JSON 항목의 `anomaly_class`, `image_path`, `mask_path`를 쓴다. 파일은 이름으로 찾고, JSON에 있는 파일이 하나라도 없으면 오류를 낸다. 실제 항목은 `{"category", "anomaly_class", "image_path", "mask_path"}` 꼴이고, 정상은 `anomaly_class: "OK"`, `mask_path: null`이다. 30 부류 모두 빠진 파일 없이 읽혔고 이상 이미지마다 마스크가 있었다.
7. `realiad_jsons.zip` 안에는 폴더가 여섯 있다: `realiad_jsons`, `realiad_jsons_sv`, `realiad_jsons_fuiad_0.0`·`0.1`·`0.2`·`0.4`. 폴더마다 부류 JSON이 30개다. 이 레포는 단일 시점 분할 `realiad_jsons_sv`를 쓴다(`configs/experiments.yaml`의 `real_iad.split`). 고른 이유는 README의 「Real-IAD를 쓰는 방식」에 있다.

## MVTec AD 2

1. MVTec 홈페이지의 MVTec AD 2 페이지에서 이용 조건에 동의하고 내려받는다. 8 부류, 30.4 GB다. 라이선스는 CC BY-NC-SA 4.0으로 알고 있다. 데이터 폴더의 `license.txt`는 아직 읽지 않았다.
2. 부류마다 `train/good`, `validation/good`, `test_public/{good,bad,ground_truth/bad}`, `test_private`, `test_private_mixed`가 있다. 정답 마스크는 `test_public/ground_truth/bad/<번호>_<조건>_mask.png`이고 값은 0과 255다. 이 레포는 `train/good`과 `test_public`만 쓴다.
3. 파일 이름은 `<번호>_<조건>.png`이다. 조건은 학습이 regular뿐이고 test_public은 regular, overexposed, underexposed, shift_1~shift_4다.
4. Sheet Metal·Vial·Wallplugs는 흑백(단일 채널)이다. 해상도는 부류마다 하나다(2026-11-13, 폴더마다 세 장 확인).

   | 부류 | 학습 | test_public (결함) | 가로 x 세로 |
   |---|---|---|---|
   | Can | 412 | 162 (90) | 2232 x 1024 |
   | Fabric | 387 | 156 (90) | 2448 x 2048 |
   | Fruit Jelly | 263 | 80 (60) | 2100 x 1520 |
   | Rice | 313 | 132 (90) | 2448 x 2048 |
   | Sheet Metal | 137 | 114 (90) | 4224 x 1056 |
   | Vial | 291 | 140 (105) | 1400 x 1900 |
   | Wallplugs | 293 | 150 (90) | 2448 x 2048 |
   | Walnuts | 432 | 150 (90) | 2448 x 2048 |
