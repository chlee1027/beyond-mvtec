# 데이터 받는 법

데이터는 레포에 올리지 않는다(`.gitignore`가 `data/` 아래 이 파일만 남긴다). 받은 뒤 아래 구조로 둔다.

```
data/
  mvtec_ad/<부류>/{train,test,ground_truth}/...
  visa/<부류>/...           + split_csv/1cls.csv
  real_iad/<부류>/...       + realiad_jsons/
```

## MVTec AD

1. MVTec 홈페이지의 MVTec AD 페이지에서 이용 조건에 동의하고 내려받는다.
2. 라이선스는 홈페이지 기준 CC BY-NC-SA 4.0이다(비상업). 논문에는 라이선스가 적혀 있지 않다.
3. Grid·Screw·Zipper는 흑백(단일 채널)이다. 읽을 때 채널을 세 번 복사한다.

## VisA

1. GitHub `amazon-science/spot-diff` 의 안내대로 내려받는다. 논문에 적힌 `amazon-research/spot-diff` 는 이 저장소로 넘어간다.
2. 같은 저장소의 공식 분할 CSV(`1cls`)를 쓴다. 논문은 「부류마다 정상 90% 학습, 남은 정상 10%와 이상 전부 시험」이라고만 적고 반올림 방식과 시드는 적지 않았다. 그래서 장수는 CSV로 확정한다.
3. 저장소 코드의 라이선스는 Apache-2.0이다. 데이터 라이선스는 내려받을 때 확인해서 여기 적는다.
4. 부류별 해상도가 논문에 없다(「4,000 x 6,000 센서」뿐이다). 받은 뒤 부류마다 해상도를 재서 여기 적는다.

## Real-IAD

1. Hugging Face 계정으로 `Real-IAD/Real-IAD` 데이터셋 페이지에서 접근을 신청한다. 저자 승인이 나야 내려받을 수 있다. 이 레포의 실행에서는 받아 둔 부류 zip(`MyDrive/Real_IAD/<부류>.zip`)과 분할 JSON(`MyDrive/realiad_jsons.zip`)을 쓴다. `data.prepare_real_iad`가 zip에서 한 시점만 푼다.
2. 원본은 한 변이 2,000~5,000px이다. 줄인 판(256·512·1024)과 시점 이름 C1~C5는 anomalib 문서의 표기이고, 논문에는 없다(논문은 원본을 직접 줄였고 시점을 No.1~No.5로 부른다). 받아 둔 zip이 어느 해상도인지는 `notebooks/00_check_visa_realiad.ipynb`에서 재서 여기 적는다. 어느 판이든 비교 단계에서 `Resize((256, 256))`으로 줄인다.
3. 논문의 single-view 설정은 위(top view) 한 시점이다. C1이 위 시점인지 같은 노트북에서 사진으로 확인한다.
4. 채점은 표본 라벨이 아니라 이미지 라벨로 한다. 이상 표본 안에도 결함이 안 보이는 시점의 사진이 있다.
5. 라이선스는 데이터셋 페이지에서 확인해서 여기 적는다.
6. 읽기 코드는 JSON 항목의 `anomaly_class`, `image_path`, `mask_path`를 쓴다. 파일은 이름으로 찾고, JSON에 있는 파일이 하나라도 없으면 오류를 낸다. 지금까지 가짜 파일로만 시험했으므로 실제 JSON과 맞는지는 위 노트북에서 처음 확인한다.
