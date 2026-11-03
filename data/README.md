# 데이터 받는 법

데이터는 레포에 올리지 않는다(`.gitignore`가 `data/` 아래 이 파일만 남긴다). 받은 뒤 아래 구조로 둔다.

```
data/
  mvtec_ad/<부류>/{train,test,ground_truth}/...
  visa/<부류>/...           + split_csv/1cls.csv
  real_iad/<부류>/...
```

## MVTec AD

1. MVTec 홈페이지의 MVTec AD 페이지에서 이용 조건에 동의하고 내려받는다.
2. 라이선스는 CC BY-NC-SA 4.0이다(비상업). 결과 그림에 이미지를 싣는 것은 출처를 밝히면 된다.

## VisA

1. GitHub `amazon-science/spot-diff` 의 안내대로 내려받는다.
2. 같은 저장소의 공식 분할 CSV(`1cls`)를 쓴다.
3. 라이선스는 내려받을 때 확인해서 여기 적는다.

## Real-IAD

1. Hugging Face 계정으로 `Real-IAD/Real-IAD` 데이터셋 페이지에서 접근을 신청한다. 저자 승인이 나야 내려받을 수 있다.
2. 원본은 한 변이 2,000~5,000px이다. 이 레포는 줄인 판(256px)을 쓴다.
3. 승인에 걸리는 기간, 줄인 판 용량, 라이선스는 신청한 뒤 여기 적는다.
