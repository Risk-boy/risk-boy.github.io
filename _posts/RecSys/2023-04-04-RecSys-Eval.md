---
title: "Recommendation System Evaluation"
categories:
  - RecSys
tags:
  - [RecSys, deep learning]


date: 2023-04-04
last_modified_at: 2023-04-10
---

# 1. 개요

새로 적용한 추천 시스템 혹은 추천 모델의 성능 평가는 어떻게 해야할까?

**비즈니스 / 서비스 관점**

- 추천 시스템 적용으로 인해 매출, PV(Page View), 구독 등의 증가
- 추천 아이템으로 인해 유저의 **CTR(Click-through rate)**의 상승

- CTR: 광고의 클릭 횟수 / 광고의 노출 횟수



**품질 관점**

- 연관성(Relevance): 추천된 아이템이 유저에게 관련이 있나, 실제 구매로 이어졌는지 여부
- 다양성(Diversity): 추천된 Top-K 아이템에 얼마나 다양한 아이템이 추천되는가 여부
- 새로움(Novelty): 얼마나 새로운 아이템이 추천되고 있는지 여부
- 참신함(Serendipity): 유저가 기대하지 못한 뜻밖의 아이템이 추천되고 있는지 여부



# 2. Offline Test

**새로운 추천 모델을 검증하기 위해 가장 우선적으로 수행되는 단계**

- 유저로부터 수집한 데이터를 train / valid / test 로 나누어 모델의 성능을 객관적인 지표로 평가
- 다양한 추천알고리즘을 쉽고 빠르게 평가할 수 있음
- 보통 offline test에서 좋은 성능을 보여야 online 서빙에 투입되지만, 수집된 데이터를 바탕으로 평가가 이루어지므로 실제 서비스 상황에서는 다양한 양상을 보임(**serving bias** 존재)
- **serving bias**: 유저의 온라인 활동에 따라 데이터가 추가됨으로써 모델이 계속 갱신됨에 따라 오프라인 테스트와는 다른 양상을 보임

**데이터 분할 방법**
- 추천 시스템은 데이터 분할 전략에 따라서 시스템 성능 평가에 큰 영향을 줄 수 있음!

1. Leave One Last
- 사용자당 마지막 구매를 Test set으로, 마지막에서 2번째를 Valid set으로 분할
- 장점
  - 학습 시 많은 데이터 사용 가능
- 단점
  - 사용자당 마지막 구매로만 평가하므로 전체적인 성능을 반영하기 어려움
  - 훈련 중 모델이 테스트 데이터 상호작용을 학습할 가능성이 있음 (Data leakage 문제)

2. Temporal User / Global Split
- 시간을 이용한 분할 전략
- Temporal User
  - 사용자 별로 시간 순서에 따라 일정 비율로 데이터 분할
  - Leave One Last와 유사
  - Data leakage 문제
- Temporal Global
  - 각 유저 간에 공유되는 시점을 고정하여 특정 시점 이후에 이루어진 모든 상호작용을 test set으로 분할
  - 학습 및 검증에 사용할 수 있는 상호작용이 적은 문제

3. Random Split
- 각 사용자 별로 interaction을 시간 순서에 관계없이 random하게 아이템을 선택하여 분할
- 사용하기 쉬움 & 원하는 비율로 test set을 구성할 수 있음
- Data leakage 문제

4. User Split
- 사용자가 겹치지 않게 사용자를 기준으로 분할
- Cold-start 문제에 대응하는 모델 생성 가능
- User-free 모델에만 사용가능
- Data leakage 문제

5. K-Fold Cross Validation
- Train set 내에서 k개의 Fold로 분할한 뒤에 한 Fold씩 Validation set으로 활용하여 검증
- 일반화와 Fold ensemble을 통한 더 높은 성능 기대
- 더 많은 resource가 필요
- time을 고려하지 않을 시 역시 data leakage의 가능성

6. Time Series Cross Validation
- 시계열 데이터의 경우 일반적인 CV 수행 시 미래 데이터를 학습하여 과거를 예측하는 문제가 있음
- Fold 분할 시 시간을 고려함
- Validation set을 고정 or 움직이며 사용
- Train data set이 줄어드는 문제


**성능 지표**

**평점 예측 문제**: RMSE, MAE
**랭킹 문제**: Precision@K, Recall@K, MAP@K, NDCG@K, Hit Rate

**평점 예측 문제**

**Root Mean Squared Error(RMSE)**

- 실제 값과 모델의 예측 값의 차이를 보는 것
- 예측 대상 값에 영향을 받음(Scale- dependent)

$$
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y_i})^2}
$$

- 평점 등 prediction problem의 추천 성능을 평가할 때 사용하는 metric
- RMSE가 낮을 수록 성능이 추천 알고리즘 성능이 더 좋다고 정량적으로 평가 가능
  - 성능이 좋다고 해서 꼭 좋은 추천을 하는 것은 아님!
  - 단순 RMSE지표를 따르기 보다는 예측값들과 실제 값들의 관계를 살펴보는 것도 필요!


**MAE(Mean Absolute Error)**

$$
MAE = \frac{1}{N}\sum_{i=1}^N|y_i-\hat{y}_i|
$$


**랭킹 문제**

**Precision@K**

- 우리가 추천한 K개 아이템 가운데 실제 유저가 관심있는 아이템의 비율


**Recall@K**

- 유저가 관심있는 전체 아이템 가운데 우리가 추천한 아이템의 비율


**AP@K**

- Precision@1부터 Precision@K까지의 평균값
- Precision@K와 달리, 관련 아이템을 더 높은 순위에 추천할수록 점수가 상승함
- 아이템의 순서가 점수에 반영됨

$$
AP@K = \frac{1}{m}\sum_{i=1}^KPrecision@i
$$

**MAP@K**

- 모든 유저에 대한 Average Precision 값의 평균

$$
MAP@K = \frac{1}{|U|}\sum_{u=1}^{|U|}(AP@K)_{u}
$$

**Precision@K, Recall@K, MAP@K 의 단점**

- 추천 또는 정보검색에서 특정 아이템에 biased된 경우
- 이미 유명하고 잘 알려진 인기있는 아이템 또는 한 명의 사용자에 의해서 만들어진 랭킹일 경우



<a href="https://medium.com/towards-data-science/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826" style="text-decoration:none; color:blue; ">Precision, Recall 에 대해 알아보기</a> 



**Normalized Discounted Cumulative Gain(NDCG)**

- 추천 시스템에 가장 많이 사용되는 지표 중 하나, 원래는 정보검색(Information Retrieval)에서 등장한 지표

- Precision@K, MAP@K 와 마찬가지로 Top-K 랭킹 리스트를 만들고 유저가 선호하는 아이템을 비교하여 값을 구함

- MAP@K와 마찬가지로 **추천의 순서에 가중치를 더 많이** 두어 성능을 평가하며 1에 가까울 수록 좋다

- MAP@K와 달리 연관성을 이진(binary)값이 아닌 수치로도 사용할 수 있기 때문에 유저에게 얼마나 더 관련 있는 아이템을 상위로 노출시키는지 알 수 있음

- 가장 이상적인 랭킹(정답 랭킹)과 현재 점수를 활용한 랭킹 사이의 점수를 cumulative 하게 비교
- $log_2i$로 normalization 하여 순위가 낮을 수록 가중치를 감소
- 검색엔진, 영상, 음악 등 컨텐츠 랭킹 추천에서 주요 평가지표로 활용

**NDCG Formula**

1. **Cumulative Gain**

- 추천된 상위 K개 아이템에 대하여 관련도를 합한 cumulative gain

- 관련도(Relevance): 유저가 추천된 아이템을 얼마나 선호하는 지의 지표

- 순서에 따라 discount 하지 않고 동일하게 더한 값

- 상위 아이템 K개에 대해서 동일한 비중으로 합함

$$
CG_{K} = \sum_{i=1}^{K}rel_{i}
$$



2. **Discount Cumulative Gain**

- 순서에 따라 Culmulative Gain을 Discount함
- 하위권 penalty 부여

$$
DCG_{K} = \sum_{i=1}^K\frac{rel_i}{log_{2}(i+1)}
$$



3. **Ideal DCG**

- 이상적인 추천이 일어났을 때의 DCG값

- 가능한 DCG값 중에 제일 큰 값

$$
IDCG=\sum_{i=1}^{K}\frac{rel_{i}^{opt}}{log_{2}(i+1)}
$$



4. **Normalized DCG**

- 추천 결과에 따라 구해진 DCG를 IDCG로 나눈 값
- 0 ~ 1의 값

$$
NDCG = \frac{DCG}{IDCG}
$$



**NDCG  예시**

NDCG@5 구하기

Ideal Order: [C(3), A(3), B(2), E(2), D(1)]

Relevance의 내림차순으로 정렬하여 추천하는 것이 이상적

Recommend Order: [E, A, C, D, B]
$$
DCG@5 = \frac{2}{log_{2}(1 + 1)} + \frac{3}{log_{2}(2 + 1)}+\frac{3}{log_{2}(3 + 1)} + \frac{1}{log_{2}(4 + 1)} + \frac{2}{log_{2}(5 + 1)} = 6.64
$$

$$
IDCG@5 =\frac{2}{log_{2}(1 + 1)} + \frac{3}{log_{2}(2 + 1)}+\frac{3}{log_{2}(3 + 1)} + \frac{2}{log_{2}(4 + 1)} + \frac{1}{log_{2}(5 + 1)} = 7.14
$$

$$
NDCG@5=\frac{DCG}{IDCG}=\frac{6.64}{7.14}=0.93
$$


**Offline 측정 지표의 문제점**
- 정확성 개선과 실제 시스템 성능 향상을 연관 짓기 어려움


**특성 평가 지표**

1. Coverage
전체 아이템 중 추천시스템이 추천하는 아이템의 비율

2. Novelty
얼마나 새롭고 흔하지 않은 추천이 가능한지 나타냄

3. Personalization
얼마나 개인화 된 추천을 하는지 나타냄

4. Serendipity
사용자가 기대하지 못했지만 만족스러운 뜻밖의 항목이 추천되는 지 나타냄

5. Diversity
얼마나 다양한 아이템이 추천하는지 나타냄


# 3 Online Test

**Online A/B Test**

- Offline Test에서 검증된 가설이나 모델을 이용해 실제 추천 결과를 서빙하는 단계
- 추천 시스템 변경 전후의 성능을 비교하는 것이 아니라, 동시에 대조군(A)과 실험군(B)의 성능을 평가
- 대조군과 실험군의 환경은 최대한 동일하게 유지
- 실제 서비스를 통해 얻어지는 결과를 통해 최종 의사결정이 이루어짐
- 실제 현업에서는 NDCG과 같은 지표보다는 매출, CTR등의 비즈니스/서비스 지표를 최종 의사결정에 이용
- 수집할 수 있는 데이터의 한계가 있으나 실제 사용자의 데이터이기 때문에 정확한 평가 가능

