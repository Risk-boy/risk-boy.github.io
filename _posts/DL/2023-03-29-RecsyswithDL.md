---
title: "Recommender System with Deep Learning(1)"
categories:
  - DL
tags:
  - [RecSys, deep learning]

date: 2023-03-29
last_modified_at: 2023-03-29
---

# 6강 Recommender System with Deep Learning (1)

## 1. Recommender System with DL

### 1.1 추천 시스템과 딥러닝

**추천 시스템에서 딥러닝을 활용하는 이유**

1. Nonlinear Transformation
   -  Deep Neural Network(DNN)은 data의 non-linearity를 효과적으로 나타낼 수 있음
   -  복잡한 user-item interaction pattern을 효과적으로 모델링하여 user의 선호도를 예측 가능!
2. Representation Learning
   - DNN은 raw data로부터 feature representation을 학습해 사람이 직접 feature design하지 않아도 됨
   - 텍스트, 이미지, 오디오 등 다양한 종류의 정보를 추천 시스템에 활용 가능!
3. Sequence Modeling
   - DNN은 자연어처리, 음성 신호 처리 등 sequential modeling task에서 성공적으로 적용됨
   - 추천 시스템에서 next-item prediction, session-based recommendation에 사용 가능!
4. Flexibility
   - Tensorflow, PyTorch 등 다양한 DL 프레임워크 사용 가능!

## 2. Recommender System with MLP

### 2.1 Multi-Layer Perceptron

**다층 퍼셉트론(Multi-Layer Perceptron**

- 퍼셉트론으로 이루어진 layer 여러 개를 순차적으로 이어 놓은 feed-forward neural network

- 선형 분류만으로 풀기 어려웠던 문제를 비선형적으로 풀 수 있게됨

### 2.2 Neural Collaborative Filtering

**Neural Collaborative Filtering**

MF(Matrix Factorization)의 한계를 지적하여 신경망 기반의 구조를 사용해 더욱 일반화된 모델을 제시한 논문

**아이디어**

**Matrix Factorization의 한계**

- user와 item embedding의 선형 조합(linear combination)을 구함

- user와 item 사이의 복잡한 관계를 표현하는 것에 한계를 가짐

- 차원을 늘려서 해결가능하지만 overfitting 문제가 발생!



**모델(MLP 파트)**

- Input Layer
  - one-hot encoding된 user(item) vector: $v_{u},\  v_i$
- Embedding Layer
  - user(item) latent vector: $P^Tv_u,\  Q^Tv_i$
- Neural CF Layers
  - $\phi_X(\cdot \cdot \cdot\phi_2(\phi_1(P^Tv_u,Q^Tv_i))\cdot \cdot \cdot)$
  - $\phi_x$: $x$번째 neural network
  - 각각의 user와 item을 concatenate 해준 뒤 layer를 쌓아나감
- Output Layer
  - target을 0 또는 1을 예측하는 형태로 구성하기 위해 activation function으로는 Logistic이나 Probit함수 사용
  - user와 item사이의 관련도
    - $\hat{y}_{ui}=\phi_{out}(\phi_X(\cdot\cdot\cdot\phi_2(\phi_1(P^Tv_u,Q^Tv_i))\cdot\cdot\cdot)) , \hat{y}_{ui}\in[0, 1]$

**모델(최종 모델)**

Neural Matrix Factorization

- GMF와 MLP를 앙상블(ensemble)하여 사용 -> 각각의 장점은 살리고 단점은 보완

  *GMF: Generalization Matrix Factorization

- GMF와 MLP는 서로 다른 embedding layer를 사용
  $$
  \begin{aligned}
  &\phi^{GMF}=(p_u^{G})^Tq_i^G \\ &\phi^{MLP}=\phi_X(\cdot\cdot\cdot\phi_2(\phi_1(p_u^M,q_i^M))\cdot\cdot\cdot) \\
  &\hat{y}_{u,i} = \sigma(h^T\begin{bmatrix}\phi^{GMF} \\\phi^{MLP} \end{bmatrix}) 
  \end{aligned}
  $$

**결과**

MovieLens, Pinterest 데이터셋에 대하여 NCF의 추천 성능이 기존 MF(BPR)나 MLP모델보다 높음!

성능향상이 크게 되었다기 보다는 MLP를 기존 MF에 처음 추가한 것에 의의가 있음!

### 2.3 YouTube Recommendation

**Deep Neural Networks for YouTube Recommendations**

딥러닝 기반 추천 시스템을 실제 유튜브 서비스에 적용한 논문



**유튜브 추천 문제 특징**

- Scale
  - 엄청 많은 유저와 아이템과 제한된 컴퓨팅 파워로 인해 효율적인 서빙과 이에 특화된 추천 알고리즘이 필요
- Freshness
  - 잘 학습된 컨텐츠와 새로 업로드 된 컨텐츠를 실시간으로 적절히 조합해야 함(exploration / exploitation)
- Noise
  - 높은 Sparsity, 다양한 외부 요인으로 유저의 행동을 예측하기 어려움
  - Implicit Feedback, 낮은 품질의 메타데이터를 잘 활용해야함



**전체 구조: 2단계 추천 시스템**

1. Candidate Generation
   - High Recall이 목표
   - 주어진 사용자에 대해 Top N 추천 아이템 생성
   - 수백만개에서 수백개의 item을 뽑아줌
2. Ranking
   - 유저, 비디오 피쳐를 좀 더 풍부하게 사용
   - 스코어를 구하고 최종 추천 리스트를 제공



**Candidate Generation**

**정의**

extreme multiclass classification

특정시간(t)dp 유저 U가 C라는 context를 가지고 있을 때 각각의 비디오(i)를 볼 확률을 계산

수백만개의 비디오가 존재하기 때문에 <span style="color:red; font-weight:bold">Extreme</span>

마지막에는 Softmax function을 사용하는 분류 문제

$$
P(w_t=i|U,C)=\frac{e^{v_{i}u}}{\sum_{j\in V}e^{v_{j}u}}
$$

**모델**

**Watch Vector and Search Vector**

- 과거 시청 이력과 검색 이력을 각각 임베딩
- 마지막 검색어가 너무 큰 힘을 갖지 않도록 평균을 냄

**Demographic & Geographic features**

- 성별, 연령 등의 인구 통계학 정보와 거주지, 접속 위치 등의 지리적 정보를 피쳐로 포함

**"Example Age" features**

- 모델이 과거 인기 데이터 위주로 편향되어 학습되는 문제
- 최근 혹은 과거의 정보를 example age라는 값으로 구성
- 시청 로그가 학습 시점으로부터 경과한 정도를 피쳐에 포함
- Bootstrapping 현상 방지 및 Freshness 제고!

**순서**

- 다양한 피쳐 벡터들을 한번에 concatenate
- n개의 dense layer들을 거쳐 User Vector 생성
- 최종 output layer는 비디오를 분류하는 softmax function

**Serving**

- 유저를 input으로 하여 상위 N개의 비디오를 추출

- 학습 후에 유저 벡터($u$)와 모든 비디오 벡터($v_j$)의 내적을 계산
  
  $$
  P(w_t=i|U,C)=\frac{e^{v_{i}u}}{\sum_{j\in V}e^{v_{j}u}}
  $$

- Annoy, Faiss 같은 ANN 라이브러리를 사용하여 빠르게 서빙

- ANN 라이브러리: 주어진 user vector와 가장 유사한 item vector들을 찾아주는 라이브러리임

**Ranking**

**정의**

- CG단계에서 생성한 비디오 후보들을 input으로 하여 최종 추천될 비디오들의 순위를 매기는 문제
- Logistic 회귀를 사용하는 기본적인 방법
  - 딥러닝 모델로 유저, 비디오 feature들을 풍부하게 사용하여 정확한 랭킹 스코어를 구해냄
- loss function에 단순한 클릭 여부가 아닌 **시청 시간**을 가중치로 한 값(weighted logistic)을 반영

**모델**

- user actions feature 사용
  - 유저가 특정 채널에서 얼마나 많은 영상을 보았는지
  - 유저가 특정 주제의 동영상을 본 지 얼마나 지났는지
  - 영상의 과거 시청 여부 등을 입력

- DL 구조보다는 도메인 전문가의 역량이 좌우하는 파트
  - 많은 Feature Selection / Engineering이 필요!

- 네트워크를 통과한 뒤 비디오가 실제로 시청될 확률로 매핑
  
  $$
  P(watch)\in [0, 1]
  $$

  *시청 여부만을 맞히는 CTR을 예측

- Loss Function
  - 단순 binary가 아닌 weighted cross-entropy loss사용
  - 비디오 시청 시간을 가중치로 줌
  - 낚시성/광고성 콘텐츠를 업로드하는 어뷰징(abusing)을 감소시키는 효과



**요약 및 결과**

- 딥러닝 기반 2단계 추천을 처음으로 제안한 논문
  - Candidate Generation: 유저에게 적합한 수백개의 후보 아이템 생성
  - Ranking: 더 풍부한 피쳐를 사용하여 최종 추천 아이템 10~20개를 제공
- Candidate Generation: 기존 CF 아이디어를 기반으로 다양한 피쳐를 사용해 추천 성능 향상
  - 유저: watch / query history / demographic / geographic 
  - 아이템: Example Age
- Ranking: 과거에 많이 사용된 선형 / 트리 기반 모델보다 제안 딥러닝 모델이 더 뛰어난 성능을 보여줌
  - Rich Feature: CG에서 사용한 피쳐 외에 더 많은 피쳐를 사용하여 Ranking
  - 단순 CTR 예측이 아닌 Expected Watch Time을 예측

## 3. Recommender System with AE

### 3.1 Autoencoder

**오토인코더(Autoencoder, AE)**

입력 데이터를 출력으로 복원(reconstruct)하는 비지도(unsupervised)학습 모델

- 중간 hidden layer를 input의 feature representation으로 활용
- 원래 기존의 이미지를 최대한 비슷하게 복원하는 task에서 시작
- 실제 적용 분야를 찾다보니 노이즈를 없애주는 task에서 좋은 성능을 보임
- 주요 활용 분야: Anomaly Detection, Representation Learning, Image Denoising Task



**디노이징 오토인코더(Denoising Autoencoder, DAE)**

- 입력 데이터에 random noise나 dropout을 추가하여 학습

- noisy input을 더 잘 복원할 수 있는 robust한 모델이 학습되어 전체적인 성능 향상

- overfitting 에 대한 부분을 어느정도 해결함으로써 generalization performance를 향상 시킴!



### 3.2 AutoRec

**AutoRec: Autoencoders Meet Collaborative Filtering**

AE를 CF에 적용하여 기본 CF 모델에 비해 Representation과 Complexity 측면에서 뛰어남을 보인 논문(2015)



**아이디어**

- Rating Vector를 입력과 출력으로하여 Encoder & Decoder Reconstruction 과정을 수행
  - **유저 또는 아이템 벡터를 저차원의 latent feature로 나타내** 이를 사용해 평점을 예측
  - Autoencoder의 representation learning을 유저와 아이템에 적용한 것
- MF와 비교
  - MF는 linear, low-order interaction을 통한 representation이 학습되지만
  - AutoRec은 non-linear activation function을 사용하므로 더 복잡한 interaction 표현 가능



**모델**

- 아이템과 유저 중, 한 번에 하나에 대한 임베딩을 진행
- $r^{(i)}$: 아이템 $i$의 Rating Vector
- $R_{ui}:$ 유저 $u$의 아이템 $i$에 대한 Rating
- $V$: 인코더 가중치 행렬, $W$: 디코더 가중치 행렬 



**학습**

- 기존의 rating과 reconstructed rating의 RMSE를 최소화하는 방향으로 학습
- 관측된 데이터에 대해서만 역전파 및 파라미터 업데이트 진행
- $S$: 점수 벡터 $r$의 집합, $f, g:$ 활성 함수(Sigmoid, Identity Function)
  
  $$
  \min_{\theta}\sum_{r\in S}||r-h(r;\theta)||_2^2 \\ 
  h(r;\theta) = f(W\cdot g(Vr + \mu) + b)
  $$



**결과**

- 무비렌즈(ML)과 넷플릭스 데이터셋에서 RBM, MF 등의 모델보다 좋은 성능
- Hidden unit의 개수가 많아질 수록 RMSE가 감소함을 보임
- 본 논문 이후 고급 AE 기법을 CF에 활용한 후속 연구들이 나옴 / DAE, VAE 



### 3.3 CDAE

**Collaborative Denoising Auto-Encoders for Top-N Recommender Systems**

Denoising Autoencoder를 CF에 적용하여 Top-N 추천에 활용한 논문



**모델 특징**

- AutoRec과의 차이점
  - AutoRec은 Rating Prediction을 위한 모델
  - CDAE는 Ranking을 통해 유저에게 Top-N 추천을 제공하는 모델
- 문제 단순화를 위해 유저-아이템 상호작용 정보를 이진정보로 바꿔서 학습데이터로 사용
  - 개별 유저에 대해서 아이템의 rating이 아닌 **preference**를 학습하게 됨!



**문제 정의 및 모델**

- AutoRec과 다르게 DAE를 사용하여 noise 추가

  $P(\tilde{y}_u=\delta y_u)=1-q, \ P(\tilde{y}_u=0)=q$

  ($\tilde{y}_u$는 $q$의 확률에 의해 0으로 drop-out된 벡터)

- 개별 유저에 대해서 $V_u$를 학습(Collaborative)
  - 유저에 따른 특징을 해당 파라미터가 학습하고 Top N 추천에 사용
- 인코더로 latent representation $z_u$를 생성하고 디코더로 regenerate

  $z_u=h(W^{\top} \tilde{y}_u + V_u + b)  \ \ \ \ \ \ \  \tilde{y}_{ui}=f(W_i^{'\top}z_u + b_i^{'})$

- $\tilde{y}_u$ 를 사용해서 input값을 noise하게 바꿈
- $V_u$를 추가하여 유저별 특징을 학습하게 하였음



**결과 및 요약**

대체적으로 N에 관계없이 다른 top-N 추천 모델에 비해 더 높은 MAP와 Recall을 보임!

