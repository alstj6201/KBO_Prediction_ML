# KBO_Prediction_ML

## 프로젝트 소개
한양대학교 데이터사이언스학부 머신러닝1 프로젝트

KBO(한국 프로야구) 경기 결과를 머신러닝 기반으로 예측하는 프로젝트입니다.  
크롤링부터 데이터 전처리, 모델링, 해석, 자연어 해설 생성까지 전 과정을 통합한 서비스 파이프라인을 구현했습니다.

## 주요 기능

- Statiz에서 KBO 데이터를 크롤링하여 최신 경기 데이터를 확보
- 최근 경기 성적, 롤링 스탯 등을 반영한 예측용 입력 데이터 생성
- 4개의 머신러닝 모델을 활용한 경기 결과 예측 (Deep Learning, Logistic Regression, XGBoost, Random Forest)
- SHAP 및 Coefficient를 활용한 주요 feature 해석
- GPT API를 활용한 자연어 해설문 자동 생성
- Streamlit 기반 웹앱 제공

---

## 파일 구성

### 📂 크롤링 및 데이터 구축

- **Crawling.ipynb**  
  Statiz 사이트에서 경기 결과 데이터를 BeautifulSoap을 이용해 크롤링합니다.
  
- **Data Preprocessing.ipynb**  
  크롤링한 원시 데이터를 모델 학습에 적합하도록 전처리합니다. (결측치 처리, 파생 변수 생성 등)

- **Pick Data.ipynb**  
  Confusion Matrix와 다중공선성을 사용해 모델 학습에 사용할 최종 feature를 선택하고, 학습용 데이터셋을 생성합니다.

---

### 📂 모델링 및 서비스 코드

- **data_loader.py**  
  - 경기일, 홈팀, 원정팀을 입력받아 예측용 데이터 행 생성
  - 최근 5경기 승패, 롤링 평균 스탯 등 계산하여 결합
  
- **predictor.py**  
  - 4개 모델별 예측 함수 제공 (DeepLearning, Logistic, XGB, RF)
  - 입력 데이터를 정규화, 인코딩, 피처 정렬 후 예측 확률 반환
  
- **shap_explainer.py**  
  - 각 모델별 중요 feature 5개 추출
  - Logistic 모델은 Coefficient 사용, 나머지는 SHAP (Tree/Kernal Explainer) 활용
  - feature_columns 기준으로 컬럼 정렬하여 일관성 유지
  
- **gpt_summary.py**  
  - GPT API 호출을 통해 자연어 해설문 자동 생성
  - 예측 확률, 모델명, 주요 feature를 기반으로 프롬프트 작성

- **app.py**  
  - Streamlit 기반 메인 웹앱
  - 모델 선택 → 경기 선택 → 예측 수행 → SHAP 해석 → GPT 해설 생성 전과정 통합

---

## 사용 모델

- Logistic Regression
- Random Forest
- XGBoost
- Deep Learning (Fully Connected Neural Network)

---

## 기술 스택

- Python (pandas, numpy, scikit-learn, xgboost, tensorflow, keras 등)
- Selenium (웹 크롤링)
- SHAP (모델 해석)
- OpenAI GPT API (자연어 해설)
- Streamlit (웹 서비스)

---

## 프로젝트 흐름도

1️⃣ 데이터 크롤링  
2️⃣ 데이터 전처리 및 피처 엔지니어링  
3️⃣ 모델 학습 및 평가  
4️⃣ SHAP을 이용한 피처 중요도 추출  
5️⃣ GPT를 활용한 자연어 해설 생성  
6️⃣ Streamlit 웹앱 통합 서비스 제공
