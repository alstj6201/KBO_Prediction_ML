# KBO_Prediction_ML

✅ data_loader.py
경기일, 홈팀, 원정팀을 입력받아 예측용 데이터 1행 생성
최근 5경기 승패 및 롤링 평균, 최신 스탯을 계산해서 결합
최종적으로 real_final.csv 형식의 한 행을 만든다

✅ gpt_summary.py
GPT API를 호출해서 자연어 해설문 생성
예측 확률, 모델명, 주요 feature를 받아 프롬프트 작성
야구 해설가처럼 한국어로 예측 이유를 설명해준다

✅ make_feature_pickle.py
각 모델 학습용 X_train.csv에서 feature list 추출
feature_columns를 pkl로 저장 (예측시 피처 정렬용)
추후 모든 모델 예측/SHAP 해석에서 컬럼 순서 맞출 때 사용

✅ predictor.py
4개 모델 (DeepLearning, Logistic, XGB, RF)별로 예측 함수
prediction_row 전처리 (정규화+인코딩+피처정렬) 후 확률 반환
feature_columns pkl을 불러와서 항상 컬럼 안전하게 맞춘다

✅ shap_explainer.py
각 모델별로 SHAP 기반 중요 feature 5개 추출
Logistic은 coef, 나머진 SHAP(Tree/KernalExplainer) 사용
feature_columns로 reindex해 컬럼 불일치 문제 예방

✅ app.py
Streamlit 웹앱 메인
모델선택 → 경기선택 → 예측수행 → SHAP해석 → GPT해설 전과정 통합
전체 KBO 예측 서비스 파이프라인을 종합적으로 연결

✅ features_model.pkl
모델 특징 저장

✅ X_train_model.csv
모델에 사용된 X_train 저장(SHAP에서 사용하기 위함)