# KBO_Prediction_ML

1. 모델 불러오기 및 예측
models/ 폴더에 .pkl 또는 .h5 형식으로 저장

predictor.py에서 joblib/keras 통해 모델 불러오고 예측 함수 정의

2. CSV 기반 rolling 처리
data_loader.py에서 pandas로 데이터 불러오고, 사용자가 고른 팀에 대해 최근 N경기 rolling 처리

3. SHAP 분석
shap_explainer.py: 각 모델별 SHAP value 추출 및 중요 feature 상위 N개 뽑기

4. 문장 생성
gpt_summary.py: 중요 feature들을 prompt로 구성해서 OpenAI API 호출

API 키는 config.py에서 환경변수로 안전하게 관리 (os.environ["OPENAI_API_KEY"])

5. 보안
.env 파일에 API 키 저장, .gitignore에 .env 포함

config.py에서 dotenv로 불러오기

6. Streamlit 앱 구성
app.py에서 모델 선택, 팀 선택, 예측 결과 및 해설 문장 출력