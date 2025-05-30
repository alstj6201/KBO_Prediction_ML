import streamlit as st
import pandas as pd
from utils.predictor import predict_model
from utils.data_loader import create_prediction_row
from utils.shap_explainer import explain_instance
from utils.gpt_summary import generate_explanation
from joblib import load
from tensorflow.keras.models import load_model


# app.py
def load_model_by_type(model_type):
    if model_type == 'DeepLearning':
        return load_model('models/deep_learning_model.h5')
    elif model_type == 'LogisticRegression':
        return load('models/logistic_model.pkl')
    elif model_type == 'XGBoost':
        return load('models/xgb_best_model.pkl')
    elif model_type == 'RandomForest':
        return load('models/rf_model.pkl')
    else:
        raise ValueError("지원하지 않는 모델 타입입니다.")


# Streamlit 앱
st.title("⚾ 2025년 6월 3일 KBO 경기 예측")

# 경기를 미리 정의
match_list = {
    "키움 vs 롯데": (3, 4),
    "삼성 vs SSG": (2, 5),
    "KIA vs 두산": (0, 1),
    "KT vs 한화": (7, 6),
    "LG vs NC": (8, 9)
}

# 모델 선택
model_type = st.selectbox("모델을 선택하세요", ['DeepLearning', 'LogisticRegression', 'XGBoost', 'RandomForest'])
model = load_model_by_type(model_type)

# 경기 선택
match = st.selectbox("경기를 선택하세요", list(match_list.keys()))
home_Team, away_Team = match_list[match]

# 예측 실행 버튼
if st.button("예측 실행하기"):

    # 1️⃣ 경기 데이터 생성
    prediction_row = create_prediction_row(GameDate='2025-06-03', home_Team=home_Team, away_Team=away_Team)

    # 2️⃣ 확률 예측
    probability = predict_model(prediction_row, model_type)
    win_team = 'home' if probability >= 0.5 else 'away'
    win_team_name = home_Team if win_team == 'home' else away_Team

    st.subheader("예측 결과")
    st.write(f"👉 {win_team_name} 승리 예상 (확률: {probability*100:.2f}%)")

    # 3️⃣ SHAP 해석
    top_features = explain_instance(model, prediction_row, model_type)

    st.subheader("중요 피처 (상위 5개)")
    for feature in top_features:
        st.write(f"- {feature}")

    # 4️⃣ GPT 해설문 생성
    team1_name = f"팀 {home_Team}"
    team2_name = f"팀 {away_Team}"
    pred_label = f"팀 {win_team_name}"

    gpt_result = generate_explanation(
        team1=team1_name,
        team2=team2_name,
        features=top_features,
        model_name=model_type,
        prediction=pred_label
    )

    st.subheader("GPT 해설")
    st.write(gpt_result)
