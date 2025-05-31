import streamlit as st
import pandas as pd
from utils.predictor import predict_model
from utils.data_loader import create_prediction_row
from utils.shap_explainer import explain_instance
from utils.gpt_summary import generate_explanation
from joblib import load
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv

# 환경설정
st.set_page_config(page_title="KBO AI 승부 예측", page_icon="⚾", layout="centered")
load_dotenv()

# 팀 이름 ↔ 숫자 매핑
team_name_to_id = {
    "KIA": 0, "두산": 1, "삼성": 2, "키움": 3, "롯데": 4,
    "SSG": 5, "한화": 6, "KT": 7, "LG": 8, "NC": 9
}
team_id_to_name = {v: k for k, v in team_name_to_id.items()}

# 모델 로딩
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

# CSS 스타일 넣기 (깔끔한 카드 스타일)
st.markdown("""
    <style>
    .title { font-size: 40px; font-weight: bold; color: #2C3E50; text-align: center; margin-bottom: 20px; }
    .result-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 22px;
        text-align: center;
        font-weight: bold;
        color: #34495E;
        margin-bottom: 20px;
    }
    .feature-box {
        background-color: #ecf0f1;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        color: #2C3E50;
        font-size: 16px;
    }
    .gpt-box {
        background-color: #f1f8e9;
        padding: 20px;
        border-radius: 10px;
        font-size: 16px;
        color: #2C3E50;
        line-height: 1.6;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #34495E;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 페이지 타이틀
st.markdown("<div class='title'>⚾ 2025 KBO AI 승부 예측</div>", unsafe_allow_html=True)

# 경기 목록
match_list = {
    "키움 vs 롯데": ("키움", "롯데"),
    "삼성 vs SSG": ("삼성", "SSG"),
    "KIA vs 두산": ("KIA", "두산"),
    "KT vs 한화": ("KT", "한화"),
    "LG vs NC": ("LG", "NC")
}

# 모델 선택
st.markdown("<div class='section-header'>모델 선택</div>", unsafe_allow_html=True)
model_type = st.selectbox("", ['DeepLearning', 'LogisticRegression', 'XGBoost', 'RandomForest'])
model = load_model_by_type(model_type)

# 경기 선택
st.markdown("<div class='section-header'>경기 선택</div>", unsafe_allow_html=True)
match = st.selectbox("", list(match_list.keys()))
home_Team, away_Team = match_list[match]

# 예측 실행
if st.button("🔮 예측 실행하기"):
    home_id = team_name_to_id[home_Team]
    away_id = team_name_to_id[away_Team]

    prediction_row = create_prediction_row(GameDate='2025-06-03', home_Team=home_id, away_Team=away_id)

    probability = predict_model(prediction_row, model_type)

    if probability >= 0.5:
        win_team = home_Team
        win_prob = probability
    else:
        win_team = away_Team
        win_prob = 1 - probability


    # 결과 표시
    st.markdown(f"<div class='result-box'>🏆 {win_team} 승리 예상 ({win_prob*100:.2f}%)</div>", unsafe_allow_html=True)

    # SHAP 피처
    # 기존: top_features = explain_instance(model, prediction_row, model_type)

    # 수정: 팀 feature 제외 필터링
    raw_top_features = explain_instance(model, prediction_row, model_type)
    top_features = [f for f in raw_top_features if not (f.startswith("home_Team_") or f.startswith("away_Team_"))]

    st.markdown("<div class='section-header'>📊 주요 피처 </div>", unsafe_allow_html=True)
    for f in top_features:
        st.markdown(f"<div class='feature-box'>- {f}</div>", unsafe_allow_html=True)

    # GPT 해설
    explanation = generate_explanation(home_Team, away_Team, top_features, model_type, win_team)
    st.markdown("<div class='section-header'>🎙 AI 해설</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='gpt-box'>{explanation}</div>", unsafe_allow_html=True)
