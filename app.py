import streamlit as st
import pandas as pd
# from utils.data_loader import load_and_prepare_data
# from utils.predictor import predict_with_model
# from utils.shap_explainer import get_important_features
from utils.gpt_summary import generate_explanation
import joblib
import os

st.set_page_config(page_title="KBO 승패 예측기", layout="wide")

st.title("⚾ KBO 승패 예측 AI")

# 1. 팀 선택
team_list = ["LG", "두산", "SSG", "NC", "KIA", "한화", "키움", "삼성", "롯데", "KT"]
team1 = st.selectbox("🏆 팀 1을 선택하세요", team_list, index=0)
team2 = st.selectbox("🆚 팀 2를 선택하세요", team_list, index=1)

# 2. 모델 선택
model_name = st.selectbox("🧠 사용할 모델 선택", ["Logistic Regression", "Random Forest", "XGBoost", "딥러닝"])

# 3. 예측 버튼
if st.button("📊 예측하기"):
    # with st.spinner("모델 불러오는 중..."):
    #     model_path = {
    #         "Logistic Regression": "models/logistic_model.pkl",
    #         "Random Forest": "models/rf_model.pkl",
    #         "XGBoost": "models/xgb_model.pkl",
    #         "딥러닝": "models/dl_model.h5"
    #     }[model_name]
        
    #     model = joblib.load(model_path) if model_name != "딥러닝" else None  # 딥러닝 모델은 따로 처리 필요
    
    # # 4. 데이터 처리
    # df_input = load_and_prepare_data(team1, team2)  # rolling 처리 포함된 함수
    
    # # 5. 예측
    # prediction = predict_with_model(model, df_input, model_name)
    # st.success(f"✅ 예측 결과: **{prediction}** 팀이 이길 확률이 높습니다!")
    
    # # 6. SHAP으로 feature 중요도 확인
    # important_features = get_important_features(model, df_input, model_name, top_n=5)
    
    st.markdown("### 🔍 주요 영향 요인 (상위 5개)")
    # st.write(important_features)
    
    # 7. GPT 해설 문장
    st.markdown("### 📄 해설")
    # explanation = generate_explanation(team1, team2, important_features, model_name, prediction)
    explanation = generate_explanation()
    st.info(explanation)

    print("djjdsljfslkjsl")