# 4개 다 확률 반환 해뒀어 프롬프트 쓸 때 참고

import pandas as pd
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def predict_model(prediction_row, model_type):

    # 공통 전처리
    row = prediction_row.copy()
    row = row.reset_index(drop=True)

    ### Deep Learning
    if model_type == 'DeepLearning':
        model = load_model('../model/deep_learning_model.h5')

        # 정규화 제외
        exclude_columns = ['GameDate', 'home_Team', 'away_Team']
        X_to_scale = row.drop(columns=exclude_columns)
        scaler = StandardScaler()
        X_scaled_part = pd.DataFrame(scaler.fit_transform(X_to_scale), columns=X_to_scale.columns)

        # 팀 인코딩
        team_encoded = pd.get_dummies(
            row[['home_Team', 'away_Team']].reset_index(drop=True),
            columns=['home_Team', 'away_Team'],
            prefix=['home_Team', 'away_Team']
        ).astype(int)

        # 정규화 + 인코딩 합치기
        X_scaled = pd.concat([X_scaled_part.reset_index(drop=True), team_encoded], axis=1)

        pred = model.predict(X_scaled)
        return float(pred[0][0])

    ### Logistic Regression
    elif model_type == 'LogisticRegression':
        model = load('../model/logistic_model.pkl')

        exclude_columns = ['GameDate', 'home_Team', 'away_Team']
        X_to_scale = row.drop(columns=exclude_columns)
        scaler = StandardScaler()
        X_scaled_part = pd.DataFrame(scaler.fit_transform(X_to_scale), columns=X_to_scale.columns)

        team_encoded = pd.get_dummies(
            row[["home_Team", "away_Team"]].reset_index(drop=True),
            columns=["home_Team", "away_Team"],
            prefix=["home_Team", "away_Team"]
        ).astype(int)

        X_scaled = pd.concat([X_scaled_part.reset_index(drop=True), team_encoded], axis=1)

        pred = model.predict_proba(X_scaled)
        return float(pred[0][1])  # 확률 반환

    ### XGBoost
    elif model_type == 'XGBoost':
        model = load('../model/xgb_best_model.pkl')

        row_no_date = row.drop(columns=['GameDate'])
        X = row_no_date.drop(columns=['Result'], errors='ignore')

        team_encoded = pd.get_dummies(
            X[['home_Team', 'away_Team']].reset_index(drop=True),
            columns=['home_Team', 'away_Team'],
            prefix=['home_Team', 'away_Team']
        ).astype(int)

        X_numeric = X.drop(columns=['home_Team', 'away_Team']).reset_index(drop=True)
        X_scaled = pd.concat([X_numeric, team_encoded], axis=1)

        pred = model.predict_proba(X_scaled)
        return float(pred[0][1])

    ### RandomForest
    elif model_type == 'RandomForest':
        model = load('../model/rf_model.pkl')

        row_no_date = row.drop(columns=['GameDate'])
        X = row_no_date.drop(columns=['Result'], errors='ignore')

        pred = model.predict_proba(X)
        return float(pred[0][1])

    ### 기타 예외처리
    else:
        raise ValueError("지원하지 않는 모델 타입입니다.")


# from your_module import predict_model  # predictor.py로 따로 빼면 좋아!

# # data_loader.py에서 만든 1행짜리 예측 row 사용
# sample_row = create_prediction_row(GameDate='2024-05-30', home_Team=2, away_Team=6, result=1)

# pred = predict_model(sample_row, model_type='DeepLearning')  
# print(f"예측 확률: {pred:.4f}")
