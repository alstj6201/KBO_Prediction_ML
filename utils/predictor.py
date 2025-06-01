import pandas as pd
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

def predict_model(prediction_row, model_type):
    row = prediction_row.copy().reset_index(drop=True)

    # ✅ 각 모델별 feature list 불러오기
    feature_paths = {
        'DeepLearning': 'models/features_deep.pkl',
        'LogisticRegression': 'models/features_logistic.pkl',
        'XGBoost': 'models/features_xgb.pkl',
        'RandomForest': 'models/features_rf.pkl'  # RF는 안 쓰더라도 일관성 위해 만들어 둬도 좋음
    }
    feature_columns = joblib.load(feature_paths[model_type])

    ### Deep Learning
    if model_type == 'DeepLearning':
        model = load_model('models/deep_learning_model.h5')

        exclude_columns = ['GameDate', 'home_Team', 'away_Team']
        X_to_scale = row.drop(columns=exclude_columns)
        scaler = StandardScaler()
        X_scaled_part = pd.DataFrame(scaler.fit_transform(X_to_scale), columns=X_to_scale.columns)

        team_encoded = pd.get_dummies(
            row[['home_Team', 'away_Team']].reset_index(drop=True),
            columns=['home_Team', 'away_Team'],
            prefix=['home_Team', 'away_Team']
        ).astype(int)

        X_scaled = pd.concat([X_scaled_part.reset_index(drop=True), team_encoded], axis=1)
        X_scaled = X_scaled.reindex(columns=feature_columns, fill_value=0)

        pred = model.predict(X_scaled)
        return float(pred[0][0])

    ### Logistic Regression
    elif model_type == 'LogisticRegression':
        model = load('models/logistic_model.pkl')

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
        X_scaled = X_scaled.reindex(columns=feature_columns, fill_value=0)

        pred = model.predict_proba(X_scaled)
        return float(pred[0][1])

    ### XGBoost
    elif model_type == 'XGBoost':
        model = load('models/xgb_best_model.pkl')

        row_no_date = row.drop(columns=['GameDate'])
        X = row_no_date.drop(columns=['Result'], errors='ignore')

        team_encoded = pd.get_dummies(
            X[['home_Team', 'away_Team']].reset_index(drop=True),
            columns=['home_Team', 'away_Team'],
            prefix=['home_Team', 'away_Team']
        ).astype(int)

        X_numeric = X.drop(columns=['home_Team', 'away_Team']).reset_index(drop=True)
        X_scaled = pd.concat([X_numeric, team_encoded], axis=1)
        X_scaled = X_scaled.reindex(columns=feature_columns, fill_value=0)

        pred = model.predict_proba(X_scaled)
        return float(pred[0][1])

    ### RandomForest (이건 기존과 동일)
    elif model_type == 'RandomForest':
        model = load('models/best_random_forest_model_manual.pkl')

        row_no_date = row.drop(columns=['GameDate'])
        X = row_no_date.drop(columns=['Result'], errors='ignore')

        pred = model.predict_proba(X)
        return float(pred[0][1])

    else:
        raise ValueError("지원하지 않는 모델 타입입니다.")
