import shap
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential

# 모델 타입별 background X_train 경로 미리 매핑
X_TRAIN_PATHS = {
    "LogisticRegression": "../models/X_train_logistic.csv",
    "RandomForest": "../models/X_train_rf.csv",
    "XGBoost": "../models/X_train_xgb.csv",
    "DeepLearning": "../models/X_train_deep.csv"
}

def explain_instance(model, prediction_row, model_type):
    """
    SHAP 기반 중요 피처 5개 추출 (X_train은 내부에서 자동 로딩)
    """

    # 모델에 맞는 background 데이터 로딩
    X_train = pd.read_csv(X_TRAIN_PATHS[model_type])

    # 1️⃣ Logistic Regression → coef 활용
    if model_type == 'LogisticRegression':
        coefs = model.coef_[0]
        values = prediction_row.values[0]
        contribs = coefs * values
        feature_contrib = pd.Series(contribs, index=prediction_row.columns)
        top_features = feature_contrib.abs().sort_values(ascending=False).head(5).index.tolist()
        return top_features

    # 2️⃣ Tree 계열 → TreeExplainer
    elif model_type in ['RandomForest', 'XGBoost']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(prediction_row)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_contrib = pd.Series(shap_values[0], index=prediction_row.columns)
        top_features = feature_contrib.abs().sort_values(ascending=False).head(5).index.tolist()
        return top_features

    # 3️⃣ 딥러닝 → KernelExplainer
    elif model_type == 'DeepLearning':
        background = shap.sample(X_train, 100, random_state=3)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(prediction_row, nsamples=100)
        feature_contrib = pd.Series(shap_values[0], index=prediction_row.columns)
        top_features = feature_contrib.abs().sort_values(ascending=False).head(5).index.tolist()
        return top_features

    else:
        raise ValueError("지원하지 않는 모델 타입입니다.")
    

#     top5_features = explain_instance(model, prediction_row, model_type='XGBoost')
#     print(top5_features)

