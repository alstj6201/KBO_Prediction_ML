import shap
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential

# 모델별 경로 (X_train, feature_columns 둘 다 함께 매핑)
X_TRAIN_PATHS = {
    "LogisticRegression": ("models/X_train_logistic.csv", "models/features_logistic.pkl"),
    "RandomForest": ("models/X_train_rf.csv", "models/features_rf.pkl"),
    "XGBoost": ("models/X_train_xgb.csv", "models/features_xgb.pkl"),
    "DeepLearning": ("models/X_train_deep.csv", "models/features_deep.pkl")
}

def explain_instance(model, prediction_row, model_type):
    """
    SHAP 기반 중요 피처 5개 추출 (X_train, feature_columns 자동 로딩)
    """

    # 모델별 X_train, feature_columns 로드
    X_train_path, feature_path = X_TRAIN_PATHS[model_type]
    X_train = pd.read_csv(X_train_path)
    feature_columns = joblib.load(feature_path)

    # LogisticRegression → coef 기반 해석
    if model_type == 'LogisticRegression':
        # ✅ feature 맞춰 정렬 (이게 핵심!)
        prediction_row = prediction_row.reindex(columns=feature_columns, fill_value=0)

        coefs = model.coef_[0]
        values = prediction_row.values[0]
        contribs = coefs * values
        feature_contrib = pd.Series(contribs, index=prediction_row.columns)
        top_features = feature_contrib.abs().sort_values(ascending=False).head(5).index.tolist()
        return top_features

    elif model_type == 'RandomForest':
        prediction_row = prediction_row.reindex(columns=feature_columns, fill_value=0)
        prediction_row = prediction_row.astype(float)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(prediction_row)

        feature_contrib = pd.Series(shap_values[0], index=prediction_row.columns)
        top_features = feature_contrib.abs().sort_values(ascending=False).head(5).index.tolist()
        return top_features

    elif model_type == 'XGBoost':
        # XGB 전용 feature_columns로 align
        prediction_row = prediction_row.reindex(columns=feature_columns, fill_value=0)
        prediction_row = prediction_row.astype(float)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(prediction_row)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        feature_contrib = pd.Series(shap_values[0], index=prediction_row.columns)
        top_features = feature_contrib.abs().sort_values(ascending=False).head(5).index.tolist()
        return top_features


    # 딥러닝 → KernelExplainer 사용
    elif model_type == 'DeepLearning':
        prediction_row = prediction_row.reindex(columns=feature_columns, fill_value=0)

        background = shap.sample(X_train, 100, random_state=42)
        explainer = shap.KernelExplainer(model.predict, background)

        # shap_values는 list로 반환됨 → 첫번째 클래스만 꺼내서 squeeze()
        shap_values = explainer.shap_values(prediction_row, nsamples=100)
        shap_array = np.array(shap_values[0]).squeeze()

        feature_contrib = pd.Series(shap_array, index=prediction_row.columns)
        top_features = feature_contrib.abs().sort_values(ascending=False).head(5).index.tolist()
        return top_features



    else:
        raise ValueError("지원하지 않는 모델 타입입니다.")
