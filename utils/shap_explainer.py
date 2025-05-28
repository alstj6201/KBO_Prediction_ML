import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model, Sequential
from joblib import load as joblib_load

def run_model_explanation(model, X_train, X_sample):
    """
    모델 유형(Logistic, Tree, Deep)을 감지해서 SHAP 해석 or coef 출력
    """
    # Logistic Regression
    if isinstance(model, LogisticRegression):
        print("🔎 Logistic Regression은 SHAP 대신 coef_ 사용")
        coefs = model.coef_[0]
        coef_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': coefs
        }).sort_values(by='Coefficient', key=abs, ascending=False)

        print(coef_df)
        return coef_df

    # Tree 모델
    elif isinstance(model, (RandomForestClassifier, XGBClassifier)):
        print("🌲 Tree 기반 모델 - TreeExplainer 사용 중...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns)

    # 딥러닝 모델
    elif hasattr(model, "predict") and "keras" in str(type(model)).lower():
        print("🧠 딥러닝 모델 - KernelExplainer 사용 중... (느릴 수 있음)")
        explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, feature_names=X_sample.columns)

    else:
        raise ValueError("❌ 지원하지 않는 모델 타입입니다.")

def load_model_by_name(model_name: str):
    """
    모델 이름에 따라 joblib or keras 모델 자동 로딩
    """
    if model_name.endswith('.h5'):
        return load_model(f"models/{model_name}")
    elif model_name.endswith('.pkl'):
        return joblib_load(f"models/{model_name}")
    else:
        raise ValueError("모델 확장자가 .h5 또는 .pkl 이어야 합니다.")
