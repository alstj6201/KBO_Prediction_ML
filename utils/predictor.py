import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load as joblib_load
import os

MODEL_PATH = 'models/'

def load_model_by_name(model_name):
    if model_name.endswith('.h5'):
        return load_model(os.path.join(MODEL_PATH, model_name))
    elif model_name.endswith('.pkl'):
        return joblib_load(os.path.join(MODEL_PATH, model_name))
    else:
        raise ValueError("지원하지 않는 모델 형식입니다.")

def predict(model, input_df):
    return model.predict(input_df)