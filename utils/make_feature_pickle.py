import pandas as pd
import joblib
import os

# 모델별 매핑 (csv → pkl 대응)
file_map = {
    "deep": ("models/X_train_deep.csv", "models/features_deep.pkl"),
    "logistic": ("models/X_train_logistic.csv", "models/features_logistic.pkl"),
    "xgb": ("models/X_train_xgb.csv", "models/features_xgb.pkl"),
    "rf": ("models/X_train_rf.csv", "models/features_rf.pkl")
}

def generate_feature_pickles():
    for model_name, (csv_path, pkl_path) in file_map.items():
        if not os.path.exists(csv_path):
            print(f"⚠️ {csv_path} 파일이 존재하지 않습니다. 건너뜁니다.")
            continue

        # CSV 불러오기
        df = pd.read_csv(csv_path)
        feature_columns = df.columns.tolist()

        # 피클 저장
        joblib.dump(feature_columns, pkl_path)
        print(f"✅ {model_name.upper()} feature 저장 완료 → {pkl_path}")

if __name__ == '__main__':
    generate_feature_pickles()
