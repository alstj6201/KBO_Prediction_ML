import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_base_data(path='data/real_final.csv'):
    return pd.read_csv(path)

def preprocess_data(df, normalize=True, onehot=True, drop_columns=['GameDate']):
    X = df.drop(columns=['result'])
    y = df['result']

    if drop_columns:
        X = X.drop(columns=drop_columns)

    if onehot:
        X = pd.get_dummies(X, columns=['home_Team', 'away_Team'], prefix=['home_Team', 'away_Team'])

    if normalize:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)