import pandas as pd

def create_prediction_row(GameDate, home_Team, away_Team, 
                          real_final_path='../data/real_final.csv',
                          not_rolling_path='../data/final_data_not_rolling.csv'):

    ### 1️⃣ real_final.csv에서 Recent_5_Win, Recent_5_Loss 갱신하기
    real_df = pd.read_csv(real_final_path)
    
    # 홈팀 경기 불러오기 (해당 GameDate 이전 경기만)
    home_history = real_df[(real_df['home_Team'] == home_Team) & (real_df['GameDate'] < GameDate)]
    
    # 가장 최근 5경기만 사용
    home_recent = home_history.sort_values('GameDate', ascending=False).head(5)
    
    home_Recent_5_Win = home_recent['Result'].sum()
    home_Recent_5_Loss = len(home_recent) - home_Recent_5_Win

    ### 2️⃣ not_rolling_only.csv에서 각 팀 feature 계산
    not_rolling_df = pd.read_csv(not_rolling_path)

    # 🟢 이 피처들은 최근 값 그대로 사용
    no_rolling_features = ['OPS', 'IR', 'IS', 'HBP']
    
    # 🟢 이 피처들은 rolling 적용
    rolling_features = ['BB', 'SO', 'ERA', 'BB_p', 'K', 'TBF']
    
    # 홈팀 데이터
    home_team_data = not_rolling_df[not_rolling_df['Team'] == home_Team].sort_values('GameDate', ascending=False)
    away_team_data = not_rolling_df[not_rolling_df['Team'] == away_Team].sort_values('GameDate', ascending=False)

    # 가장 최근 값
    home_no_rolling = home_team_data.iloc[0][no_rolling_features]
    away_no_rolling = away_team_data.iloc[0][no_rolling_features]

    # rolling 평균
    home_rolling = home_team_data.head(5)[rolling_features].mean()
    away_rolling = away_team_data.head(5)[rolling_features].mean()

    ### 3️⃣ 최종 데이터 조립
    game_row = {
        'GameDate': GameDate,
        'home_Team': home_Team,
        'away_Team': away_Team,
        'home_Recent_5_Win': home_Recent_5_Win,
        'home_Recent_5_Loss': home_Recent_5_Loss,
    }

    # 홈팀 피처 붙이기
    for col in no_rolling_features:
        game_row[f'home_{col}'] = home_no_rolling[col]
    for col in rolling_features:
        game_row[f'home_{col}'] = home_rolling[col]

    # 원정팀 피처 붙이기
    for col in no_rolling_features:
        game_row[f'away_{col}'] = away_no_rolling[col]
    for col in rolling_features:
        game_row[f'away_{col}'] = away_rolling[col]

    # 최종 DataFrame으로 변환
    prediction_df = pd.DataFrame([game_row])

    ### 4️⃣ 컬럼 순서 정리 (너가 원하는 순서)
    final_columns = [
        'GameDate', 'home_Team', 'away_Team',
        'home_Recent_5_Win', 'home_Recent_5_Loss',
        'home_OPS', 'home_BB', 'home_HBP', 'home_SO', 'home_ERA', 'home_BB_p', 'home_K', 'home_IR', 'home_IS', 'home_TBF',
        'away_OPS', 'away_BB', 'away_HBP', 'away_SO', 'away_ERA', 'away_BB_p', 'away_K', 'away_IR', 'away_IS', 'away_TBF'
    ]

    prediction_df = prediction_df[final_columns]

    return prediction_df

# ### 사용 예시
# if __name__ == '__main__':
#     sample_row = create_prediction_row(
#         GameDate='2024-05-30',  # 오늘 날짜 예시
#         home_Team=2,            # 삼성
#         away_Team=6,            # 한화
#         result=1
#     )

#     print(sample_row)
#     sample_row.to_csv('../data/sample_prediction_row.csv', index=False, encoding='utf-8-sig')