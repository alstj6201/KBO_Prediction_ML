Final 열
GameDate,
home_Team,away_Team,
result,
home_Recent_5_Win,home_Recent_5_Loss,
home_OPS,home_BB,home_HBP,home_SO,home_ERA,home_BB_p,home_K,home_IR,home_IS,home_TBF,
away_OPS,away_BB,away_HBP,away_SO,away_ERA,away_BB_p,away_K,away_IR,away_IS,away_TBF


함수 parameter로 주어짐
GameDate,
home_Team,away_Team,

예측값
result,

real_final.csv의 GameDate,home_Team,away_Team,Result,home_Recent_5_Win,home_Recent_5_Loss를 가지고 최근 5경기 승/패 작성
* 각 home_Recent_5_Win과 home_Recent_5_Loss에는 그 당일 경기의 결과가 포함 안 되어 있어서 Result 가지고 그 두 개를 갱신해야함
home_Recent_5_Win,home_Recent_5_Loss,

not_rolling_only.csv에서 각 팀의 가장 최근 데이터 사용
OPS, IR, IS, HBP, 

not_rolling_only.csv에서 각 팀의 최근 5경기에서 
BB, SO, ERA, BB_p, K, TBF rolling

import pandas as pd

# 데이터 불러오기
df_new = pd.read_csv("../data/final_data_rolling.csv")

games = []
for i in range(0, len(df_new), 2):

    if df_new.loc[i, 'home_away'] == 0:
        home = df_new.iloc[i+1]
        away = df_new.iloc[i]
    else:
        home = df_new.iloc[i]
        away = df_new.iloc[i + 1]

    # 새로운 경기 단위 row 생성
    game_row = {
        'GameDate': home['GameDate'],
        'home_Team': home['Team'],
        'away_Team': away['Team'],
        'Result': int(home['Result'] > away['Result']) , # 홈팀이 이기면 1, 지면 0
        'home_Recent_5_Win': home['Recent_5_Win'],
        'home_Recent_5_Loss': home['Recent_5_Loss']
    }

    # 제외할 열
    excluded_cols = ['GameDate', 'Team', 'Result','home_away','Recent_5_Win','Recent_5_Loss']

    # 홈팀 feature 추가
    for col in df_new.columns:
        if col not in excluded_cols:
            game_row[f'home_{col}'] = home[col]

    # 원정팀 feature 추가
    for col in df_new.columns:
        if col not in excluded_cols:
            game_row[f'away_{col}'] = away[col]

    games.append(game_row)

# 새로운 데이터프레임으로 변환
df_game_unit = pd.DataFrame(games)

df_game_unit.to_csv('../data/real_final.csv', index=False, encoding='utf-8-sig')  # ← 저장 파일명 지정

print("✅ 컬럼 붙이기 완료!")

활용해서 칼럼 2개 붙이기기

마지막 정렬 순서 * Result 없음
GameDate,
home_Team,away_Team,
home_Recent_5_Win,home_Recent_5_Loss,
home_OPS,home_BB,home_HBP,home_SO,home_ERA,home_BB_p,home_K,home_IR,home_IS,home_TBF,
away_OPS,away_BB,away_HBP,away_SO,away_ERA,away_BB_p,away_K,away_IR,away_IS,away_TBF


