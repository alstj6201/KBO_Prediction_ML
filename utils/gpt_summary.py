import os
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1️⃣ feature 설명 사전 (기존 그대로 유지)
feature_description = {
    "home_Recent_5_Win": "홈팀의 최근 5경기 승리 수",
    "home_Recent_5_Loss": "홈팀의 최근 5경기 패배 수",
    "home_OPS": "홈팀의 OPS (출루율+장타율)",
    "home_BB": "홈팀의 볼넷 수",
    "home_HBP": "홈팀의 몸에 맞는 공",
    "home_SO": "홈팀의 삼진 수",
    "home_ERA": "홈팀의 평균자책점(ERA)",
    "home_BB_p": "홈팀의 볼넷 비율",
    "home_K": "홈팀의 탈삼진 수",
    "home_IR": "홈팀의 잔루율(IR)",
    "home_IS": "홈팀의 IS (순수장타력)",
    "home_TBF": "홈팀 투수가 상대 타자를 상대한 횟수(TBF)",
    "away_OPS": "원정팀의 OPS (출루율+장타율)",
    "away_BB": "원정팀의 볼넷 수",
    "away_HBP": "원정팀의 몸에 맞는 공",
    "away_SO": "원정팀의 삼진 수",
    "away_ERA": "원정팀의 평균자책점(ERA)",
    "away_BB_p": "원정팀의 볼넷 비율",
    "away_K": "원정팀의 탈삼진 수",
    "away_IR": "원정팀의 잔루율(IR)",
    "away_IS": "원정팀의 IS (순수장타력)",
    "away_TBF": "원정팀 투수가 상대 타자를 상대한 횟수(TBF)"
}

# 2️⃣ 내부 팀 정보 DB (민서가 준 표 반영해서 만든 dict)
team_info = {
    "LG": {"rank": 1, "recent": "4승 1무 5패"},
    "한화": {"rank": 2, "recent": "5승 0무 5패"},
    "롯데": {"rank": 3, "recent": "3승 1무 6패"},
    "KT": {"rank": 4, "recent": "8승 0무 2패"},
    "삼성": {"rank": 5, "recent": "9승 0무 1패"},
    "SSG": {"rank": 6, "recent": "5승 1무 4패"},
    "KIA": {"rank": 7, "recent": "4승 1무 5패"},
    "NC": {"rank": 8, "recent": "3승 2무 5패"},
    "두산": {"rank": 9, "recent": "4승 1무 5패"},
    "키움": {"rank": 10, "recent": "1승 1무 8패"},
}

# 3️⃣ generate_explanation 함수
def generate_explanation(team1: str, team2: str, features: list[str], model_name: str, prediction: str) -> str:

    explained_features = [feature_description.get(f, f) for f in features]

    # 내부 DB에서 팀정보 불러오기
    team1_info = team_info.get(team1, {"rank": "정보 없음", "recent": "정보 없음"})
    team2_info = team_info.get(team2, {"rank": "정보 없음", "recent": "정보 없음"})

    # 프롬프트 생성
    prompt = f"""
    두 KBO 팀 '{team1}'와 '{team2}'의 경기가 예정되어 있습니다.
    AI {model_name} 모델은 '{prediction}'의 승리를 예측했습니다.

    각 팀의 최근 성적 정보는 다음과 같습니다:
    - {team1}: 현재 {team1_info['rank']}위, 최근 10경기 {team1_info['recent']}
    - {team2}: 현재 {team2_info['rank']}위, 최근 10경기 {team2_info['recent']}

    모델이 참고한 주요 피처는 다음과 같습니다: {", ".join(explained_features)}

    이 데이터를 바탕으로 야구 전문 해설가 스타일로 경기 프리뷰를 작성해줘.
    - 데이터 기반 근거가 드러나게
    - 팬들이 이해하기 쉽게
    - 너무 오버하거나 확정적으로 말하지 말고 균형감 있게
    - 반복 최소화
    - 문장 흐름 자연스럽게

    최종 결과를 한글로 써줘.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 KBO AI 데이터 기반 야구 전문 해설가이다. 전문가 해설을 팬들이 쉽게 이해할 수 있게 작성한다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=600
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ GPT 설명 생성 중 오류 발생: {str(e)}"
