import os
import openai
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_explanation(team1: str, team2: str, features: list[str], model_name: str, prediction: str) -> str:
    prompt = f"""
    두 KBO 야구팀 '{team1}'와 '{team2}'가 경기를 합니다.
    {model_name} 모델이 '{prediction}' 팀이 이길 것이라고 예측했습니다.
    
    이 예측의 주요 요인은 다음과 같습니다: {", ".join(features)}
    
    위 특징들을 바탕으로 왜 {prediction} 팀이 이길 것인지, 자연스럽고 간결한 해설 문장을 한국어로 써줘.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 야구 분석 전문가야. 데이터를 바탕으로 예측 해설을 잘 해줘야 해."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ GPT 설명 생성 중 오류 발생: {str(e)}"

