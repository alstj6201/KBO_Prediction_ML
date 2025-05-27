# 필요한 외부 라이브러리 임포트
import ollama  # ← Ollama 라이브러리 (로컬 LLM 실행을 위한)
import streamlit as st  # ← Streamlit 웹앱 프레임워크

# 📌 세션 상태 초기화 함수 정의
def init_session_state(keys: dict):
    for key, value in keys.items():
        if key not in st.session_state:  # ← 세션 상태에 해당 key가 없으면
            st.session_state[key] = value  # ← 기본값으로 새로 저장

# 📌 사용자 입력 메시지를 Streamlit에 출력하고 기록
def chat_message_user(prompt: str) -> dict:
    with st.chat_message("user"):  # ← 사용자 메시지 블록
        st.markdown(prompt)  # ← 사용자 입력 출력
        return dict(role="user", content=prompt)  # ← 기록용 딕셔너리 반환

# 📌 LLM 응답 메시지를 스트리밍 방식으로 출력
def chat_message_llm_stream(role: str, model: str, messages: list) -> dict:
    with st.chat_message(role):  # ← "assistant" 역할로 메시지 출력
        with st.spinner("대화를 생성하는 중입니다..."):  # ← 로딩 스피너 표시

            # ollama.chat(): LLM에게 메시지를 보냄, stream=True 옵션 필수!
            stream = ollama.chat(model=model, messages=messages, stream=True)

            # 제네레이터 함수: 스트림 응답을 하나씩 읽어옴
            def stream_parser(stream):
                for chunk in stream:
                    yield chunk["message"]["content"]  # ← 실제 응답 텍스트 추출

            # 📌 Streamlit에서 제네레이터로 메시지 출력
            content = st.write_stream(stream_parser(stream))  # ← 실시간 출력
            return dict(role="assistant", content=content)  # ← 기록용 딕셔너리 반환

# ------------------- 메인 코드 영역 -------------------
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("🤖 만들면서 배우는 챗봇")

    # 📌 세션 상태 초기화: 이전 대화 저장용 msgs와 실행 플래그 running
    init_session_state(dict(msgs=[], running=False))
    msgs: list = st.session_state["msgs"]
    running: bool = st.session_state["running"]

    # 📌 사용자가 prompt를 입력했는지 여부 확인하여 running 설정
    if "prompt" in st.session_state and st.session_state["prompt"] is not None:
        running = True
    else:
        running = False

    # 📌 이전 대화 내용 다시 출력 (페이지 새로고침 시 유지됨)
    for row in msgs:
        with st.chat_message(row["role"]):
            st.markdown(row["content"])

    # 📌 사용자 입력창 생성
    #   disabled → running이 True면 입력창 비활성화됨
    #   key → prompt 입력을 session_state로 추적
    if prompt := st.chat_input("대화를 입력하세요!", disabled=running, key="promph"):
        msg_user = chat_message_user(prompt)
        msgs.append(msg_user)  # 사용자 메시지 저장

        # 📌 LLM 호출 (모델 이름은 예시로 "gemma2:9b")
        msg_llm = chat_message_llm_stream("assistant", "gemma2:9b", msgs)
        msgs.append(msg_llm)  # 응답 메시지 저장

        # 📌 대화가 끝난 뒤 앱을 다시 실행 (running 값 등 갱신)
        st.rerun()
