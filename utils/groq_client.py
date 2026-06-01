"""Groq API 클라이언트."""
from __future__ import annotations
import requests
import streamlit as st


def get_groq_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


@st.cache_data(ttl=60 * 10, show_spinner=False)
def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile", lang: str = "ko") -> str:
    """Groq 호출. 10분 캐시."""
    key = get_groq_key()
    if not key:
        return "⚠️ Groq API 키 미설정 — Streamlit Cloud → Settings → Secrets 에 GROQ_API_KEY 추가"

    sys_ko = (
        "당신은 한국어로 답변하는 미국 시장 전문 거시·기술 분석가입니다. "
        "주식 일/주봉 스윙 트레이더에게 도움이 되도록 간결하고 실용적으로 답하세요. "
        "불확실하면 그렇다고 명시하세요."
    )
    sys_en = (
        "You are a US market macro and technical analyst. "
        "Give concise, practical answers for a swing trader. "
        "Be clear about uncertainty."
    )

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_ko if lang == "ko" else sys_en},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 1500,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Groq 호출 실패: {e}"
