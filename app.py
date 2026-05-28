"""
Market Dashboard — 시장 심리 / 매크로 / 관심종목 시그널
무료 스택: Streamlit + yfinance
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ---------- Thematic ETF universe ----------
THEMES: dict[str, list[tuple[str, str]]] = {
    "🤖 AI / 반도체": [
        ("SMH", "반도체"), ("SOXX", "반도체"), ("AIQ", "AI 종합"),
        ("CHAT", "생성형 AI"), ("BOTZ", "AI/로봇"),
    ],
    "🦾 로봇 / 피지컬 AI": [
        ("BOTZ", "로봇·AI"), ("ROBO", "로보틱스"), ("ARKQ", "자율기술"),
        ("IRBO", "AI·로봇"),
    ],
    "🧬 바이오 / 바이오AI": [
        ("IBB", "바이오테크"), ("XBI", "바이오 SMID"),
        ("ARKG", "유전체혁명"), ("IDNA", "유전체"),
    ],
    "🚀 우주 / UAM / 항공": [
        ("ARKX", "우주탐사"), ("UFO", "우주산업"),
        ("ITA", "방산·항공"), ("PPA", "방산·항공"),
    ],
    "🛰️ 드론 / 방산": [
        ("ITA", "방산"), ("PPA", "방산"), ("XAR", "방산"),
    ],
    "⚡ 원자력 / 클린에너지": [
        ("URA", "우라늄"), ("NLR", "원자력"),
        ("ICLN", "클린에너지"), ("TAN", "태양광"),
    ],
    "🔋 EV / 자율주행 / 배터리": [
        ("DRIV", "자율주행"), ("LIT", "배터리·리튬"), ("IDRV", "EV"),
    ],
    "🛡️ 사이버보안": [
        ("CIBR", "사이버보안"), ("HACK", "사이버보안"),
    ],
    "💰 핀테크 / 크립토": [
        ("ARKF", "핀테크"), ("IBIT", "비트코인 ETF"), ("BITQ", "크립토 산업"),
    ],
    "⚛️ 양자컴퓨팅 / 신기술": [
        ("QTUM", "양자·신기술"), ("ARKK", "혁신성장"), ("ARKW", "차세대 인터넷"),
    ],
    "🏗️ 인프라 / 리쇼어링": [
        ("PAVE", "미국 인프라"), ("SLX", "철강"),
    ],
}

st.set_page_config(page_title="Market Dashboard", layout="wide", page_icon="📊")

# ---------- Data fetch ----------
# 탭별로 갱신 주기를 다르게: 시장심리(1분) / 매크로(5분) / 관심종목(15분)
def _fetch_impl(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    df = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna(how="all")


@st.cache_data(ttl=60)  # 1분
def fetch_fast(tickers: tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    return _fetch_impl(tickers, period)


@st.cache_data(ttl=60 * 5)  # 5분
def fetch_macro(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    return _fetch_impl(tickers, period)


@st.cache_data(ttl=60 * 15)  # 15분
def fetch_slow(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    return _fetch_impl(tickers, period)


@st.cache_data(ttl=30)  # 30초 — VIX 단독 빠른 조회
def fetch_vix_quote() -> tuple[float, float, str]:
    """현재 VIX와 전일대비. yfinance Ticker.info / fast_info 사용."""
    t = yf.Ticker("^VIX")
    try:
        fi = t.fast_info
        last = float(fi["last_price"])
        prev = float(fi["previous_close"])
    except Exception:
        hist = t.history(period="5d")
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
    chg = (last / prev - 1) * 100
    return last, chg, dt.datetime.now().strftime("%H:%M:%S")


def pct_rank(series: pd.Series, value: float, lookback: int = 252) -> float:
    """How greedy is `value` vs last N days? 0=extreme fear, 100=extreme greed."""
    s = series.dropna().tail(lookback)
    if len(s) < 20:
        return 50.0
    return float((s < value).mean() * 100)


# ---------- Groq LLM ----------
def _get_groq_key() -> str | None:
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


@st.cache_data(ttl=60 * 10, show_spinner=False)
def call_groq(prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Groq 무료 API 호출. 10분 캐시."""
    key = _get_groq_key()
    if not key:
        return "⚠️ Groq API 키 미설정 — Streamlit Cloud Settings → Secrets 에 `GROQ_API_KEY=...` 추가하세요. (https://console.groq.com/keys 에서 무료 발급)"
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": (
                        "당신은 한국어로 답변하는 미국 시장 전문 거시·테마 분석가입니다. "
                        "주식 일/주봉 스윙 트레이더에게 도움이 되도록 간결하고 실용적으로 답하세요. "
                        "추측보다 데이터에 근거해서 말하고, 불확실하면 그렇다고 명시하세요."
                    )},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.4,
                "max_tokens": 1200,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Groq 호출 실패: {e}"


# ---------- Sentiment score ----------
@dataclass
class Component:
    name: str
    score: float  # 0=fear, 100=greed
    value: float
    note: str


def compute_sentiment() -> tuple[float, list[Component]]:
    tickers = ["^VIX", "SPY", "QQQ", "TLT", "HYG", "LQD", "GLD", "UUP", "^TNX"]
    df = fetch_fast(tuple(tickers))

    comps: list[Component] = []

    # 1) VIX — 낮을수록 탐욕
    vix = df["^VIX"].dropna()
    vix_now = float(vix.iloc[-1])
    vix_score = 100 - pct_rank(vix, vix_now)  # invert: high vix = fear
    comps.append(Component("VIX (변동성)", vix_score, vix_now,
                           "낮을수록 탐욕, 30↑ 공포"))

    # 2) SPY 모멘텀 (125일 수익률)
    spy = df["SPY"].dropna()
    mom = spy.pct_change(125).iloc[-1] * 100
    mom_score = pct_rank(spy.pct_change(125), mom)
    comps.append(Component("SPY 125일 모멘텀", mom_score, mom,
                           f"{mom:+.1f}% (양수=상승추세)"))

    # 3) SPY vs 125일선 거리
    ma = spy.rolling(125).mean()
    dist = (spy / ma - 1).dropna()
    dist_now = float(dist.iloc[-1] * 100)
    dist_score = pct_rank(dist, dist.iloc[-1])
    comps.append(Component("SPY 125MA 이격도", dist_score, dist_now,
                           f"{dist_now:+.1f}% (200MA 위면 강세)"))

    # 4) 정크본드 수요 (HYG/LQD) — 높을수록 위험선호
    if "HYG" in df and "LQD" in df:
        ratio = (df["HYG"] / df["LQD"]).dropna()
        ratio_now = float(ratio.iloc[-1])
        ratio_score = pct_rank(ratio, ratio.iloc[-1])
        comps.append(Component("정크본드 수요 (HYG/LQD)", ratio_score, ratio_now,
                               "높을수록 위험선호"))

    # 5) 안전자산 회피 (SPY 20일 vs TLT 20일)
    spy_ret = spy.pct_change(20)
    tlt_ret = df["TLT"].pct_change(20)
    diff = (spy_ret - tlt_ret).dropna()
    diff_now = float(diff.iloc[-1] * 100)
    diff_score = pct_rank(diff, diff.iloc[-1])
    comps.append(Component("안전자산 회피 (SPY-TLT 20d)", diff_score, diff_now,
                           f"{diff_now:+.1f}%p (높을수록 위험선호)"))

    # 6) 금/달러 — 약달러면 위험자산 우호
    uup = df["UUP"].dropna()
    uup_ret = uup.pct_change(60).iloc[-1] * 100
    # 강달러 = fear → invert
    uup_score = 100 - pct_rank(uup.pct_change(60), uup.pct_change(60).iloc[-1])
    comps.append(Component("달러 강세 (UUP 60d)", uup_score, uup_ret,
                           f"{uup_ret:+.1f}% (약달러=위험선호)"))

    composite = float(np.mean([c.score for c in comps]))
    return composite, comps


# ---------- UI ----------
st.title("📊 Market Dashboard")

# 상단 실시간 VIX 헤더 + 수동 새로고침
hcol1, hcol2, hcol3, hcol4 = st.columns([2, 2, 2, 2])
try:
    vix_last, vix_chg, vix_ts = fetch_vix_quote()
    hcol1.metric("VIX (실시간 ~15분 지연)", f"{vix_last:.2f}", f"{vix_chg:+.2f}%")
    hcol2.caption(f"퀵 조회: {vix_ts}")
except Exception as e:
    hcol1.error(f"VIX 조회 실패: {e}")

if hcol4.button("🔄 강제 새로고침", help="모든 캐시를 비우고 다시 받음"):
    st.cache_data.clear()
    st.rerun()

st.caption(
    f"페이지 로드: {dt.datetime.now():%Y-%m-%d %H:%M:%S} · "
    "캐시 — 시장심리 1분 / 매크로 5분 / 관심종목 15분 · "
    "yfinance 자체 지연 ~15분"
)

tab_ai, tab_sentiment, tab_macro, tab_themes, tab_watchlist = st.tabs(
    ["🤖 AI 해석", "🧠 시장 심리", "🌐 매크로", "🚀 테마", "👀 관심종목"]
)

# ===== Sentiment tab =====
with tab_sentiment:
    with st.spinner("시장 데이터 로딩중..."):
        score, comps = compute_sentiment()

    col1, col2 = st.columns([1, 2])

    with col1:
        if score < 25:
            label, color = "극단적 공포", "#c62828"
        elif score < 45:
            label, color = "공포", "#ef6c00"
        elif score < 55:
            label, color = "중립", "#fbc02d"
        elif score < 75:
            label, color = "탐욕", "#7cb342"
        else:
            label, color = "극단적 탐욕", "#2e7d32"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"<b>{label}</b>", "font": {"size": 24}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 25], "color": "#ffebee"},
                    {"range": [25, 45], "color": "#fff3e0"},
                    {"range": [45, 55], "color": "#fffde7"},
                    {"range": [55, 75], "color": "#f1f8e9"},
                    {"range": [75, 100], "color": "#e8f5e9"},
                ],
            },
        ))
        gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge, use_container_width=True)

    with col2:
        st.subheader("구성요소 (0=공포 · 100=탐욕)")
        for c in comps:
            cols = st.columns([3, 1, 4])
            cols[0].write(f"**{c.name}**")
            cols[1].write(f"`{c.score:.0f}`")
            cols[2].progress(int(c.score) / 100, text=c.note)

# ===== Macro tab =====
with tab_macro:
    st.subheader("주요 매크로 지표")
    macro_tickers = {
        "^VIX": "VIX (변동성)",
        "UUP": "달러 (UUP)",
        "^TNX": "10년 국채금리",
        "GLD": "금 (GLD)",
        "USO": "원유 (USO)",
        "TLT": "장기국채 (TLT)",
    }
    macro_df = fetch_macro(tuple(macro_tickers.keys()), period="1y")
    cols = st.columns(3)
    for i, (t, name) in enumerate(macro_tickers.items()):
        if t not in macro_df.columns:
            continue
        s = macro_df[t].dropna()
        last = float(s.iloc[-1])
        chg = float(s.pct_change(20).iloc[-1] * 100)
        with cols[i % 3]:
            st.metric(name, f"{last:,.2f}", f"{chg:+.1f}% (20d)")

    st.subheader("섹터 ETF 상대강도 (60일 수익률)")
    sectors = {
        "XLK": "기술", "XLY": "임의소비재", "XLC": "통신",
        "XLF": "금융", "XLI": "산업", "XLE": "에너지",
        "XLB": "소재", "XLV": "헬스케어", "XLP": "필수소비재",
        "XLU": "유틸리티", "XLRE": "리츠",
    }
    sec_df = fetch_macro(tuple(sectors.keys()), period="6mo")
    rets = (sec_df.iloc[-1] / sec_df.iloc[0] - 1).sort_values(ascending=True) * 100
    fig = go.Figure(go.Bar(
        x=rets.values,
        y=[f"{sectors.get(t, t)} ({t})" for t in rets.index],
        orientation="h",
        marker_color=["#2e7d32" if v > 0 else "#c62828" for v in rets.values],
        text=[f"{v:+.1f}%" for v in rets.values],
        textposition="outside",
    ))
    fig.update_layout(height=420, margin=dict(l=20, r=60, t=20, b=20),
                      xaxis_title="6개월 수익률 (%)")
    st.plotly_chart(fig, use_container_width=True)


# ---------- Theme performance helpers ----------
def _all_theme_tickers() -> tuple[str, ...]:
    seen = []
    for items in THEMES.values():
        for t, _ in items:
            if t not in seen:
                seen.append(t)
    return tuple(seen)


def compute_theme_perf(period: str = "3mo") -> pd.DataFrame:
    """각 ETF의 1m / 3m / YTD 수익률을 반환."""
    df = fetch_macro(_all_theme_tickers(), period="ytd")
    df3 = fetch_macro(_all_theme_tickers(), period=period)
    rows = []
    for theme, items in THEMES.items():
        for t, desc in items:
            if t not in df.columns:
                continue
            s = df[t].dropna()
            s3 = df3[t].dropna() if t in df3.columns else s
            if len(s) < 21:
                continue
            rows.append({
                "테마": theme,
                "티커": t,
                "설명": desc,
                "현재가": float(s.iloc[-1]),
                "1개월": float(s.pct_change(21).iloc[-1] * 100),
                "3개월": float(s3.iloc[-1] / s3.iloc[0] - 1) * 100 if len(s3) > 5 else np.nan,
                "YTD": float(s.iloc[-1] / s.iloc[0] - 1) * 100,
            })
    return pd.DataFrame(rows)


# ===== Themes tab =====
with tab_themes:
    st.subheader("테마별 ETF 성과")
    st.caption("미국 상장 ETF · 1개월 / 3개월 / YTD 수익률")

    with st.spinner("테마 데이터 로딩중..."):
        tdf = compute_theme_perf()

    if tdf.empty:
        st.warning("데이터를 불러올 수 없습니다.")
    else:
        # 테마 평균으로 상위 테마 랭킹
        theme_avg = tdf.groupby("테마")["1개월"].mean().sort_values(ascending=True)
        fig_t = go.Figure(go.Bar(
            x=theme_avg.values,
            y=theme_avg.index,
            orientation="h",
            marker_color=["#2e7d32" if v > 0 else "#c62828" for v in theme_avg.values],
            text=[f"{v:+.1f}%" for v in theme_avg.values],
            textposition="outside",
        ))
        fig_t.update_layout(
            height=420, margin=dict(l=20, r=60, t=20, b=20),
            xaxis_title="1개월 평균 수익률 (%)",
            title="🏆 테마 랭킹 (자금 흐름)",
        )
        st.plotly_chart(fig_t, use_container_width=True)

        # 개별 ETF 표 (정렬 가능)
        st.subheader("개별 ETF 상세")
        display = tdf.copy()
        for col in ["현재가", "1개월", "3개월", "YTD"]:
            display[col] = display[col].round(2)
        st.dataframe(
            display.sort_values("1개월", ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                "1개월": st.column_config.NumberColumn(format="%.1f%%"),
                "3개월": st.column_config.NumberColumn(format="%.1f%%"),
                "YTD": st.column_config.NumberColumn(format="%.1f%%"),
                "현재가": st.column_config.NumberColumn(format="$%.2f"),
            },
        )


# ===== AI Interpretation tab =====
with tab_ai:
    st.subheader("🤖 AI 시장 진단")
    st.caption("Groq Llama 3.3 70B · 모든 지표를 종합해 한국어로 해석 · 10분 캐시")

    has_key = _get_groq_key() is not None
    if not has_key:
        st.warning(
            "**Groq API 키 미설정**  \n"
            "1. https://console.groq.com/keys 에서 무료 발급 (가입 30초)  \n"
            "2. Streamlit Cloud → 앱 → ⋮ → Settings → Secrets 에 추가:\n"
            "    ```\n    GROQ_API_KEY = \"gsk_xxxxx\"\n    ```\n"
            "3. 저장하면 자동 재시작 → 이 탭에서 해석 표시"
        )

    if st.button("🔮 지금 해석 생성", type="primary", disabled=not has_key):
        with st.spinner("데이터 수집 + LLM 분석중..."):
            # 스냅샷 데이터 수집
            score, comps = compute_sentiment()
            tdf = compute_theme_perf()

            # 섹터 성과
            sectors = {
                "XLK": "기술", "XLY": "임의소비재", "XLC": "통신",
                "XLF": "금융", "XLI": "산업", "XLE": "에너지",
                "XLB": "소재", "XLV": "헬스케어", "XLP": "필수소비재",
                "XLU": "유틸리티", "XLRE": "리츠",
            }
            sec_df = fetch_macro(tuple(sectors.keys()), period="3mo")
            sec_perf = ((sec_df.iloc[-1] / sec_df.iloc[0] - 1) * 100).sort_values(ascending=False)

            # 매크로 스냅
            macro_df = fetch_macro(("^VIX", "UUP", "^TNX", "GLD", "USO", "TLT"), period="3mo")
            macro_snap = {
                t: {
                    "현재": float(macro_df[t].dropna().iloc[-1]),
                    "20일변동%": float(macro_df[t].pct_change(20).dropna().iloc[-1] * 100),
                }
                for t in macro_df.columns
            }

            # 테마 랭킹
            theme_avg = tdf.groupby("테마")["1개월"].mean().sort_values(ascending=False)
            top_themes = theme_avg.head(5).to_dict()
            bot_themes = theme_avg.tail(3).to_dict()

            # ETF 단위 핫/콜드
            top_etfs = tdf.nlargest(8, "1개월")[["티커", "설명", "1개월"]].to_dict("records")
            bot_etfs = tdf.nsmallest(5, "1개월")[["티커", "설명", "1개월"]].to_dict("records")

            prompt = f"""아래는 오늘({dt.date.today()}) 미국 시장 스냅샷이다.

# 종합 Fear/Greed: {score:.0f}/100
구성요소:
{chr(10).join(f"- {c.name}: 점수 {c.score:.0f}, 값 {c.value:.2f} ({c.note})" for c in comps)}

# 매크로 (현재값, 20일 변동률)
{chr(10).join(f"- {k}: {v['현재']:.2f}, {v['20일변동%']:+.1f}%" for k, v in macro_snap.items())}

# 섹터 ETF 3개월 성과 (강→약)
{chr(10).join(f"- {sectors.get(t,t)} ({t}): {v:+.1f}%" for t, v in sec_perf.items())}

# 테마 1개월 평균 (상위 5)
{chr(10).join(f"- {k}: {v:+.1f}%" for k, v in top_themes.items())}

# 테마 1개월 평균 (하위 3)
{chr(10).join(f"- {k}: {v:+.1f}%" for k, v in bot_themes.items())}

# 1개월 핫 ETF
{chr(10).join(f"- {e['티커']} ({e['설명']}): {e['1개월']:+.1f}%" for e in top_etfs)}

# 1개월 콜드 ETF
{chr(10).join(f"- {e['티커']} ({e['설명']}): {e['1개월']:+.1f}%" for e in bot_etfs)}

다음 5개 섹션으로 답변하라 (각 섹션 헤더는 ## 로 시작):

## 한 줄 요약
오늘 시장 분위기를 한 문장으로.

## 시장 심리 진단
Fear/Greed 점수와 구성요소들을 보고 지금 시장이 어떤 국면인지. 어떤 컴포넌트가 끌어내리거나 끌어올리고 있는지 구체적으로.

## 매크로 환경
VIX/달러/금리/금/원유 흐름이 위험자산에 우호적인지 적대적인지. 특히 주목할 변화.

## 자금 흐름 / 핫한 테마
어떤 섹터·테마로 돈이 들어오고 빠지는지. 그 이유에 대한 가설(실적 시즌, 정책, 금리 등 추정).

## 스윙 트레이더 액션
일/주봉 스윙하는 한국 투자자 입장에서 (1) 지금 신규 진입을 적극적으로 늘릴 환경인지 (2) 우선 볼 만한 테마/섹터 2~3개 (3) 피해야 할 영역. 단정하지 말고 근거와 함께.

답변은 한국어. 마크다운 사용. 과한 디스클레이머 없이 명확하게."""

            answer = call_groq(prompt)
            st.session_state["last_ai_answer"] = answer
            st.session_state["last_ai_time"] = dt.datetime.now()

    if "last_ai_answer" in st.session_state:
        st.caption(f"생성 시각: {st.session_state['last_ai_time']:%Y-%m-%d %H:%M:%S}")
        st.markdown(st.session_state["last_ai_answer"])
    elif has_key:
        st.info("👆 '지금 해석 생성' 버튼을 눌러 AI 진단을 받아보세요.")


# ===== Watchlist tab =====
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "AVGO", "AMD", "TSM", "ASML", "NFLX", "PLTR",
]

with tab_watchlist:
    st.subheader("관심종목 시그널")
    st.caption("조건: 20MA > 60MA(상승추세) · 현재가가 20MA 근처(±3%) · RSI 30~50")

    user_list = st.text_input("티커 (쉼표로 구분)", value=",".join(WATCHLIST))
    tickers = [t.strip().upper() for t in user_list.split(",") if t.strip()]

    if tickers:
        wdf = fetch_slow(tuple(tickers), period="1y")
        rows = []
        for t in tickers:
            if t not in wdf.columns:
                continue
            s = wdf[t].dropna()
            if len(s) < 80:
                continue
            ma20 = s.rolling(20).mean()
            ma60 = s.rolling(60).mean()
            # RSI(14)
            delta = s.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            price = float(s.iloc[-1])
            dist20 = (price / ma20.iloc[-1] - 1) * 100
            trend_up = ma20.iloc[-1] > ma60.iloc[-1]
            rsi_now = float(rsi.iloc[-1])
            ret_5d = float(s.pct_change(5).iloc[-1] * 100)

            signal = trend_up and abs(dist20) < 3 and 30 < rsi_now < 50
            rows.append({
                "티커": t,
                "현재가": round(price, 2),
                "5일": f"{ret_5d:+.1f}%",
                "20MA 이격": f"{dist20:+.1f}%",
                "추세": "↑" if trend_up else "↓",
                "RSI": round(rsi_now, 1),
                "시그널": "🎯" if signal else "",
            })

        rdf = pd.DataFrame(rows).sort_values("시그널", ascending=False)
        st.dataframe(rdf, use_container_width=True, hide_index=True)

        st.caption("🎯 = 상승추세 + 20MA 지지 근접 + RSI 과매도 반등 구간")
