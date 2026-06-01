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

tab_ai, tab_sentiment, tab_macro, tab_themes, tab_watchlist, tab_stock = st.tabs(
    ["🤖 AI 해석", "🧠 시장 심리", "🌐 매크로", "🚀 테마", "👀 관심종목", "🔬 종목 분석"]
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


# ---------- Indicator & scoring engine ----------
DEFAULT_WATCHLIST = [
    "RKLB", "PL", "ASTS", "JOBY", "SERV",
    "NVDA", "PLTR", "AMD", "AVGO", "TSM",
]


@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_ohlc(ticker: str, period: str = "1y") -> pd.DataFrame:
    """단일 종목 OHLCV. 캔들차트/지표용."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def compute_indicators(close: pd.Series, vol: pd.Series | None = None) -> dict:
    """다중 지표 한 번에 계산."""
    c = close.dropna()
    out: dict = {}
    if len(c) < 60:
        return out

    ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    ma60 = c.rolling(60).mean()
    ma200 = c.rolling(200).mean() if len(c) >= 200 else pd.Series(index=c.index, dtype=float)

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_sig

    # Bollinger(20,2)
    bb_mid = ma20
    bb_std = c.rolling(20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    bb_pct = (c - bb_low) / (bb_up - bb_low)  # 0=하단, 1=상단

    # ATR(14) 근사 (종가 기준)
    atr = c.diff().abs().rolling(14).mean()

    price = float(c.iloc[-1])
    out.update({
        "price": price,
        "ma10": float(ma10.iloc[-1]),
        "ma20": float(ma20.iloc[-1]),
        "ma50": float(ma50.iloc[-1]),
        "ma60": float(ma60.iloc[-1]),
        "ma200": float(ma200.iloc[-1]) if ma200.notna().any() else np.nan,
        "dist20": (price / ma20.iloc[-1] - 1) * 100,
        "dist50": (price / ma50.iloc[-1] - 1) * 100,
        "rsi": float(rsi.iloc[-1]),
        "rsi_prev": float(rsi.iloc[-2]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "macd_hist_prev": float(macd_hist.iloc[-2]),
        "macd_cross_up": bool(macd.iloc[-2] <= macd_sig.iloc[-2] and macd.iloc[-1] > macd_sig.iloc[-1]),
        "bb_pct": float(bb_pct.iloc[-1]),
        "atr_pct": float(atr.iloc[-1] / price * 100),
        "trend_up": bool(ma20.iloc[-1] > ma60.iloc[-1]),
        "above_200": bool(price > ma200.iloc[-1]) if ma200.notna().any() else None,
        "ret_5d": float(c.pct_change(5).iloc[-1] * 100),
        "ret_20d": float(c.pct_change(20).iloc[-1] * 100),
        "high_52w": float(c.tail(252).max()),
        "low_52w": float(c.tail(252).min()),
    })
    out["from_52w_high"] = (price / out["high_52w"] - 1) * 100

    # 거래량 급증 여부
    if vol is not None and len(vol.dropna()) > 20:
        v = vol.dropna()
        out["vol_ratio"] = float(v.iloc[-1] / v.tail(20).mean())
    else:
        out["vol_ratio"] = np.nan
    return out


def score_ticker(ind: dict) -> tuple[int, list[str], list[str]]:
    """다중 요인 0~100 스코어. (점수, 강세요인, 약세요인)"""
    score = 50
    bull: list[str] = []
    bear: list[str] = []

    # 1) 중기 추세 (20MA vs 60MA)
    if ind["trend_up"]:
        score += 12; bull.append("20MA>60MA 상승추세")
    else:
        score -= 12; bear.append("20MA<60MA 하락추세")

    # 2) 장기 추세 (200MA)
    if ind.get("above_200") is True:
        score += 8; bull.append("200MA 위 (장기 강세)")
    elif ind.get("above_200") is False:
        score -= 8; bear.append("200MA 아래 (장기 약세)")

    # 3) RSI 구간
    rsi = ind["rsi"]
    if 40 <= rsi <= 60:
        score += 6; bull.append(f"RSI {rsi:.0f} 중립 건강")
    elif 30 <= rsi < 40:
        score += 10; bull.append(f"RSI {rsi:.0f} 과매도 반등 유망")
    elif rsi < 30:
        score += 4; bear.append(f"RSI {rsi:.0f} 과매도 (하락강도 주의)")
    elif 60 < rsi <= 70:
        score -= 2; bear.append(f"RSI {rsi:.0f} 다소 과열")
    else:
        score -= 8; bear.append(f"RSI {rsi:.0f} 과매수 (조정 위험)")

    # 4) RSI 방향 전환
    if ind["rsi"] > ind["rsi_prev"] and ind["rsi_prev"] < 45:
        score += 5; bull.append("RSI 저점 반등 전환")

    # 5) MACD
    if ind["macd_cross_up"]:
        score += 10; bull.append("MACD 골든크로스 발생")
    elif ind["macd_hist"] > 0 and ind["macd_hist"] > ind["macd_hist_prev"]:
        score += 5; bull.append("MACD 모멘텀 강화")
    elif ind["macd_hist"] < 0 and ind["macd_hist"] < ind["macd_hist_prev"]:
        score -= 6; bear.append("MACD 모멘텀 약화")

    # 6) 20MA 지지 근접 (눌림목)
    d20 = ind["dist20"]
    if ind["trend_up"] and -4 < d20 < 1:
        score += 10; bull.append(f"20MA 지지 근접 ({d20:+.1f}%) 눌림목")
    elif d20 > 12:
        score -= 6; bear.append(f"20MA 이격 과대 ({d20:+.1f}%) 추격 위험")

    # 7) 볼린저 위치
    bb = ind["bb_pct"]
    if bb < 0.2:
        score += 4; bull.append("볼린저 하단 (반등 기대)")
    elif bb > 0.95:
        score -= 4; bear.append("볼린저 상단 돌파 (과열)")

    # 8) 52주 고점 근접
    fh = ind["from_52w_high"]
    if fh > -3:
        score += 4; bull.append(f"52주 신고가 부근 ({fh:+.1f}%)")
    elif fh < -40:
        bear.append(f"52주 고점 대비 {fh:.0f}% (낙폭 과대)")

    # 9) 거래량
    vr = ind.get("vol_ratio")
    if vr and vr > 1.8:
        score += 4; bull.append(f"거래량 급증 ({vr:.1f}x)")

    score = max(0, min(100, score))
    return int(score), bull, bear


def signal_label(score: int) -> str:
    if score >= 75:
        return "🟢 강력매수"
    if score >= 62:
        return "🟢 매수"
    if score >= 45:
        return "🟡 관망"
    if score >= 32:
        return "🟠 주의"
    return "🔴 회피"


@st.cache_data(ttl=60 * 30, show_spinner=False)
def get_fundamentals(ticker: str) -> dict:
    """yfinance 펀더멘털 (sp-global 대체)."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return {}
    keys = {
        "longName": "회사명", "sector": "섹터", "industry": "산업",
        "marketCap": "시가총액", "enterpriseValue": "EV",
        "trailingPE": "PER(TTM)", "forwardPE": "PER(Fwd)",
        "priceToSalesTrailing12Months": "PSR", "priceToBook": "PBR",
        "enterpriseToRevenue": "EV/Rev", "enterpriseToEbitda": "EV/EBITDA",
        "profitMargins": "순이익률", "grossMargins": "매출총이익률",
        "revenueGrowth": "매출성장률", "earningsGrowth": "이익성장률",
        "totalRevenue": "매출(TTM)", "totalCash": "현금",
        "totalDebt": "부채", "freeCashflow": "FCF", "beta": "베타",
        "targetMeanPrice": "목표주가(평균)", "recommendationKey": "투자의견",
        "numberOfAnalystOpinions": "애널리스트수",
    }
    return {kr: info.get(en) for en, kr in keys.items()}


def make_candle_chart(ticker: str, ohlc: pd.DataFrame) -> go.Figure:
    c = ohlc["Close"]
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    ma200 = c.rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=ohlc.index, open=ohlc["Open"], high=ohlc["High"],
        low=ohlc["Low"], close=ohlc["Close"], name=ticker,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    fig.add_trace(go.Scatter(x=ohlc.index, y=ma20, name="20MA",
                             line=dict(color="#ffa726", width=1)))
    fig.add_trace(go.Scatter(x=ohlc.index, y=ma60, name="60MA",
                             line=dict(color="#42a5f5", width=1)))
    fig.add_trace(go.Scatter(x=ohlc.index, y=ma200, name="200MA",
                             line=dict(color="#ab47bc", width=1)))
    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False, legend=dict(orientation="h"),
    )
    return fig


def fmt_big(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
        if abs(v) >= div:
            return f"${v/div:.2f}{unit}"
    return f"${v:,.0f}"


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{float(v)*100:.1f}%"


# ---------- Watchlist persistence (URL query param) ----------
def load_watchlist() -> list[str]:
    qp = st.query_params.get("wl")
    if qp:
        return [t.strip().upper() for t in qp.split(",") if t.strip()]
    return DEFAULT_WATCHLIST.copy()


def save_watchlist(tickers: list[str]) -> None:
    st.query_params["wl"] = ",".join(tickers)


# ===== Watchlist tab =====
with tab_watchlist:
    st.subheader("👀 관심종목 시그널")

    if "watchlist" not in st.session_state:
        st.session_state.watchlist = load_watchlist()

    # --- 종목 추가/삭제 UI ---
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        new_t = c1.text_input("종목 추가", placeholder="예: AAPL (엔터)",
                              label_visibility="collapsed").strip().upper()
        if c2.button("➕ 추가", use_container_width=True) and new_t:
            if new_t not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_t)
                save_watchlist(st.session_state.watchlist)
                st.rerun()
        if c3.button("🔄 기본값", use_container_width=True,
                     help="기본 관심종목으로 리셋"):
            st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
            save_watchlist(st.session_state.watchlist)
            st.rerun()

        # 현재 종목 칩 (삭제 가능)
        if st.session_state.watchlist:
            chip_cols = st.columns(min(len(st.session_state.watchlist), 8))
            for i, t in enumerate(st.session_state.watchlist):
                if chip_cols[i % 8].button(f"✕ {t}", key=f"del_{t}",
                                           use_container_width=True):
                    st.session_state.watchlist.remove(t)
                    save_watchlist(st.session_state.watchlist)
                    st.rerun()

    tickers = st.session_state.watchlist
    st.caption("💾 관심종목은 URL에 저장돼요 — 이 페이지를 **북마크**하면 다음에도 유지됩니다.")

    # --- 필터 ---
    fcol1, fcol2 = st.columns([2, 3])
    only_signal = fcol1.toggle("매수 시그널(62점↑)만 보기", value=False)

    if tickers:
        with st.spinner("지표 계산중..."):
            wdf = fetch_slow(tuple(tickers), period="1y")
        rows = []
        details: dict[str, tuple] = {}
        for t in tickers:
            if t not in (wdf.columns if hasattr(wdf, "columns") else []):
                continue
            s = wdf[t].dropna()
            ind = compute_indicators(s)
            if not ind:
                continue
            score, bull, bear = score_ticker(ind)
            details[t] = (ind, score, bull, bear)
            rows.append({
                "티커": t,
                "점수": score,
                "시그널": signal_label(score),
                "현재가": round(ind["price"], 2),
                "RSI": round(ind["rsi"], 0),
                "20MA이격": round(ind["dist20"], 1),
                "추세": "↑" if ind["trend_up"] else "↓",
                "MACD": "▲" if ind["macd_hist"] > 0 else "▼",
                "5일%": round(ind["ret_5d"], 1),
                "52H대비": round(ind["from_52w_high"], 0),
            })

        if rows:
            rdf = pd.DataFrame(rows).sort_values("점수", ascending=False)
            if only_signal:
                rdf = rdf[rdf["점수"] >= 62]

            st.dataframe(
                rdf, use_container_width=True, hide_index=True,
                column_config={
                    "점수": st.column_config.ProgressColumn(
                        "점수", min_value=0, max_value=100, format="%d"),
                    "20MA이격": st.column_config.NumberColumn(format="%.1f%%"),
                    "5일%": st.column_config.NumberColumn(format="%.1f%%"),
                    "52H대비": st.column_config.NumberColumn(format="%.0f%%"),
                    "현재가": st.column_config.NumberColumn(format="$%.2f"),
                },
            )

            # --- 종목별 근거 펼쳐보기 ---
            st.subheader("📋 종목별 시그널 근거")
            pick = st.selectbox("종목 선택", rdf["티커"].tolist())
            if pick in details:
                ind, score, bull, bear = details[pick]
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("종합점수", f"{score}", signal_label(score).split()[1])
                mc2.metric("RSI", f"{ind['rsi']:.0f}")
                mc3.metric("20MA 이격", f"{ind['dist20']:+.1f}%")
                mc4.metric("52주高 대비", f"{ind['from_52w_high']:+.0f}%")
                bc1, bc2 = st.columns(2)
                with bc1:
                    st.markdown("**🟢 강세 요인**")
                    for b in bull:
                        st.markdown(f"- {b}")
                    if not bull:
                        st.caption("없음")
                with bc2:
                    st.markdown("**🔴 약세 요인**")
                    for b in bear:
                        st.markdown(f"- {b}")
                    if not bear:
                        st.caption("없음")
                st.info(f"💡 **{pick}** 더 자세히 보려면 → 상단 **🔬 종목 분석** 탭에서 입력")
        else:
            st.warning("데이터를 불러오지 못했습니다. 티커를 확인하세요.")


# ===== Stock deep-dive tab (sp-global 대체) =====
with tab_stock:
    st.subheader("🔬 종목 심층분석")
    st.caption("펀더멘털 + 기술적 + AI 코멘트 · 무료 데이터 기반 tear-sheet")

    dc1, dc2 = st.columns([2, 1])
    ticker = dc1.text_input("티커 입력", value="RKLB").strip().upper()
    period = dc2.selectbox("기간", ["6mo", "1y", "2y", "5y"], index=1)

    if ticker:
        ohlc = fetch_ohlc(ticker, period)
        if ohlc.empty or len(ohlc) < 30:
            st.error(f"'{ticker}' 데이터를 찾을 수 없습니다.")
        else:
            ind = compute_indicators(ohlc["Close"], ohlc.get("Volume"))
            score, bull, bear = score_ticker(ind)
            fund = get_fundamentals(ticker)

            # 상단 요약
            name = fund.get("회사명") or ticker
            st.markdown(f"### {name} ({ticker})")
            sub = []
            if fund.get("섹터"):
                sub.append(fund["섹터"])
            if fund.get("산업"):
                sub.append(fund["산업"])
            if sub:
                st.caption(" · ".join(sub))

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("현재가", f"${ind['price']:.2f}", f"{ind['ret_5d']:+.1f}% (5d)")
            k2.metric("종합점수", f"{score}", signal_label(score).split()[1])
            k3.metric("RSI", f"{ind['rsi']:.0f}")
            k4.metric("시총", fmt_big(fund.get("시가총액")))
            tgt = fund.get("목표주가(평균)")
            if tgt:
                upside = (tgt / ind["price"] - 1) * 100
                k5.metric("목표주가", f"${tgt:.0f}", f"{upside:+.0f}%")
            else:
                k5.metric("목표주가", "N/A")

            # 차트
            st.plotly_chart(make_candle_chart(ticker, ohlc), use_container_width=True)

            # 펀더멘털 표
            st.markdown("#### 💼 펀더멘털")
            fcols = st.columns(4)
            fund_display = [
                ("PER(Fwd)", fund.get("PER(Fwd)"), "x"),
                ("PSR", fund.get("PSR"), "x"),
                ("EV/Rev", fund.get("EV/Rev"), "x"),
                ("EV/EBITDA", fund.get("EV/EBITDA"), "x"),
                ("매출성장률", fund.get("매출성장률"), "%"),
                ("매출총이익률", fund.get("매출총이익률"), "%"),
                ("순이익률", fund.get("순이익률"), "%"),
                ("베타", fund.get("베타"), "x"),
            ]
            for i, (lbl, val, unit) in enumerate(fund_display):
                with fcols[i % 4]:
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        st.metric(lbl, "N/A")
                    elif unit == "%":
                        st.metric(lbl, fmt_pct(val))
                    else:
                        st.metric(lbl, f"{float(val):.1f}x")

            fc2 = st.columns(4)
            fc2[0].metric("매출(TTM)", fmt_big(fund.get("매출(TTM)")))
            fc2[1].metric("현금", fmt_big(fund.get("현금")))
            fc2[2].metric("부채", fmt_big(fund.get("부채")))
            fc2[3].metric("FCF", fmt_big(fund.get("FCF")))

            # 기술적 근거
            st.markdown("#### 📊 기술적 시그널")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("**🟢 강세 요인**")
                for b in bull:
                    st.markdown(f"- {b}")
                if not bull:
                    st.caption("없음")
            with tc2:
                st.markdown("**🔴 약세 요인**")
                for b in bear:
                    st.markdown(f"- {b}")
                if not bear:
                    st.caption("없음")

            # AI 종합 코멘트
            st.markdown("#### 🤖 AI 종합 코멘트")
            if st.button("🔮 AI 분석 생성", type="primary",
                         disabled=_get_groq_key() is None, key="stock_ai"):
                with st.spinner("AI 분석중..."):
                    fund_txt = "\n".join(
                        f"- {k}: {v}" for k, v in fund.items()
                        if v is not None and not (isinstance(v, float) and np.isnan(v))
                    )
                    prompt = f"""다음은 {name}({ticker}) 종목 데이터다.

# 펀더멘털
{fund_txt}

# 기술적 지표
- 현재가: ${ind['price']:.2f}
- 종합 기술점수: {score}/100 ({signal_label(score)})
- RSI: {ind['rsi']:.0f}
- 추세: {'상승(20MA>60MA)' if ind['trend_up'] else '하락'}
- 200MA 위치: {'위' if ind.get('above_200') else '아래' if ind.get('above_200') is False else 'N/A'}
- 20MA 이격: {ind['dist20']:+.1f}%
- MACD 히스토그램: {ind['macd_hist']:.2f}
- 52주 고점 대비: {ind['from_52w_high']:+.1f}%
- 강세요인: {', '.join(bull) if bull else '없음'}
- 약세요인: {', '.join(bear) if bear else '없음'}

일/주봉 스윙 트레이더 관점에서 한국어로 다음을 작성하라 (## 헤더):

## 한 줄 결론
지금 이 종목을 어떻게 봐야 하는지 한 문장.

## 펀더멘털 평가
밸류에이션이 비싼지 싼지, 성장성/수익성은 어떤지. 동종 섹터(우주/방산/AI 등) 맥락에서.

## 기술적 위치
현재 차트상 위치 — 눌림목인지 과열인지, 지지/저항 어디 보는지.

## 진입 전략
(1) 지금 들어갈 자리인지 (2) 들어간다면 분할매수 vs 대기 (3) 손절 기준 어디로 (4) 주의할 리스크.

근거와 함께 명확하게. 과한 디스클레이머 금지. 단, 투자 결정은 본인 책임임을 마지막 한 줄로만 명시."""
                    ans = call_groq(prompt)
                    st.session_state[f"stock_ai_{ticker}"] = ans
            if f"stock_ai_{ticker}" in st.session_state:
                st.markdown(st.session_state[f"stock_ai_{ticker}"])
            elif _get_groq_key() is None:
                st.caption("⚠️ Groq API 키 설정 시 AI 코멘트 사용 가능 (🤖 AI 해석 탭 참고)")
