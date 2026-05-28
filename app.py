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
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Market Dashboard", layout="wide", page_icon="📊")

# ---------- Data fetch ----------
@st.cache_data(ttl=60 * 30)
def fetch(tickers: list[str], period: str = "2y") -> pd.DataFrame:
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna(how="all")


def pct_rank(series: pd.Series, value: float, lookback: int = 252) -> float:
    """How greedy is `value` vs last N days? 0=extreme fear, 100=extreme greed."""
    s = series.dropna().tail(lookback)
    if len(s) < 20:
        return 50.0
    return float((s < value).mean() * 100)


# ---------- Sentiment score ----------
@dataclass
class Component:
    name: str
    score: float  # 0=fear, 100=greed
    value: float
    note: str


def compute_sentiment() -> tuple[float, list[Component]]:
    tickers = ["^VIX", "SPY", "QQQ", "TLT", "HYG", "LQD", "GLD", "UUP", "^TNX"]
    df = fetch(tickers)

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
st.caption(f"마지막 업데이트: {dt.datetime.now():%Y-%m-%d %H:%M} · 데이터 캐시 30분")

tab_sentiment, tab_macro, tab_watchlist = st.tabs(
    ["🧠 시장 심리", "🌐 매크로", "👀 관심종목"]
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
    macro_df = fetch(list(macro_tickers.keys()), period="1y")
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
    sec_df = fetch(list(sectors.keys()), period="6mo")
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
        wdf = fetch(tickers, period="1y")
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
