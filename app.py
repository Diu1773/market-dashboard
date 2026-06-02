"""
Market Dashboard — 올인원 리서치 플랫폼
스택: Streamlit + yfinance + Groq (완전 무료)
"""
from __future__ import annotations
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.i18n import t
from utils.data import (
    fetch_fast, fetch_macro, fetch_slow, fetch_ohlc,
    fetch_vix_quote, get_fundamentals, get_factor_raw,
    get_earnings_calendar, get_quarterly_financials, load_history,
)
from utils.news import fetch_ticker_news, fetch_market_news
from utils.indicators import (
    compute_sentiment, sentiment_label,
    compute_indicators, score_ticker, signal_label,
    compute_factor_scores, fmt_big, fmt_pct, Component,
)
from utils.backtest import run_backtest, run_monte_carlo, STRATEGIES, STRATEGIES_EN
from utils.groq_client import call_groq, get_groq_key

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(page_title="Market Dashboard", layout="wide", page_icon="📊")

# ── 언어 설정 (사이드바) ───────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    lang = st.radio("언어 / Language", ["🇰🇷 한국어", "🇺🇸 English"],
                    index=0, horizontal=True, label_visibility="collapsed")
    lang = "ko" if "한국어" in lang else "en"
    st.divider()
    st.caption("https://diu1773-market.streamlit.app/")

# ── 상단 헤더 ──────────────────────────────────────────────────
st.title("📊 Market Dashboard")

hcol1, hcol2, hcol3, hcol4 = st.columns([2, 2, 2, 2])
try:
    vix_last, vix_chg, vix_ts = fetch_vix_quote()
    hcol1.metric("VIX", f"{vix_last:.2f}", f"{vix_chg:+.2f}%")
    hcol2.caption(f"조회: {vix_ts}")
except Exception:
    hcol1.metric("VIX", "N/A")

if hcol4.button(t("refresh", lang), help="모든 캐시를 비우고 다시 받음"):
    st.cache_data.clear()
    st.rerun()

st.caption(
    f"{'마지막 업데이트' if lang=='ko' else 'Last update'}: {dt.datetime.now():%Y-%m-%d %H:%M} · "
    f"{'캐시' if lang=='ko' else 'Cache'} — "
    f"{'시장심리' if lang=='ko' else 'Sentiment'} 1{'분' if lang=='ko' else 'min'} / "
    f"{'매크로' if lang=='ko' else 'Macro'} 5{'분' if lang=='ko' else 'min'} / "
    f"{'관심종목' if lang=='ko' else 'Watchlist'} 15{'분' if lang=='ko' else 'min'}"
)

# ── 탭 ────────────────────────────────────────────────────────
tab_names = [
    t("tab_ai", lang), t("tab_sentiment", lang), t("tab_macro", lang),
    t("tab_themes", lang), t("tab_watchlist", lang), t("tab_stock", lang),
    t("tab_backtest", lang), t("tab_earnings", lang), "📰 뉴스",
]
tabs = st.tabs(tab_names)
(tab_ai, tab_sentiment, tab_macro, tab_themes,
 tab_watchlist, tab_stock, tab_backtest, tab_earnings, tab_news) = tabs

# ── 테마 ETF 유니버스 ──────────────────────────────────────────
THEMES: dict[str, list[tuple[str, str]]] = {
    "🤖 AI / 반도체": [("SMH","반도체"),("SOXX","반도체"),("AIQ","AI 종합"),("CHAT","생성형 AI"),("BOTZ","AI/로봇")],
    "🦾 로봇 / 피지컬 AI": [("BOTZ","로봇·AI"),("ROBO","로보틱스"),("ARKQ","자율기술"),("IRBO","AI·로봇")],
    "🧬 바이오 / 바이오AI": [("IBB","바이오테크"),("XBI","바이오 SMID"),("ARKG","유전체혁명"),("IDNA","유전체")],
    "🚀 우주 / UAM / 항공": [("ARKX","우주탐사"),("UFO","우주산업"),("ITA","방산·항공"),("PPA","방산·항공")],
    "🛰️ 드론 / 방산": [("ITA","방산"),("PPA","방산"),("XAR","방산")],
    "⚡ 원자력 / 클린에너지": [("URA","우라늄"),("NLR","원자력"),("ICLN","클린에너지"),("TAN","태양광")],
    "🔋 EV / 자율주행": [("DRIV","자율주행"),("LIT","배터리·리튬"),("IDRV","EV")],
    "🛡️ 사이버보안": [("CIBR","사이버보안"),("HACK","사이버보안")],
    "💰 핀테크 / 크립토": [("ARKF","핀테크"),("IBIT","비트코인 ETF"),("BITQ","크립토")],
    "⚛️ 양자 / 신기술": [("QTUM","양자·신기술"),("ARKK","혁신성장"),("ARKW","차세대 인터넷")],
    "🏗️ 인프라 / 리쇼어링": [("PAVE","미국 인프라"),("SLX","철강")],
}

DEFAULT_WATCHLIST = ["RKLB","PL","ASTS","JOBY","SERV","NVDA","PLTR","AMD","AVGO","TSM"]


# ════════════════════════════════════════════════
# TAB: AI 해석
# ════════════════════════════════════════════════
with tab_ai:
    st.subheader(t("ai_title", lang))
    st.caption(t("ai_caption", lang))

    has_key = get_groq_key() is not None
    if not has_key:
        st.warning(t("ai_no_key", lang))
        st.code(t("ai_key_guide", lang))

    if st.button(t("generate", lang) + " AI", type="primary", disabled=not has_key, key="ai_main"):
        with st.spinner(t("loading", lang)):
            score, comps = compute_sentiment()
            macro_df = fetch_macro(("^VIX","UUP","^TNX","GLD","USO","TLT"), period="3mo")
            macro_snap = {
                tk: {"현재": float(macro_df[tk].dropna().iloc[-1]),
                     "20일변동%": float(macro_df[tk].pct_change(20).dropna().iloc[-1]*100)}
                for tk in macro_df.columns
            }
            sectors = {"XLK":"기술","XLY":"임의소비재","XLC":"통신","XLF":"금융",
                       "XLI":"산업","XLE":"에너지","XLB":"소재","XLV":"헬스케어",
                       "XLP":"필수소비재","XLU":"유틸리티","XLRE":"리츠"}
            sec_df = fetch_macro(tuple(sectors.keys()), period="3mo")
            sec_perf = ((sec_df.iloc[-1]/sec_df.iloc[0]-1)*100).sort_values(ascending=False)

            prompt = f"""오늘({dt.date.today()}) 미국 시장 스냅샷:

Fear/Greed: {score:.0f}/100
{chr(10).join(f"- {c.name}: {c.score:.0f}pt ({c.value:.2f})" for c in comps)}

매크로:
{chr(10).join(f"- {k}: {v['현재']:.2f}, 20일 {v['20일변동%']:+.1f}%" for k,v in macro_snap.items())}

섹터 3개월:
{chr(10).join(f"- {sectors.get(tk,tk)}: {v:+.1f}%" for tk,v in sec_perf.items())}

다음 5섹션으로 답변 (## 헤더, 한국어):
## 한 줄 요약
## 시장 심리 진단
## 매크로 환경
## 자금 흐름 / 핫한 테마
## 스윙 트레이더 액션
"""
            answer = call_groq(prompt, lang=lang)
            st.session_state["last_ai"] = (answer, dt.datetime.now())

    if "last_ai" in st.session_state:
        ans, ts = st.session_state["last_ai"]
        st.caption(f"생성: {ts:%H:%M:%S}")
        st.markdown(ans)
    elif has_key:
        st.info("👆 버튼을 눌러 AI 진단을 받아보세요")


# ════════════════════════════════════════════════
# TAB: 시장 심리
# ════════════════════════════════════════════════
with tab_sentiment:
    with st.spinner(t("loading", lang)):
        score, comps = compute_sentiment()

    label, color = sentiment_label(score)
    col1, col2 = st.columns([1, 2])
    with col1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={"text": f"<b>{label}</b>", "font": {"size": 22}},
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
        gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(gauge, use_container_width=True)

    with col2:
        st.subheader(t("sentiment_caption", lang))
        for c in comps:
            cols = st.columns([3, 1, 4])
            cols[0].write(f"**{c.name}**")
            cols[1].write(f"`{c.score:.0f}`")
            cols[2].progress(int(c.score) / 100, text=c.note)

    # ── Fear/Greed 추세 (GitHub Actions가 매일 기록) ──
    hist = load_history()
    if not hist.empty and len(hist) > 2:
        st.subheader("📈 " + ("Fear & Greed 추세" if lang == "ko" else "Fear & Greed Trend"))
        fg_fig = go.Figure()
        # 배경 구간 색칠
        for y0, y1, clr in [(0,25,"#ffebee"),(25,45,"#fff3e0"),(45,55,"#fffde7"),
                            (55,75,"#f1f8e9"),(75,100,"#e8f5e9")]:
            fg_fig.add_hrect(y0=y0, y1=y1, fillcolor=clr, line_width=0, layer="below")
        fg_fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["fear_greed"],
            mode="lines+markers", name="Fear/Greed",
            line=dict(color="#1a1a4e", width=2),
            marker=dict(size=4),
        ))
        # 현재 값 점
        fg_fig.add_trace(go.Scatter(
            x=[hist["date"].iloc[-1]], y=[hist["fear_greed"].iloc[-1]],
            mode="markers", marker=dict(size=12, color=color),
            showlegend=False,
        ))
        fg_fig.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 100], title="Fear ← → Greed"),
            showlegend=False,
        )
        st.plotly_chart(fg_fig, use_container_width=True)
        st.caption(
            f"{'최근' if lang=='ko' else 'Last'} {len(hist)} {'개 기록' if lang=='ko' else 'records'} · "
            f"{'GitHub Actions가 매일 자동 기록' if lang=='ko' else 'Auto-recorded daily via GitHub Actions'}"
        )
    else:
        st.info("📈 " + ("추세 차트는 GitHub Actions가 데이터를 쌓으면 표시됩니다"
                         if lang == "ko" else "Trend chart appears once GitHub Actions accumulates data"))


# ════════════════════════════════════════════════
# TAB: 매크로
# ════════════════════════════════════════════════
with tab_macro:
    st.subheader(t("macro_title", lang))
    macro_tickers = {
        "^VIX": "VIX", "UUP": "달러(UUP)",
        "^TNX": "10Y 국채금리", "GLD": "금(GLD)",
        "USO": "원유(USO)", "TLT": "장기국채(TLT)",
    }
    macro_df = fetch_macro(tuple(macro_tickers.keys()), period="1y")
    cols = st.columns(3)
    for i, (tk, name) in enumerate(macro_tickers.items()):
        if tk not in macro_df.columns:
            continue
        s = macro_df[tk].dropna()
        last = float(s.iloc[-1])
        chg = float(s.pct_change(20).iloc[-1] * 100)
        cols[i % 3].metric(name, f"{last:,.2f}", f"{chg:+.1f}% (20d)")

    st.subheader(t("sector_title", lang))
    sectors = {
        "XLK":"기술","XLY":"임의소비재","XLC":"통신","XLF":"금융",
        "XLI":"산업","XLE":"에너지","XLB":"소재","XLV":"헬스케어",
        "XLP":"필수소비재","XLU":"유틸리티","XLRE":"리츠",
    }
    sec_df = fetch_macro(tuple(sectors.keys()), period="6mo")
    rets = (sec_df.iloc[-1] / sec_df.iloc[0] - 1).sort_values(ascending=True) * 100
    fig = go.Figure(go.Bar(
        x=rets.values,
        y=[f"{sectors.get(t_, t_)} ({t_})" for t_ in rets.index],
        orientation="h",
        marker_color=["#2e7d32" if v > 0 else "#c62828" for v in rets.values],
        text=[f"{v:+.1f}%" for v in rets.values],
        textposition="outside",
    ))
    fig.update_layout(height=420, margin=dict(l=20, r=60, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════
# TAB: 테마
# ════════════════════════════════════════════════
with tab_themes:
    st.subheader("테마별 ETF 성과" if lang == "ko" else "Thematic ETF Performance")
    all_theme_tickers = list(dict.fromkeys(t_ for items in THEMES.values() for t_, _ in items))
    with st.spinner(t("loading", lang)):
        tdf_raw = fetch_macro(tuple(all_theme_tickers), period="ytd")
        tdf3 = fetch_macro(tuple(all_theme_tickers), period="3mo")

    rows = []
    for theme, items in THEMES.items():
        for tk, desc in items:
            if tk not in tdf_raw.columns:
                continue
            s = tdf_raw[tk].dropna()
            s3 = tdf3[tk].dropna() if tk in tdf3.columns else s
            if len(s) < 5:
                continue
            rows.append({
                "테마": theme, "티커": tk, "설명": desc,
                "현재가": round(float(s.iloc[-1]), 2),
                "1개월": round(float(s.pct_change(21).iloc[-1] * 100), 2),
                "3개월": round(float(s3.iloc[-1] / s3.iloc[0] - 1) * 100, 2) if len(s3) > 5 else np.nan,
                "YTD": round(float(s.iloc[-1] / s.iloc[0] - 1) * 100, 2),
            })

    if rows:
        tdf = pd.DataFrame(rows)
        theme_avg = tdf.groupby("테마")["1개월"].mean().sort_values(ascending=True)
        fig_t = go.Figure(go.Bar(
            x=theme_avg.values, y=theme_avg.index, orientation="h",
            marker_color=["#2e7d32" if v > 0 else "#c62828" for v in theme_avg.values],
            text=[f"{v:+.1f}%" for v in theme_avg.values],
            textposition="outside",
        ))
        fig_t.update_layout(height=400, margin=dict(l=20, r=60, t=20, b=20),
                            title="🏆 테마 랭킹 (1개월 자금 흐름)")
        st.plotly_chart(fig_t, use_container_width=True)

        st.subheader("개별 ETF 상세")
        st.dataframe(
            tdf.sort_values("1개월", ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                "1개월": st.column_config.NumberColumn(format="%.1f%%"),
                "3개월": st.column_config.NumberColumn(format="%.1f%%"),
                "YTD": st.column_config.NumberColumn(format="%.1f%%"),
                "현재가": st.column_config.NumberColumn(format="$%.2f"),
            },
        )


# ════════════════════════════════════════════════
# TAB: 관심종목
# ════════════════════════════════════════════════
with tab_watchlist:
    st.subheader(t("watchlist_title", lang))

    if "watchlist" not in st.session_state:
        qp = st.query_params.get("wl")
        st.session_state.watchlist = (
            [tk.strip().upper() for tk in qp.split(",") if tk.strip()]
            if qp else DEFAULT_WATCHLIST.copy()
        )

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        new_t = c1.text_input(t("watchlist_add", lang), placeholder="AAPL",
                              label_visibility="collapsed").strip().upper()
        if c2.button(t("watchlist_add_btn", lang), use_container_width=True) and new_t:
            if new_t not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_t)
                st.query_params["wl"] = ",".join(st.session_state.watchlist)
                st.rerun()
        if c3.button(t("watchlist_reset", lang), use_container_width=True):
            st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
            st.query_params["wl"] = ",".join(DEFAULT_WATCHLIST)
            st.rerun()

        if st.session_state.watchlist:
            chip_cols = st.columns(min(len(st.session_state.watchlist), 8))
            for i, tk in enumerate(st.session_state.watchlist):
                if chip_cols[i % 8].button(f"✕ {tk}", key=f"del_{tk}", use_container_width=True):
                    st.session_state.watchlist.remove(tk)
                    st.query_params["wl"] = ",".join(st.session_state.watchlist)
                    st.rerun()

    tickers = st.session_state.watchlist
    st.caption(t("watchlist_save", lang))

    fc1, fc2 = st.columns([2, 2])
    only_signal = fc1.toggle(t("watchlist_filter", lang))
    use_factors = fc2.toggle(t("watchlist_factor", lang))

    if tickers:
        with st.spinner(t("loading", lang)):
            wdf = fetch_slow(tuple(tickers), period="1y")
        factor_df = None
        if use_factors:
            with st.spinner("팩터 데이터 수집중..."):
                factor_df = compute_factor_scores(tickers, wdf)

        rows = []
        details: dict = {}
        for tk in tickers:
            if tk not in (wdf.columns if hasattr(wdf, "columns") else []):
                continue
            s = wdf[tk].dropna()
            ind = compute_indicators(s)
            if not ind:
                continue
            sc, bull, bear = score_ticker(ind)
            details[tk] = (ind, sc, bull, bear)
            row = {
                "티커": tk, "기술점수": sc,
                "시그널": signal_label(sc, lang),
                "현재가": round(ind["price"], 2),
                "RSI": round(ind["rsi"], 0),
                "20MA이격": round(ind["dist20"], 1),
                "추세": "↑" if ind["trend_up"] else "↓",
                "MACD": "▲" if ind["macd_hist"] > 0 else "▼",
                "52H대비": round(ind["from_52w_high"], 0),
            }
            if factor_df is not None and tk in factor_df.index:
                fr = factor_df.loc[tk]
                row["밸류"] = int(fr["밸류"])
                row["모멘텀"] = int(fr["모멘텀"])
                row["퀄리티"] = int(fr["퀄리티"])
                row["종합"] = int(round(0.5 * sc + 0.5 * fr["팩터종합"]))
            rows.append(row)

        if rows:
            sort_col = "종합" if factor_df is not None else "기술점수"
            rdf = pd.DataFrame(rows).sort_values(sort_col, ascending=False)
            if only_signal:
                rdf = rdf[rdf["기술점수"] >= 62]

            col_cfg = {
                "기술점수": st.column_config.ProgressColumn("기술점수", min_value=0, max_value=100, format="%d"),
                "20MA이격": st.column_config.NumberColumn(format="%.1f%%"),
                "52H대비": st.column_config.NumberColumn(format="%.0f%%"),
                "현재가": st.column_config.NumberColumn(format="$%.2f"),
            }
            if factor_df is not None:
                col_cfg["종합"] = st.column_config.ProgressColumn("종합", min_value=0, max_value=100, format="%d")
            st.dataframe(rdf, use_container_width=True, hide_index=True, column_config=col_cfg)

            st.subheader("📋 " + ("종목별 시그널 근거" if lang == "ko" else "Signal Breakdown"))
            pick = st.selectbox("종목 선택", rdf["티커"].tolist())
            if pick in details:
                ind, sc, bull, bear = details[pick]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("종합점수", f"{sc}", signal_label(sc, lang).split()[1])
                m2.metric("RSI", f"{ind['rsi']:.0f}")
                m3.metric("20MA 이격", f"{ind['dist20']:+.1f}%")
                m4.metric("52주高 대비", f"{ind['from_52w_high']:+.0f}%")
                bc1, bc2 = st.columns(2)
                with bc1:
                    st.markdown(f"**{t('bull_factors', lang)}**")
                    for b in (bull or ["없음"]):
                        st.markdown(f"- {b}")
                with bc2:
                    st.markdown(f"**{t('bear_factors', lang)}**")
                    for b in (bear or ["없음"]):
                        st.markdown(f"- {b}")
                st.info(f"💡 더 자세히 → **{t('tab_stock', lang)}** 탭")


# ════════════════════════════════════════════════
# TAB: 종목 분석
# ════════════════════════════════════════════════
with tab_stock:
    st.subheader(t("stock_title", lang))
    st.caption(t("stock_caption", lang))

    dc1, dc2 = st.columns([2, 1])
    ticker = dc1.text_input(t("ticker_input", lang), value="RKLB", key="stock_ticker").strip().upper()
    period = dc2.selectbox(t("period", lang), ["6mo", "1y", "2y", "5y"], index=1)

    if ticker:
        ohlc = fetch_ohlc(ticker, period)
        if ohlc.empty or len(ohlc) < 30:
            st.error(f"'{ticker}' 데이터 없음")
        else:
            ind = compute_indicators(ohlc["Close"], ohlc.get("Volume"))
            sc, bull, bear = score_ticker(ind)
            fund = get_fundamentals(ticker)
            name = fund.get("회사명") or ticker

            st.markdown(f"### {name} ({ticker})")
            sub = [fund.get("섹터"), fund.get("산업")]
            st.caption(" · ".join(x for x in sub if x))

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("현재가", f"${ind['price']:.2f}", f"{ind['ret_5d']:+.1f}% (5d)")
            k2.metric("기술점수", f"{sc}", signal_label(sc, lang).split()[1])
            k3.metric("RSI", f"{ind['rsi']:.0f}")
            k4.metric("시총", fmt_big(fund.get("시가총액")))
            tgt = fund.get("목표주가(평균)")
            if tgt:
                k5.metric("목표주가", f"${tgt:.0f}", f"{(tgt/ind['price']-1)*100:+.0f}%")
            else:
                k5.metric("목표주가", "N/A")

            # ── 고급 차트 (캔들 + 볼린저 + 거래량 + RSI + MACD) ──
            from plotly.subplots import make_subplots

            c = ohlc["Close"]
            vol = ohlc.get("Volume", pd.Series(dtype=float))

            # 볼린저밴드
            bb_mid = c.rolling(20).mean()
            bb_std = c.rolling(20).std()
            bb_up  = bb_mid + 2 * bb_std
            bb_dn  = bb_mid - 2 * bb_std

            # RSI
            d = c.diff()
            rsi_line = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() /
                                     (-d.clip(upper=0)).rolling(14).mean().replace(0, 1e-9))

            # MACD
            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist_s = macd_line - macd_sig

            # 지지/저항 (로컬 고점/저점)
            try:
                from scipy.signal import argrelextrema
                close_arr = c.values
                local_max_idx = argrelextrema(close_arr, np.greater, order=10)[0]
                local_min_idx = argrelextrema(close_arr, np.less,    order=10)[0]
                resistance_levels = sorted(set([round(close_arr[i], 2) for i in local_max_idx[-5:]]), reverse=True)[:3]
                support_levels    = sorted(set([round(close_arr[i], 2) for i in local_min_idx[-5:]]))[:3]
            except Exception:
                resistance_levels, support_levels = [], []

            has_vol = vol is not None and len(vol.dropna()) > 10

            row_heights = [0.45, 0.15, 0.2, 0.2] if has_vol else [0.5, 0.25, 0.25]
            subplot_rows = 4 if has_vol else 3
            row_titles = (["", "Volume", "RSI", "MACD"] if has_vol
                          else ["", "RSI", "MACD"])

            fig = make_subplots(
                rows=subplot_rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=row_heights,
                subplot_titles=row_titles,
            )

            # 캔들
            fig.add_trace(go.Candlestick(
                x=ohlc.index, open=ohlc["Open"], high=ohlc["High"],
                low=ohlc["Low"], close=c, name=ticker,
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                showlegend=False,
            ), row=1, col=1)

            # 이평선
            for ma, clr, nm in [(20,"#ffa726","20MA"),(60,"#42a5f5","60MA"),(200,"#ab47bc","200MA")]:
                fig.add_trace(go.Scatter(x=ohlc.index, y=c.rolling(ma).mean(),
                                          name=nm, line=dict(color=clr, width=1),
                                          showlegend=True), row=1, col=1)

            # 볼린저
            fig.add_trace(go.Scatter(x=ohlc.index, y=bb_up, name="BB상단",
                                      line=dict(color="#607d8b", width=1, dash="dot"),
                                      showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=ohlc.index, y=bb_dn, name="BB하단",
                                      line=dict(color="#607d8b", width=1, dash="dot"),
                                      fill="tonexty", fillcolor="rgba(96,125,139,0.07)",
                                      showlegend=False), row=1, col=1)

            # 지지/저항 수평선
            for lvl in resistance_levels:
                fig.add_hline(y=lvl, line_color="#ef5350", line_dash="dash",
                              line_width=1, row=1, col=1,
                              annotation_text=f"저항 ${lvl}", annotation_position="right",
                              annotation_font_size=9)
            for lvl in support_levels:
                fig.add_hline(y=lvl, line_color="#26a69a", line_dash="dash",
                              line_width=1, row=1, col=1,
                              annotation_text=f"지지 ${lvl}", annotation_position="right",
                              annotation_font_size=9)

            cur_row = 2
            # 거래량
            if has_vol:
                vol_colors = ["#26a69a" if float(c.iloc[i]) >= float(c.iloc[i-1])
                              else "#ef5350" for i in range(len(c))]
                fig.add_trace(go.Bar(x=ohlc.index, y=vol.values,
                                      name="거래량", marker_color=vol_colors,
                                      showlegend=False), row=cur_row, col=1)
                cur_row += 1

            # RSI
            fig.add_trace(go.Scatter(x=ohlc.index, y=rsi_line, name="RSI",
                                      line=dict(color="#7b1fa2", width=1.5),
                                      showlegend=False), row=cur_row, col=1)
            fig.add_hline(y=70, line_color="#ef5350", line_dash="dot",
                          line_width=1, row=cur_row, col=1)
            fig.add_hline(y=30, line_color="#26a69a", line_dash="dot",
                          line_width=1, row=cur_row, col=1)
            fig.update_yaxes(range=[0, 100], row=cur_row, col=1)
            cur_row += 1

            # MACD
            macd_bar_colors = ["#26a69a" if v >= 0 else "#ef5350"
                               for v in macd_hist_s.values]
            fig.add_trace(go.Bar(x=ohlc.index, y=macd_hist_s.values,
                                  name="MACD Hist", marker_color=macd_bar_colors,
                                  showlegend=False), row=cur_row, col=1)
            fig.add_trace(go.Scatter(x=ohlc.index, y=macd_line, name="MACD",
                                      line=dict(color="#1565c0", width=1.2),
                                      showlegend=False), row=cur_row, col=1)
            fig.add_trace(go.Scatter(x=ohlc.index, y=macd_sig, name="Signal",
                                      line=dict(color="#e65100", width=1.2),
                                      showlegend=False), row=cur_row, col=1)

            fig.update_layout(
                height=620, margin=dict(l=10, r=80, t=30, b=10),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1.02),
                plot_bgcolor="#fafbfc",
            )
            st.plotly_chart(fig, use_container_width=True)

            # 펀더멘털
            st.markdown(f"#### {t('fundamental', lang)}")
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

            # 기술적
            st.markdown(f"#### {t('technical', lang)}")
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"**{t('bull_factors', lang)}**")
                for b in (bull or ["없음"]):
                    st.markdown(f"- {b}")
            with tc2:
                st.markdown(f"**{t('bear_factors', lang)}**")
                for b in (bear or ["없음"]):
                    st.markdown(f"- {b}")

            # ── 종목 요약 카드 ──
            st.markdown("#### 📝 " + ("종목 요약" if lang == "ko" else "Summary"))
            rec = fund.get("투자의견", "N/A")
            rec_emoji = {"strong_buy": "🟢", "buy": "🟢", "hold": "🟡",
                         "underperform": "🟠", "sell": "🔴"}.get(str(rec).lower(), "⚪")
            analyst_n = fund.get("애널리스트수", "?")
            tgt = fund.get("목표주가(평균)")
            upside = f"{(tgt/ind['price']-1)*100:+.0f}%" if tgt else "N/A"

            def _v(val, fmt="x"):
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "N/A"
                return f"{float(val):.1f}{fmt}"

            summary_lines = [
                f"**{name}** ({ticker}) | {fund.get('섹터','N/A')} · {fund.get('산업','N/A')}",
                f"시총 **{fmt_big(fund.get('시가총액'))}** · EV **{fmt_big(fund.get('EV'))}** · FCF **{fmt_big(fund.get('FCF'))}**",
                "밸류에이션: PER(Fwd) **" + _v(fund.get('PER(Fwd)')) + "** · "
                "PSR **" + _v(fund.get('PSR')) + "** · "
                "EV/Rev **" + _v(fund.get('EV/Rev')) + "**",
                "성장: 매출 **" + fmt_pct(fund.get('매출성장률')) + "** · 이익 **" + fmt_pct(fund.get('이익성장률')) +
                "** | 마진: 총이익 **" + fmt_pct(fund.get('매출총이익률')) + "** · 순이익 **" + fmt_pct(fund.get('순이익률')) + "**",
                f"기술: RSI **{ind['rsi']:.0f}** · 추세 **{'↑ 상승' if ind['trend_up'] else '↓ 하락'}** · "
                f"52주高 대비 **{ind['from_52w_high']:+.0f}%** · 베타 **{fund.get('베타', 'N/A')}**",
                ("컨센서스: " + rec_emoji + " **" + str(rec) + "** (" + str(analyst_n) + "명) · "
                 "목표주가 **$" + f"{tgt:.0f}" + "** (" + upside + ")")
                if tgt else
                ("컨센서스: " + rec_emoji + " **" + str(rec) + "** (" + str(analyst_n) + "명)"),
            ]
            with st.container(border=True):
                for line in summary_lines:
                    st.markdown(line)

            # ── AI
            st.markdown(f"#### {t('ai_comment', lang)}")
            if st.button(t("ai_generate", lang), type="primary",
                         disabled=not get_groq_key(), key="stock_ai"):
                with st.spinner("AI 분석중..."):
                    fund_txt = "\n".join(
                        f"- {k}: {v}" for k, v in fund.items()
                        if v is not None and not (isinstance(v, float) and np.isnan(v))
                    )
                    prompt = f"""{name}({ticker}) 분석:

펀더멘털:
{fund_txt}

기술:
- 현재가: ${ind['price']:.2f}
- 기술점수: {sc}/100 ({signal_label(sc, lang)})
- RSI: {ind['rsi']:.0f}
- 추세: {'상승' if ind['trend_up'] else '하락'}
- 200MA: {'위' if ind.get('above_200') else '아래'}
- 강세: {', '.join(bull) if bull else '없음'}
- 약세: {', '.join(bear) if bear else '없음'}

## 한 줄 결론
## 펀더멘털 평가
## 기술적 위치
## 진입 전략 (분할매수/손절/리스크)

한국어, 마크다운, 디스클레이머 마지막 1줄만."""
                    ans = call_groq(prompt, lang=lang)
                    st.session_state[f"stock_ai_{ticker}"] = ans
            if f"stock_ai_{ticker}" in st.session_state:
                st.markdown(st.session_state[f"stock_ai_{ticker}"])


# ════════════════════════════════════════════════
# TAB: 백테스트
# ════════════════════════════════════════════════
with tab_backtest:
    st.subheader(t("bt_title", lang))
    st.caption(t("bt_caption", lang))

    strat_map = STRATEGIES_EN if lang == "en" else STRATEGIES

    bc1, bc2, bc3 = st.columns([2, 2, 1])
    bt_ticker = bc1.text_input("티커", value="QQQ", key="bt_ticker").strip().upper()
    bt_strategy = bc2.selectbox(t("strategy", lang), list(strat_map.keys()),
                                format_func=lambda k: strat_map[k])
    bt_period = bc3.selectbox(t("period", lang), ["2y", "5y", "10y", "max"], index=1)

    with st.expander(t("bt_params", lang), expanded=True):
        pc1, pc2, pc3, pc4 = st.columns(4)
        p_fast = pc1.slider("단기 MA", 5, 50, 20)
        p_slow = pc2.slider("장기 MA", 20, 200, 60)
        p_rsi_buy = pc3.slider("RSI 매수선", 20, 50, 35)
        p_rsi_sell = pc4.slider("RSI 매도선", 50, 80, 65)
        sc1, sc2, sc3 = st.columns(3)
        p_fee = sc1.slider(t("commission", lang), 0.0, 0.5, 0.1, 0.05) / 100
        p_sl = sc2.slider(t("stop_loss", lang), 0, 30, 0) / 100
        p_tp = sc3.slider(t("take_profit", lang), 0, 100, 0) / 100

    if st.button(t("bt_run", lang), type="primary"):
        with st.spinner("시뮬레이션 중..."):
            ohlc = fetch_ohlc(bt_ticker, bt_period)
            if ohlc.empty or len(ohlc) < 100:
                st.error(f"'{bt_ticker}' 데이터 부족")
            else:
                params = {"fast": p_fast, "slow": p_slow,
                          "rsi_buy": p_rsi_buy, "rsi_sell": p_rsi_sell}
                res = run_backtest(ohlc["Close"], bt_strategy, params,
                                   fee=p_fee, stop_loss=p_sl, take_profit=p_tp)
                mc = run_monte_carlo(res["trade_returns"], n_sim=1000,
                                     n_trades=max(len(res["trade_returns"]), 20))
                st.session_state["bt_result"] = (bt_ticker, bt_strategy, res, mc)

    if "bt_result" in st.session_state:
        tk, strat, res, mc = st.session_state["bt_result"]
        beat = res["total_ret"] - res["bh_ret"]

        # 성과 카드
        m = st.columns(4)
        m[0].metric("전략 수익률", f"{res['total_ret']:+.1f}%", f"{beat:+.1f}%p vs B&H")
        m[1].metric("Buy&Hold", f"{res['bh_ret']:+.1f}%")
        m[2].metric("CAGR", f"{res['cagr']:+.1f}%")
        m[3].metric(t("sharpe", lang), f"{res['sharpe']:.2f}")

        m2 = st.columns(4)
        m2[0].metric(t("max_drawdown", lang), f"{res['mdd']:.1f}%",
                     f"B&H {res['bh_mdd']:.1f}%", delta_color="off")
        m2[1].metric(t("win_rate", lang), f"{res['win_rate']:.0f}%",
                     f"{res['n_trades']}회 {t('trades', lang)}")
        m2[2].metric("Sortino", f"{res['sortino']:.2f}")
        m2[3].metric("Profit Factor", f"{res['profit_factor']:.2f}x"
                     if res['profit_factor'] != np.inf else "∞")

        # 수익곡선
        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"].values,
                                    name="전략", line=dict(color="#2e7d32", width=2)))
        eq_fig.add_trace(go.Scatter(x=res["bh"].index, y=res["bh"].values,
                                    name="Buy & Hold", line=dict(color="#90a4ae", width=1.5, dash="dot")))
        eq_fig.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10),
                             title=f"{tk} · {strat_map[strat]}")
        st.plotly_chart(eq_fig, use_container_width=True)

        # 월별 수익률 히트맵 스타일
        if len(res["monthly"]) > 0:
            st.subheader("📅 월별 수익률")
            monthly_df = pd.DataFrame({
                "전략": res["monthly"],
                "B&H": res["bh_monthly"],
            }).dropna()
            monthly_df.index = monthly_df.index.strftime("%Y-%m")
            monthly_df = monthly_df.tail(24)
            mfig = go.Figure()
            mfig.add_trace(go.Bar(x=monthly_df.index, y=monthly_df["전략"],
                                  name="전략",
                                  marker_color=monthly_df["전략"].apply(lambda v: "#2e7d32" if v >= 0 else "#c62828")))
            mfig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df["B&H"],
                                      name="B&H", line=dict(color="#90a4ae", width=1.5)))
            mfig.update_layout(height=280, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(mfig, use_container_width=True)

        # 몬테카를로
        if mc:
            st.subheader("🎲 몬테카를로 시뮬레이션")
            st.caption(f"{mc['n_sim']}회 무작위 샘플링 — 전략 robustness 검증")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("수익 확률", f"{mc['prob_profit']:.0f}%")
            mc2.metric("중간 수익률 (P50)", f"{mc['final_p50']:+.1f}%")
            mc3.metric("낙관 (P75)", f"{mc['final_p75']:+.1f}%")
            mc4.metric("비관 (P25)", f"{mc['final_p25']:+.1f}%")

            mc_fig = go.Figure()
            mc_fig.add_shape(type="rect",
                x0=0, x1=1, y0=mc["final_p25"], y1=mc["final_p75"],
                xref="paper", fillcolor="#e8f5e9", line_width=0, opacity=0.5)
            mc_fig.add_shape(type="rect",
                x0=0, x1=1, y0=mc["final_p5"], y1=mc["final_p95"],
                xref="paper", fillcolor="#e8f5e9", line_width=0, opacity=0.2)
            mc_fig.add_hline(y=mc["final_p50"], line_color="#2e7d32", line_width=2,
                             annotation_text=f"P50: {mc['final_p50']:+.1f}%")
            mc_fig.add_hline(y=0, line_color="#90a4ae", line_dash="dot")
            mc_fig.update_layout(height=220, margin=dict(l=10,r=10,t=20,b=10),
                                 showlegend=False,
                                 yaxis_title="최종 수익률 (%)")
            st.plotly_chart(mc_fig, use_container_width=True)
            st.caption(f"P5~P95 범위: {mc['final_p5']:+.1f}% ~ {mc['final_p95']:+.1f}% · "
                       f"예상 MDD P50: {mc['mdd_p50']:.1f}%")

        # 판정
        st.markdown("---")
        verdicts = []
        if beat > 0 and res["sharpe"] > 0.8:
            verdicts.append("✅ **B&H를 이겼고 Sharpe도 양호** — 통계적 엣지 있어 보임")
        elif beat > 0:
            verdicts.append("🟡 B&H보다 수익은 높지만 Sharpe 평범")
        else:
            verdicts.append("🔴 **B&H보다 못함** — 이 종목/기간엔 이 전략 엣지 없음")
        if res["mdd"] > res["bh_mdd"]:
            verdicts.append(f"⚠️ MDD가 B&H보다 큼 ({res['mdd']:.0f}% vs {res['bh_mdd']:.0f}%)")
        else:
            verdicts.append(f"🛡️ MDD를 줄임 ({res['mdd']:.0f}% vs {res['bh_mdd']:.0f}%)")
        if mc and mc["prob_profit"] < 55:
            verdicts.append(f"⚠️ 몬테카를로 수익확률 {mc['prob_profit']:.0f}% — 불안정한 전략")
        for v in verdicts:
            st.markdown(v)
        st.caption(t("bt_warning", lang))


# ════════════════════════════════════════════════
# TAB: 실적 프리뷰 (NEW)
# ════════════════════════════════════════════════
with tab_earnings:
    st.subheader(t("ep_title", lang))
    st.caption(t("ep_caption", lang))

    ec1, ec2 = st.columns([2, 1])
    ep_ticker = ec1.text_input(t("ticker_input", lang), value="NVDA", key="ep_ticker").strip().upper()

    if ep_ticker:
        cal = get_earnings_calendar(ep_ticker)
        fund_ep = get_fundamentals(ep_ticker)
        qfin = get_quarterly_financials(ep_ticker)

        # 다음 실적 정보
        next_date = None
        if cal:
            ed = cal.get("Earnings Date")
            if ed:
                next_date = ed[0] if isinstance(ed, list) else ed

        info_cols = st.columns(4)
        info_cols[0].metric(t("ep_next", lang),
                            str(next_date) if next_date else "N/A")
        cons_rev = cal.get("Revenue Average") if cal else None
        cons_eps = cal.get("Earnings Average") if cal else None
        info_cols[1].metric(t("ep_consensus_rev", lang),
                            f"${cons_rev/1e9:.1f}B" if cons_rev else "N/A")
        info_cols[2].metric(t("ep_consensus_eps", lang),
                            f"${cons_eps:.2f}" if cons_eps else "N/A")
        info_cols[3].metric("목표주가",
                            f"${fund_ep.get('목표주가(평균)',0):.0f}" if fund_ep.get("목표주가(평균)") else "N/A",
                            fund_ep.get("투자의견",""))

        # 분기 재무 차트
        if not qfin.empty:
            periods = [str(c)[:10] for c in qfin.columns[:6]]
            periods_label = [f"Q{i+1}" for i in range(len(periods))][::-1]

            rev_row = qfin.loc["Total Revenue"] if "Total Revenue" in qfin.index else None
            eps_row = qfin.loc["Diluted EPS"] if "Diluted EPS" in qfin.index else None
            gp_row  = qfin.loc["Gross Profit"] if "Gross Profit" in qfin.index else None
            oi_row  = qfin.loc["Operating Income"] if "Operating Income" in qfin.index else None

            if rev_row is not None:
                rev_vals = [float(v)/1e9 for v in rev_row.head(6).values][::-1]
                eps_vals = [float(v) for v in eps_row.head(6).values][::-1] if eps_row is not None else []

                fig_ep = go.Figure()
                fig_ep.add_trace(go.Bar(x=periods_label, y=rev_vals,
                                        name="매출 ($B)", marker_color="rgba(26,26,78,0.6)",
                                        yaxis="y"))
                if eps_vals:
                    fig_ep.add_trace(go.Scatter(x=periods_label, y=eps_vals,
                                                name="EPS ($)", mode="lines+markers",
                                                line=dict(color="#e67e22", width=2.5),
                                                yaxis="y2"))
                fig_ep.update_layout(
                    height=340, margin=dict(l=10,r=50,t=30,b=10),
                    title=f"{ep_ticker} — 분기 매출 & EPS",
                    yaxis=dict(title="매출 ($B)"),
                    yaxis2=dict(title="EPS ($)", overlaying="y", side="right", showgrid=False),
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_ep, use_container_width=True)

                # 마진 차트
                if gp_row is not None and oi_row is not None:
                    gross_m = [float(gp)/float(rv)*100
                               for gp, rv in zip(gp_row.head(6).values, rev_row.head(6).values)][::-1]
                    op_m = [float(oi)/float(rv)*100
                            for oi, rv in zip(oi_row.head(6).values, rev_row.head(6).values)][::-1]
                    fig_m = go.Figure()
                    fig_m.add_trace(go.Scatter(x=periods_label, y=gross_m,
                                               name="매출총이익률", mode="lines+markers",
                                               line=dict(color="#3366cc", width=2)))
                    fig_m.add_trace(go.Scatter(x=periods_label, y=op_m,
                                               name="영업이익률", mode="lines+markers",
                                               line=dict(color="#0d9488", width=2)))
                    fig_m.update_layout(height=260, margin=dict(l=10,r=10,t=20,b=10),
                                        yaxis=dict(title="마진 (%)"),
                                        legend=dict(orientation="h"))
                    st.plotly_chart(fig_m, use_container_width=True)

        # 컨센서스 추정치 표
        if cal and (cons_rev or cons_eps):
            st.subheader(t("ep_estimates", lang))
            rev_low = cal.get("Revenue Low")
            rev_high = cal.get("Revenue High")
            eps_low = cal.get("Earnings Low")
            eps_high = cal.get("Earnings High")
            est_data = {
                "지표": ["매출 ($B)", "EPS ($)"],
                "컨센서스": [
                    f"${cons_rev/1e9:.1f}B" if cons_rev else "N/A",
                    f"${cons_eps:.2f}" if cons_eps else "N/A",
                ],
                "범위 Low": [
                    f"${rev_low/1e9:.1f}B" if rev_low else "N/A",
                    f"${eps_low:.2f}" if eps_low else "N/A",
                ],
                "범위 High": [
                    f"${rev_high/1e9:.1f}B" if rev_high else "N/A",
                    f"${eps_high:.2f}" if eps_high else "N/A",
                ],
            }
            st.dataframe(pd.DataFrame(est_data), use_container_width=True, hide_index=True)

        # AI 실적 프리뷰 분석
        st.subheader(t("ep_themes", lang))
        if st.button(t("ep_generate", lang), type="primary",
                     disabled=not get_groq_key(), key="ep_ai"):
            with st.spinner("AI 분석중 (30초 소요)..."):
                fund_txt = "\n".join(
                    f"- {k}: {v}" for k, v in fund_ep.items()
                    if v is not None and not (isinstance(v, float) and np.isnan(v))
                )
                rev_summary = ""
                if not qfin.empty and "Total Revenue" in qfin.index:
                    rv = qfin.loc["Total Revenue"].head(4)
                    rev_summary = "최근 4분기 매출: " + " / ".join(
                        f"{str(d)[:7]}: ${float(v)/1e9:.1f}B" for d,v in rv.items()
                    )

                prompt = f"""{ep_ticker} 다음 실적 발표 프리뷰 작성:

펀더멘털:
{fund_txt}

{rev_summary}

다음 실적 발표: {next_date or 'N/A'}
컨센서스 매출: ${cons_rev/1e9:.1f}B if {cons_rev} else N/A
컨센서스 EPS: ${cons_eps:.2f} if {cons_eps} else N/A

## 핵심 투자 논리 (2-3문장)
## 컨센서스 달성 가능성 (불/중립/강세)
## 주목할 포인트 3가지 (불릿)
-
## 잠재 리스크 (불릿)
-
## 스윙 트레이더 관점 포지셔닝

한국어, 마크다운, 간결하게."""
                ans = call_groq(prompt, lang=lang)
                st.session_state[f"ep_ai_{ep_ticker}"] = ans

        if f"ep_ai_{ep_ticker}" in st.session_state:
            st.markdown(st.session_state[f"ep_ai_{ep_ticker}"])
        elif get_groq_key() is None:
            st.caption("⚠️ Groq API 키 설정 시 AI 분석 가능")

        # 경쟁사 비교
        st.subheader(t("ep_competitor", lang))
        sector_peers = {
            "Semiconductors": ["NVDA","AMD","INTC","AVGO","QCOM","TSM"],
            "Software": ["MSFT","GOOGL","ORCL","SAP","CRM"],
            "Consumer": ["AAPL","META","AMZN","NFLX"],
        }
        sector = fund_ep.get("섹터", "")
        auto_peers = ["NVDA","AMD","AVGO","QCOM","TSM"]  # default
        for k, v in sector_peers.items():
            if ep_ticker in v:
                auto_peers = [t_ for t_ in v if t_ != ep_ticker][:5]
                break

        peer_list = [ep_ticker] + auto_peers[:4]
        peer_info = {}
        for p_ in peer_list:
            try:
                pi = yf.Ticker(p_).info
                peer_info[p_] = {
                    "시총($B)": round(pi.get("marketCap",0)/1e9, 1),
                    "PER(Fwd)": round(pi.get("forwardPE",0) or 0, 1),
                    "PER(TTM)": round(pi.get("trailingPE",0) or 0, 1),
                    "매출성장률": f"{(pi.get('revenueGrowth',0) or 0)*100:.1f}%",
                    "순이익률": f"{(pi.get('profitMargins',0) or 0)*100:.1f}%",
                    "목표주가": f"${pi.get('targetMeanPrice',0):.0f}" if pi.get("targetMeanPrice") else "N/A",
                }
            except Exception:
                pass

        if peer_info:
            peer_df = pd.DataFrame(peer_info).T.reset_index().rename(columns={"index":"티커"})
            st.dataframe(peer_df, use_container_width=True, hide_index=True)

import yfinance as yf  # peer comparison에서 직접 사용


# ════════════════════════════════════════════════
# TAB: 뉴스
# ════════════════════════════════════════════════
with tab_news:
    st.subheader("📰 " + ("뉴스 분석" if lang == "ko" else "News Analysis"))

    ncol1, ncol2 = st.columns([3, 1])
    news_mode = ncol1.radio(
        "모드",
        ["관심종목 뉴스", "시장 전체 뉴스"] if lang == "ko" else ["Watchlist News", "Market News"],
        horizontal=True, label_visibility="collapsed",
    )

    if "관심종목" in news_mode or "Watchlist" in news_mode:
        # 관심종목 뉴스
        wl = st.session_state.get("watchlist", DEFAULT_WATCHLIST[:5])
        news_ticker = st.selectbox("종목 선택", wl, key="news_ticker")

        col_news, col_ai = st.columns([3, 2])

        with col_news:
            with st.spinner("뉴스 수집중..."):
                articles = fetch_ticker_news(news_ticker, max_items=10)

            st.markdown(f"**{news_ticker} 최신 뉴스**")
            for art in articles:
                if not art["title"]:
                    continue
                with st.container(border=True):
                    ncols = st.columns([5, 1])
                    if art["link"]:
                        ncols[0].markdown(f"**[{art['title']}]({art['link']})**")
                    else:
                        ncols[0].markdown(f"**{art['title']}**")
                    ncols[1].caption(art["date"])
                    if art["desc"]:
                        st.caption(art["desc"][:150] + "...")

        with col_ai:
            st.markdown("**🤖 AI 감성 분석**")
            has_key = get_groq_key() is not None
            if not has_key:
                st.warning("Groq API 키 필요")
            else:
                if st.button("분석 시작", key="news_ai_btn", type="primary"):
                    with st.spinner("AI 분석중..."):
                        headlines = "\n".join(
                            f"- {a['title']}" for a in articles
                            if a["title"] and "실패" not in a["title"]
                        )
                        prompt = f"""{news_ticker} 최신 뉴스 헤드라인:
{headlines}

다음 형식으로 한국어 분석:
## 감성 판단
🟢 긍정 / 🟡 중립 / 🔴 부정 중 하나로 결론 + 이유 1문장

## 주요 테마 (불릿 3개)
-
## 주가 영향 예상
단기(1~2주) 관점에서 이 뉴스들이 주가에 미칠 영향.

## 주목할 뉴스
가장 중요한 헤드라인 1개와 그 이유.

간결하게, 투자 판단에 도움되는 내용만."""
                        ans = call_groq(prompt, lang=lang)
                        st.session_state[f"news_ai_{news_ticker}"] = ans

                if f"news_ai_{news_ticker}" in st.session_state:
                    st.markdown(st.session_state[f"news_ai_{news_ticker}"])
                else:
                    st.info("버튼을 눌러 AI 분석을 받아보세요")

    else:
        # 시장 전체 뉴스
        with st.spinner("시장 뉴스 수집중..."):
            market_articles = fetch_market_news(max_items=15)

        col_mn, col_mai = st.columns([3, 2])

        with col_mn:
            st.markdown("**오늘의 시장 주요 뉴스**")
            for art in market_articles:
                if not art["title"]:
                    continue
                with st.container(border=True):
                    mcols = st.columns([5, 1])
                    if art["link"]:
                        mcols[0].markdown(f"[{art['title']}]({art['link']})")
                    else:
                        mcols[0].markdown(art["title"])
                    mcols[1].caption(art["date"])

        with col_mai:
            st.markdown("**🤖 시장 요약**")
            if not get_groq_key():
                st.warning("Groq API 키 필요")
            else:
                if st.button("시장 요약 생성", key="market_news_ai", type="primary"):
                    with st.spinner("AI 분석중..."):
                        headlines = "\n".join(
                            f"- {a['title']}" for a in market_articles
                            if a["title"] and "실패" not in a["title"]
                        )
                        prompt = f"""오늘 미국 시장 주요 뉴스:
{headlines}

## 오늘 시장 한 줄 요약
## 섹터별 영향 (2-3개 섹터)
## 스윙 트레이더 주목 포인트
내일 주가에 영향 줄 이슈 2가지.

한국어, 간결하게."""
                        ans = call_groq(prompt, lang=lang)
                        st.session_state["market_news_ai"] = ans

                if "market_news_ai" in st.session_state:
                    st.markdown(st.session_state["market_news_ai"])
                else:
                    st.info("버튼을 눌러 시장 요약을 받아보세요")
