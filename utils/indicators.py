"""기술 지표 · 팩터 스코어 · 시장심리 계산 모듈."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st

from utils.data import fetch_fast, fetch_slow, get_factor_raw


# ── 시장심리 ──────────────────────────────────────────────────
def pct_rank(series: pd.Series, value: float, lookback: int = 252) -> float:
    s = series.dropna().tail(lookback)
    if len(s) < 20:
        return 50.0
    return float((s < value).mean() * 100)


@dataclass
class Component:
    name: str
    score: float
    value: float
    note: str


def compute_sentiment() -> tuple[float, list[Component]]:
    tickers = ["^VIX", "SPY", "QQQ", "TLT", "HYG", "LQD", "GLD", "UUP", "^TNX"]
    df = fetch_fast(tuple(tickers))
    comps: list[Component] = []

    vix = df["^VIX"].dropna()
    vix_now = float(vix.iloc[-1])
    vix_score = 100 - pct_rank(vix, vix_now)
    comps.append(Component("VIX (변동성)", vix_score, vix_now, "낮을수록 탐욕, 30↑ 공포"))

    spy = df["SPY"].dropna()
    mom = spy.pct_change(125).iloc[-1] * 100
    mom_score = pct_rank(spy.pct_change(125), spy.pct_change(125).iloc[-1])
    comps.append(Component("SPY 125일 모멘텀", mom_score, mom, f"{mom:+.1f}%"))

    ma = spy.rolling(125).mean()
    dist = (spy / ma - 1).dropna()
    dist_now = float(dist.iloc[-1] * 100)
    dist_score = pct_rank(dist, dist.iloc[-1])
    comps.append(Component("SPY 125MA 이격도", dist_score, dist_now, f"{dist_now:+.1f}%"))

    if "HYG" in df and "LQD" in df:
        ratio = (df["HYG"] / df["LQD"]).dropna()
        ratio_score = pct_rank(ratio, ratio.iloc[-1])
        comps.append(Component("정크본드 수요 (HYG/LQD)", ratio_score, float(ratio.iloc[-1]), "높을수록 위험선호"))

    spy_ret = spy.pct_change(20)
    tlt_ret = df["TLT"].pct_change(20)
    diff = (spy_ret - tlt_ret).dropna()
    diff_now = float(diff.iloc[-1] * 100)
    diff_score = pct_rank(diff, diff.iloc[-1])
    comps.append(Component("안전자산 회피 (SPY-TLT 20d)", diff_score, diff_now, f"{diff_now:+.1f}%p"))

    uup = df["UUP"].dropna()
    uup_score = 100 - pct_rank(uup.pct_change(60), uup.pct_change(60).iloc[-1])
    comps.append(Component("달러 강세 (UUP 60d)", uup_score, float(uup.pct_change(60).iloc[-1] * 100), "약달러=위험선호"))

    composite = float(np.mean([c.score for c in comps]))
    return composite, comps


def sentiment_label(score: float) -> tuple[str, str]:
    """(라벨, 색상) 반환."""
    if score < 25:   return "극단적 공포", "#c62828"
    if score < 45:   return "공포", "#ef6c00"
    if score < 55:   return "중립", "#fbc02d"
    if score < 75:   return "탐욕", "#7cb342"
    return "극단적 탐욕", "#2e7d32"


# ── 기술 지표 ─────────────────────────────────────────────────
def compute_indicators(close: pd.Series, vol: pd.Series | None = None) -> dict:
    c = close.dropna()
    out: dict = {}
    if len(c) < 60:
        return out

    ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    ma60 = c.rolling(60).mean()
    ma200 = c.rolling(200).mean() if len(c) >= 200 else pd.Series(index=c.index, dtype=float)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_sig

    bb_mid = ma20
    bb_std = c.rolling(20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    bb_pct = (c - bb_low) / (bb_up - bb_low)

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

    if vol is not None and len(vol.dropna()) > 20:
        v = vol.dropna()
        out["vol_ratio"] = float(v.iloc[-1] / v.tail(20).mean())
    else:
        out["vol_ratio"] = np.nan
    return out


def score_ticker(ind: dict) -> tuple[int, list[str], list[str]]:
    score = 50
    bull: list[str] = []
    bear: list[str] = []

    if ind["trend_up"]:
        score += 12; bull.append("20MA>60MA 상승추세")
    else:
        score -= 12; bear.append("20MA<60MA 하락추세")

    if ind.get("above_200") is True:
        score += 8; bull.append("200MA 위 (장기 강세)")
    elif ind.get("above_200") is False:
        score -= 8; bear.append("200MA 아래 (장기 약세)")

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

    if ind["rsi"] > ind["rsi_prev"] and ind["rsi_prev"] < 45:
        score += 5; bull.append("RSI 저점 반등 전환")

    if ind["macd_cross_up"]:
        score += 10; bull.append("MACD 골든크로스 발생")
    elif ind["macd_hist"] > 0 and ind["macd_hist"] > ind["macd_hist_prev"]:
        score += 5; bull.append("MACD 모멘텀 강화")
    elif ind["macd_hist"] < 0 and ind["macd_hist"] < ind["macd_hist_prev"]:
        score -= 6; bear.append("MACD 모멘텀 약화")

    d20 = ind["dist20"]
    if ind["trend_up"] and -4 < d20 < 1:
        score += 10; bull.append(f"20MA 지지 근접 ({d20:+.1f}%) 눌림목")
    elif d20 > 12:
        score -= 6; bear.append(f"20MA 이격 과대 ({d20:+.1f}%) 추격 위험")

    bb = ind["bb_pct"]
    if bb < 0.2:
        score += 4; bull.append("볼린저 하단 (반등 기대)")
    elif bb > 0.95:
        score -= 4; bear.append("볼린저 상단 돌파 (과열)")

    fh = ind["from_52w_high"]
    if fh > -3:
        score += 4; bull.append(f"52주 신고가 부근 ({fh:+.1f}%)")
    elif fh < -40:
        bear.append(f"52주 고점 대비 {fh:.0f}% (낙폭 과대)")

    vr = ind.get("vol_ratio")
    if vr and vr > 1.8:
        score += 4; bull.append(f"거래량 급증 ({vr:.1f}x)")

    return max(0, min(100, score)), bull, bear


def signal_label(score: int, lang: str = "ko") -> str:
    labels = {
        "ko": {75: "🟢 강력매수", 62: "🟢 매수", 45: "🟡 관망", 32: "🟠 주의", 0: "🔴 회피"},
        "en": {75: "🟢 Strong Buy", 62: "🟢 Buy", 45: "🟡 Neutral", 32: "🟠 Caution", 0: "🔴 Avoid"},
    }
    d = labels.get(lang, labels["ko"])
    for threshold in sorted(d.keys(), reverse=True):
        if score >= threshold:
            return d[threshold]
    return d[0]


# ── 팩터 스코어 ───────────────────────────────────────────────
def momentum_12_1(close: pd.Series) -> float:
    c = close.dropna()
    if len(c) >= 252:
        return float(c.iloc[-21] / c.iloc[-252] - 1) * 100
    if len(c) >= 42:
        return float(c.iloc[-21] / c.iloc[0] - 1) * 100
    return np.nan


def _pctl(series: pd.Series, higher_better: bool = True) -> pd.Series:
    r = series.rank(pct=True)
    return (r * 100) if higher_better else ((1 - r) * 100)


def compute_factor_scores(tickers: list[str], close_df) -> pd.DataFrame:
    rows = []
    for t in tickers:
        fr = get_factor_raw(t)
        mom = momentum_12_1(close_df[t]) if (hasattr(close_df, "columns") and t in close_df.columns) else np.nan
        rows.append({
            "티커": t, "per": fr.get("per"), "psr": fr.get("psr"),
            "ev_ebitda": fr.get("ev_ebitda"), "gross": fr.get("gross_margin"),
            "profit": fr.get("profit_margin"), "growth": fr.get("rev_growth"),
            "roe": fr.get("roe"), "mom": mom,
        })
    df = pd.DataFrame(rows).set_index("티커")

    val_parts = []
    for col in ["per", "psr", "ev_ebitda"]:
        s = pd.to_numeric(df[col], errors="coerce").where(lambda x: x > 0)
        if s.notna().sum() >= 2:
            val_parts.append(_pctl(s, higher_better=False))
    value = pd.concat(val_parts, axis=1).mean(axis=1) if val_parts else pd.Series(np.nan, index=df.index)

    qual_parts = []
    for col in ["gross", "profit", "growth", "roe"]:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= 2:
            qual_parts.append(_pctl(s, higher_better=True))
    quality = pd.concat(qual_parts, axis=1).mean(axis=1) if qual_parts else pd.Series(np.nan, index=df.index)

    ms = pd.to_numeric(df["mom"], errors="coerce")
    momentum = _pctl(ms, higher_better=True) if ms.notna().sum() >= 2 else pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({"밸류": value, "모멘텀": momentum, "퀄리티": quality})
    out = out.fillna(50)
    out["팩터종합"] = out.mean(axis=1)
    return out.round(0)


# ── 포맷 유틸 ─────────────────────────────────────────────────
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
