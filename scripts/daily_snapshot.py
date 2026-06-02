"""
일일 시장 스냅샷 — GitHub Actions가 매일 실행.
Fear/Greed 점수 + 주요 지표를 data/history.csv 에 append.
대시보드는 이 파일을 읽어서 추세 차트를 그림 (Streamlit Cloud는 파일 휘발성이라 repo 커밋 방식).
"""
from __future__ import annotations
import datetime as dt
import os
import numpy as np
import pandas as pd
import yfinance as yf

HIST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "history.csv")


def pct_rank(series: pd.Series, value: float, lookback: int = 252) -> float:
    s = series.dropna().tail(lookback)
    if len(s) < 20:
        return 50.0
    return float((s < value).mean() * 100)


def compute_snapshot() -> dict:
    tickers = ["^VIX", "SPY", "QQQ", "TLT", "HYG", "LQD", "GLD", "UUP", "^TNX", "USO"]
    df = yf.download(tickers, period="2y", auto_adjust=True, progress=False)["Close"].dropna(how="all")

    scores = []

    vix = df["^VIX"].dropna()
    vix_now = float(vix.iloc[-1])
    vix_score = 100 - pct_rank(vix, vix_now)
    scores.append(vix_score)

    spy = df["SPY"].dropna()
    mom_score = pct_rank(spy.pct_change(125), spy.pct_change(125).iloc[-1])
    scores.append(mom_score)

    ma = spy.rolling(125).mean()
    dist = (spy / ma - 1).dropna()
    scores.append(pct_rank(dist, dist.iloc[-1]))

    if "HYG" in df and "LQD" in df:
        ratio = (df["HYG"] / df["LQD"]).dropna()
        scores.append(pct_rank(ratio, ratio.iloc[-1]))

    spy_ret = spy.pct_change(20)
    tlt_ret = df["TLT"].pct_change(20)
    diff = (spy_ret - tlt_ret).dropna()
    scores.append(pct_rank(diff, diff.iloc[-1]))

    uup = df["UUP"].dropna()
    scores.append(100 - pct_rank(uup.pct_change(60), uup.pct_change(60).iloc[-1]))

    composite = float(np.mean(scores))

    return {
        "date": dt.date.today().isoformat(),
        "fear_greed": round(composite, 1),
        "vix": round(vix_now, 2),
        "spy": round(float(spy.iloc[-1]), 2),
        "qqq": round(float(df["QQQ"].dropna().iloc[-1]), 2),
        "us10y": round(float(df["^TNX"].dropna().iloc[-1]), 2),
        "dxy_uup": round(float(uup.iloc[-1]), 2),
        "gold": round(float(df["GLD"].dropna().iloc[-1]), 2),
        "oil": round(float(df["USO"].dropna().iloc[-1]), 2),
    }


def main():
    snap = compute_snapshot()
    print("Snapshot:", snap)

    path = os.path.abspath(HIST_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        hist = pd.read_csv(path)
        # 같은 날짜 있으면 덮어쓰기
        hist = hist[hist["date"] != snap["date"]]
        hist = pd.concat([hist, pd.DataFrame([snap])], ignore_index=True)
    else:
        hist = pd.DataFrame([snap])

    hist = hist.sort_values("date").tail(730)  # 최근 2년치만 보관
    hist.to_csv(path, index=False)
    print(f"Saved {len(hist)} rows to {path}")


if __name__ == "__main__":
    main()
