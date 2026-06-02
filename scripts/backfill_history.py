"""
과거 Fear/Greed 히스토리 백필 — 한 번만 실행.
과거 ~1년을 주간 단위로 계산해서 data/history.csv 시드.
이후엔 daily_snapshot.py가 매일 이어서 append.
"""
from __future__ import annotations
import datetime as dt
import os
import numpy as np
import pandas as pd
import yfinance as yf

HIST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "history.csv")


def rolling_pct_rank(s: pd.Series, lookback: int = 252) -> pd.Series:
    """각 시점에서 과거 lookback 윈도우 내 백분위 (0~100)."""
    return s.rolling(lookback, min_periods=20).apply(
        lambda w: (w < w.iloc[-1]).mean() * 100, raw=False
    )


def main():
    tickers = ["^VIX", "SPY", "QQQ", "TLT", "HYG", "LQD", "GLD", "UUP", "^TNX", "USO"]
    df = yf.download(tickers, period="3y", auto_adjust=True, progress=False)["Close"].dropna(how="all")

    vix = df["^VIX"]
    spy = df["SPY"]
    uup = df["UUP"]

    # 각 컴포넌트 시계열
    vix_score = 100 - rolling_pct_rank(vix)
    mom_score = rolling_pct_rank(spy.pct_change(125))
    dist = spy / spy.rolling(125).mean() - 1
    dist_score = rolling_pct_rank(dist)
    ratio = df["HYG"] / df["LQD"]
    ratio_score = rolling_pct_rank(ratio)
    diff = spy.pct_change(20) - df["TLT"].pct_change(20)
    diff_score = rolling_pct_rank(diff)
    uup_score = 100 - rolling_pct_rank(uup.pct_change(60))

    composite = pd.concat(
        [vix_score, mom_score, dist_score, ratio_score, diff_score, uup_score], axis=1
    ).mean(axis=1)

    # 주간 샘플링 (최근 1년)
    cutoff = df.index[-1] - pd.Timedelta(days=400)
    weekly = composite[composite.index >= cutoff].resample("W").last().dropna()

    rows = []
    for date, fg in weekly.items():
        try:
            rows.append({
                "date": date.date().isoformat(),
                "fear_greed": round(float(fg), 1),
                "vix": round(float(vix.asof(date)), 2),
                "spy": round(float(spy.asof(date)), 2),
                "qqq": round(float(df["QQQ"].asof(date)), 2),
                "us10y": round(float(df["^TNX"].asof(date)), 2),
                "dxy_uup": round(float(uup.asof(date)), 2),
                "gold": round(float(df["GLD"].asof(date)), 2),
                "oil": round(float(df["USO"].asof(date)), 2),
            })
        except Exception:
            continue

    hist = pd.DataFrame(rows).dropna()
    path = os.path.abspath(HIST_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hist.to_csv(path, index=False)
    print(f"Backfilled {len(hist)} weekly rows to {path}")
    print(hist.tail(5).to_string())


if __name__ == "__main__":
    main()
