"""데이터 수집 모듈 — yfinance 캐시 계층."""
from __future__ import annotations
import datetime as dt
import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

_HIST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "history.csv")


@st.cache_data(ttl=60 * 60)
def load_history() -> pd.DataFrame:
    """GitHub Actions가 쌓은 일일 시장 스냅샷 히스토리."""
    try:
        path = os.path.abspath(_HIST_PATH)
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")
    except Exception:
        return pd.DataFrame()


# ── 내부 구현 ──────────────────────────────────────────────────
def _fetch_impl(tickers: tuple[str, ...], period: str) -> pd.DataFrame:
    df = yf.download(list(tickers), period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    return df.dropna(how="all")


# ── 캐시 계층 (탭별 차등) ──────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_fast(tickers: tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    """1분 캐시 — 시장심리 탭."""
    return _fetch_impl(tickers, period)


@st.cache_data(ttl=60 * 5)
def fetch_macro(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    """5분 캐시 — 매크로/테마 탭."""
    return _fetch_impl(tickers, period)


@st.cache_data(ttl=60 * 15)
def fetch_slow(tickers: tuple[str, ...], period: str = "1y") -> pd.DataFrame:
    """15분 캐시 — 관심종목 탭."""
    return _fetch_impl(tickers, period)


@st.cache_data(ttl=60 * 15)
def fetch_ohlc(ticker: str, period: str = "1y") -> pd.DataFrame:
    """단일 종목 OHLCV."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


@st.cache_data(ttl=30)
def fetch_vix_quote() -> tuple[float, float, str]:
    """VIX 빠른 조회 (30초 캐시)."""
    tk = yf.Ticker("^VIX")
    try:
        fi = tk.fast_info
        last = float(fi["last_price"])
        prev = float(fi["previous_close"])
    except Exception:
        hist = tk.history(period="5d")
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
    chg = (last / prev - 1) * 100
    return last, chg, dt.datetime.now().strftime("%H:%M:%S")


@st.cache_data(ttl=60 * 30)
def get_fundamentals(ticker: str) -> dict:
    """종목 펀더멘털 (30분 캐시)."""
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


@st.cache_data(ttl=60 * 60)
def get_factor_raw(ticker: str) -> dict:
    """팩터 계산용 원시 펀더멘털 (1시간 캐시)."""
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return {}
    return {
        "per": info.get("forwardPE") or info.get("trailingPE"),
        "psr": info.get("priceToSalesTrailing12Months"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "gross_margin": info.get("grossMargins"),
        "profit_margin": info.get("profitMargins"),
        "rev_growth": info.get("revenueGrowth"),
        "roe": info.get("returnOnEquity"),
    }


@st.cache_data(ttl=60 * 60)
def get_earnings_calendar(ticker: str) -> dict:
    """다음 실적 발표일 + 컨센서스."""
    try:
        cal = yf.Ticker(ticker).calendar
        return cal if cal else {}
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60)
def get_quarterly_financials(ticker: str) -> pd.DataFrame:
    """분기 재무제표 (최근 8분기)."""
    try:
        return yf.Ticker(ticker).quarterly_income_stmt
    except Exception:
        return pd.DataFrame()
