"""뉴스 수집 모듈 — Yahoo Finance RSS (무료)."""
from __future__ import annotations
import datetime as dt
import xml.etree.ElementTree as ET
import streamlit as st
import requests


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_ticker_news(ticker: str, max_items: int = 8) -> list[dict]:
    """Yahoo Finance RSS로 종목 뉴스 수집."""
    url = f"https://finance.yahoo.com/rss/headline?s={ticker}"
    try:
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title", "")
            link  = item.findtext("link", "")
            pub   = item.findtext("pubDate", "")
            desc  = item.findtext("description", "")
            # 날짜 파싱
            try:
                pub_dt = dt.datetime.strptime(pub[:25], "%a, %d %b %Y %H:%M:%S")
                pub_str = pub_dt.strftime("%m/%d %H:%M")
            except Exception:
                pub_str = pub[:16]
            items.append({
                "title": title,
                "link": link,
                "date": pub_str,
                "desc": desc[:200],
            })
        return items
    except Exception as e:
        return [{"title": f"뉴스 로드 실패: {e}", "link": "", "date": "", "desc": ""}]


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_market_news(max_items: int = 10) -> list[dict]:
    """Yahoo Finance 시장 전체 뉴스."""
    url = "https://finance.yahoo.com/rss/topstories"
    try:
        r = requests.get(url, timeout=10,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        root = ET.fromstring(r.text)
        items = []
        for item in root.findall(".//item")[:max_items]:
            title = item.findtext("title", "")
            link  = item.findtext("link", "")
            pub   = item.findtext("pubDate", "")
            desc  = item.findtext("description", "")
            try:
                pub_dt = dt.datetime.strptime(pub[:25], "%a, %d %b %Y %H:%M:%S")
                pub_str = pub_dt.strftime("%m/%d %H:%M")
            except Exception:
                pub_str = pub[:16]
            items.append({
                "title": title, "link": link,
                "date": pub_str, "desc": desc[:200],
            })
        return items
    except Exception as e:
        return [{"title": f"뉴스 로드 실패: {e}", "link": "", "date": "", "desc": ""}]
