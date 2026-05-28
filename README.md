# Market Dashboard

미장 시장 심리 / 매크로 / 관심종목 시그널 대시보드. 완전 무료 스택.

## 스택
- **Streamlit** (UI) — Streamlit Community Cloud 무료 호스팅
- **yfinance** (데이터) — 무료
- **plotly** (차트)

## 로컬 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 구성
1. **시장 심리** — VIX, 모멘텀, 정크본드, 안전자산 회피, 달러 등 6개 컴포넌트로 0~100 fear/greed 스코어
2. **매크로** — 주요 지표 + 섹터 ETF 6개월 상대강도
3. **관심종목** — 20/60MA + RSI 기반 진입 시그널 스캐너

## 배포 (Streamlit Cloud)
1. https://share.streamlit.io → Sign in with GitHub
2. New app → repo `Diu1773/market-dashboard`, branch `main`, file `app.py`
3. Deploy
