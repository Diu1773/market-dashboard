"""한국어 / English 다국어 지원."""

TEXTS = {
    "ko": {
        # 탭 이름
        "tab_ai": "🤖 AI 해석",
        "tab_sentiment": "🧠 시장 심리",
        "tab_macro": "🌐 매크로",
        "tab_themes": "🚀 테마",
        "tab_watchlist": "👀 관심종목",
        "tab_stock": "🔬 종목 분석",
        "tab_backtest": "🧪 백테스트",
        "tab_earnings": "📋 실적 프리뷰",

        # 공통
        "loading": "데이터 로딩중...",
        "refresh": "🔄 강제 새로고침",
        "lang_toggle": "🇺🇸 English",
        "generate": "🔮 생성",
        "run": "▶️ 실행",
        "ticker_input": "티커 입력",
        "period": "기간",
        "no_data": "데이터 없음",

        # 시장 심리
        "sentiment_title": "시장 심리 게이지",
        "sentiment_caption": "구성요소 (0=공포 · 100=탐욕)",
        "extreme_fear": "극단적 공포",
        "fear": "공포",
        "neutral": "중립",
        "greed": "탐욕",
        "extreme_greed": "극단적 탐욕",

        # 매크로
        "macro_title": "주요 매크로 지표",
        "sector_title": "섹터 ETF 상대강도",

        # 관심종목
        "watchlist_title": "관심종목 시그널",
        "watchlist_add": "종목 추가",
        "watchlist_add_btn": "➕ 추가",
        "watchlist_reset": "🔄 기본값",
        "watchlist_filter": "매수 시그널(62점↑)만 보기",
        "watchlist_factor": "📐 팩터 점수 포함 (느림)",
        "watchlist_save": "💾 북마크하면 관심종목이 유지됩니다",
        "signal_strong_buy": "🟢 강력매수",
        "signal_buy": "🟢 매수",
        "signal_neutral": "🟡 관망",
        "signal_caution": "🟠 주의",
        "signal_avoid": "🔴 회피",
        "bull_factors": "🟢 강세 요인",
        "bear_factors": "🔴 약세 요인",

        # 종목 분석
        "stock_title": "종목 심층분석",
        "stock_caption": "펀더멘털 + 기술적 + AI · 무료 데이터",
        "fundamental": "💼 펀더멘털",
        "technical": "📊 기술적 시그널",
        "ai_comment": "🤖 AI 종합 코멘트",
        "ai_generate": "🔮 AI 분석 생성",

        # 백테스트
        "bt_title": "백테스트",
        "bt_caption": "과거 데이터로 전략 검증",
        "bt_run": "▶️ 백테스트 실행",
        "bt_params": "⚙️ 파라미터 튜닝",
        "strategy": "전략",
        "commission": "수수료/슬리피지 (%)",
        "stop_loss": "손절 (%, 0=미사용)",
        "take_profit": "익절 (%, 0=미사용)",
        "equity_curve": "수익곡선",
        "win_rate": "승률",
        "max_drawdown": "최대낙폭(MDD)",
        "sharpe": "샤프",
        "trades": "거래",
        "vs_bh": "vs 보유",
        "bt_warning": "⚠️ 과거가 미래를 보장하지 않음 · 오버피팅 주의",

        # 실적 프리뷰
        "ep_title": "실적 프리뷰",
        "ep_caption": "다음 실적 발표 분석 · yfinance + AI",
        "ep_generate": "📋 프리뷰 생성",
        "ep_next": "다음 실적 발표",
        "ep_consensus_rev": "컨센서스 매출",
        "ep_consensus_eps": "컨센서스 EPS",
        "ep_thesis": "투자 핵심",
        "ep_estimates": "컨센서스 추정치",
        "ep_themes": "주목할 포인트",
        "ep_risks": "리스크",
        "ep_competitor": "경쟁사 비교",

        # AI 해석
        "ai_title": "AI 시장 진단",
        "ai_caption": "Groq Llama · 모든 지표 종합 해석 · 10분 캐시",
        "ai_no_key": "Groq API 키 미설정",
        "ai_key_guide": "1. https://console.groq.com/keys 에서 무료 발급\n2. Streamlit Cloud → Settings → Secrets 에 추가:\n```\nGROQ_API_KEY = \"gsk_...\"\n```",
    },
    "en": {
        # tabs
        "tab_ai": "🤖 AI Analysis",
        "tab_sentiment": "🧠 Sentiment",
        "tab_macro": "🌐 Macro",
        "tab_themes": "🚀 Themes",
        "tab_watchlist": "👀 Watchlist",
        "tab_stock": "🔬 Stock Analysis",
        "tab_backtest": "🧪 Backtest",
        "tab_earnings": "📋 Earnings Preview",

        # common
        "loading": "Loading data...",
        "refresh": "🔄 Force Refresh",
        "lang_toggle": "🇰🇷 한국어",
        "generate": "🔮 Generate",
        "run": "▶️ Run",
        "ticker_input": "Ticker",
        "period": "Period",
        "no_data": "No data",

        # sentiment
        "sentiment_title": "Market Sentiment Gauge",
        "sentiment_caption": "Components (0=Fear · 100=Greed)",
        "extreme_fear": "Extreme Fear",
        "fear": "Fear",
        "neutral": "Neutral",
        "greed": "Greed",
        "extreme_greed": "Extreme Greed",

        # macro
        "macro_title": "Key Macro Indicators",
        "sector_title": "Sector ETF Relative Strength",

        # watchlist
        "watchlist_title": "Watchlist Signals",
        "watchlist_add": "Add ticker",
        "watchlist_add_btn": "➕ Add",
        "watchlist_reset": "🔄 Reset",
        "watchlist_filter": "Buy signals only (62+)",
        "watchlist_factor": "📐 Include factor scores (slow)",
        "watchlist_save": "💾 Bookmark this page to save your watchlist",
        "signal_strong_buy": "🟢 Strong Buy",
        "signal_buy": "🟢 Buy",
        "signal_neutral": "🟡 Neutral",
        "signal_caution": "🟠 Caution",
        "signal_avoid": "🔴 Avoid",
        "bull_factors": "🟢 Bullish Factors",
        "bear_factors": "🔴 Bearish Factors",

        # stock
        "stock_title": "Stock Deep Dive",
        "stock_caption": "Fundamentals + Technical + AI · Free data",
        "fundamental": "💼 Fundamentals",
        "technical": "📊 Technical Signals",
        "ai_comment": "🤖 AI Commentary",
        "ai_generate": "🔮 Generate AI Analysis",

        # backtest
        "bt_title": "Backtest",
        "bt_caption": "Validate strategies on historical data",
        "bt_run": "▶️ Run Backtest",
        "bt_params": "⚙️ Parameter Tuning",
        "strategy": "Strategy",
        "commission": "Commission/Slippage (%)",
        "stop_loss": "Stop Loss (%, 0=off)",
        "take_profit": "Take Profit (%, 0=off)",
        "equity_curve": "Equity Curve",
        "win_rate": "Win Rate",
        "max_drawdown": "Max Drawdown",
        "sharpe": "Sharpe",
        "trades": "Trades",
        "vs_bh": "vs B&H",
        "bt_warning": "⚠️ Past performance does not guarantee future results · Beware overfitting",

        # earnings preview
        "ep_title": "Earnings Preview",
        "ep_caption": "Pre-earnings analysis · yfinance + AI",
        "ep_generate": "📋 Generate Preview",
        "ep_next": "Next Earnings",
        "ep_consensus_rev": "Consensus Revenue",
        "ep_consensus_eps": "Consensus EPS",
        "ep_thesis": "Investment Thesis",
        "ep_estimates": "Consensus Estimates",
        "ep_themes": "Key Themes to Watch",
        "ep_risks": "Risks",
        "ep_competitor": "Competitor Comparison",

        # AI
        "ai_title": "AI Market Analysis",
        "ai_caption": "Groq Llama · Synthesizes all indicators · 10-min cache",
        "ai_no_key": "Groq API key not set",
        "ai_key_guide": "1. Get free key at https://console.groq.com/keys\n2. Streamlit Cloud → Settings → Secrets:\n```\nGROQ_API_KEY = \"gsk_...\"\n```",
    }
}


def t(key: str, lang: str = "ko") -> str:
    """텍스트 번역 헬퍼."""
    return TEXTS.get(lang, TEXTS["ko"]).get(key, key)
