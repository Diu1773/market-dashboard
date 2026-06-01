"""백테스트 엔진 — 전략 시뮬레이션 + 통계 강화."""
from __future__ import annotations
import numpy as np
import pandas as pd

STRATEGIES = {
    "ma_cross":   "이동평균 교차 (추세추종)",
    "macd":       "MACD 교차 (모멘텀)",
    "rsi_mr":     "RSI 평균회귀 (저점매수)",
    "rsi_trend":  "RSI + 추세필터 (눌림목)",
}

STRATEGIES_EN = {
    "ma_cross":   "MA Crossover (Trend Following)",
    "macd":       "MACD Crossover (Momentum)",
    "rsi_mr":     "RSI Mean Reversion (Buy Dips)",
    "rsi_trend":  "RSI + Trend Filter (Pullback Buy)",
}


def _bt_indicators(c: pd.Series, fast: int, slow: int) -> dict:
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    return {
        "maf": c.rolling(fast).mean(),
        "mas": c.rolling(slow).mean(),
        "rsi": rsi, "macd": macd, "macd_sig": macd_sig,
    }


def generate_positions(c: pd.Series, strategy: str, p: dict) -> pd.Series:
    ind = _bt_indicators(c, p["fast"], p["slow"])
    n = len(c)
    pos = np.zeros(n)

    if strategy == "ma_cross":
        cond = (ind["maf"] > ind["mas"]).values
        pos = np.where(cond, 1.0, 0.0)

    elif strategy == "macd":
        cond = (ind["macd"] > ind["macd_sig"]).values
        pos = np.where(cond, 1.0, 0.0)

    elif strategy in ("rsi_mr", "rsi_trend"):
        rsi = ind["rsi"].values
        trend = (ind["maf"] > ind["mas"]).values
        state = 0
        for i in range(n):
            if np.isnan(rsi[i]):
                pos[i] = state; continue
            if state == 0:
                enter = rsi[i] < p["rsi_buy"]
                if strategy == "rsi_trend":
                    enter = enter and bool(trend[i])
                if enter:
                    state = 1
            elif state == 1:
                if rsi[i] > p["rsi_sell"]:
                    state = 0
            pos[i] = state

    return pd.Series(pos, index=c.index)


def run_backtest(
    c: pd.Series,
    strategy: str,
    p: dict,
    fee: float = 0.001,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
) -> dict:
    c = c.dropna()
    raw_pos = generate_positions(c, strategy, p)

    if stop_loss > 0 or take_profit > 0:
        pos_vals = raw_pos.values.copy()
        prices = c.values
        entry_price = None
        for i in range(len(pos_vals)):
            if pos_vals[i] == 1:
                if entry_price is None:
                    entry_price = prices[i]
                else:
                    chg = prices[i] / entry_price - 1
                    if stop_loss > 0 and chg <= -stop_loss:
                        pos_vals[i] = 0; entry_price = None
                    elif take_profit > 0 and chg >= take_profit:
                        pos_vals[i] = 0; entry_price = None
            else:
                entry_price = None
        raw_pos = pd.Series(pos_vals, index=c.index)

    # 룩어헤드 방지: 신호 다음날 진입
    pos = raw_pos.shift(1).fillna(0)
    ret = c.pct_change().fillna(0)
    trade_chg = pos.diff().abs().fillna(0)
    strat_ret = pos * ret - trade_chg * fee
    equity = (1 + strat_ret).cumprod()
    bh = (1 + ret).cumprod()

    # 거래 분석
    entries = (pos.diff() > 0)
    exits = (pos.diff() < 0)
    trade_returns = []
    in_pos = False
    entry_eq = None
    for i in range(len(pos)):
        if entries.iloc[i] and not in_pos:
            in_pos = True; entry_eq = equity.iloc[i]
        elif exits.iloc[i] and in_pos:
            in_pos = False
            trade_returns.append(equity.iloc[i] / entry_eq - 1)
    if in_pos and entry_eq is not None:
        trade_returns.append(equity.iloc[-1] / entry_eq - 1)

    n_trades = len(trade_returns)
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = (len(wins) / n_trades * 100) if n_trades else 0

    days = (c.index[-1] - c.index[0]).days or 1
    years = days / 365.25
    total_ret = (equity.iloc[-1] - 1) * 100
    bh_ret = (bh.iloc[-1] - 1) * 100
    cagr = ((equity.iloc[-1]) ** (1 / years) - 1) * 100 if years > 0 else 0
    mdd = ((equity / equity.cummax()) - 1).min() * 100
    bh_mdd = ((bh / bh.cummax()) - 1).min() * 100
    sharpe = (strat_ret.mean() / strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0
    sortino_neg = strat_ret[strat_ret < 0].std()
    sortino = (strat_ret.mean() / sortino_neg * np.sqrt(252)) if sortino_neg > 0 else 0
    calmar = abs(cagr / mdd) if mdd != 0 else 0
    exposure = (pos > 0).mean() * 100

    # 평균 손익비
    avg_win = float(np.mean(wins) * 100) if wins else 0
    avg_loss = float(np.mean(losses) * 100) if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else np.inf

    # 월별 수익률
    monthly = strat_ret.resample("ME").sum() * 100
    bh_monthly = ret.resample("ME").sum() * 100

    return {
        "equity": equity, "bh": bh,
        "strat_ret": strat_ret,
        "monthly": monthly, "bh_monthly": bh_monthly,
        "total_ret": total_ret, "bh_ret": bh_ret, "cagr": cagr,
        "mdd": mdd, "bh_mdd": bh_mdd,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "n_trades": n_trades, "win_rate": win_rate, "exposure": exposure,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "trade_returns": trade_returns,
    }


def run_monte_carlo(trade_returns: list[float], n_sim: int = 1000, n_trades: int = 50) -> dict:
    """몬테카를로 시뮬레이션 — 전략 robustness 검증."""
    if len(trade_returns) < 5:
        return {}
    arr = np.array(trade_returns)
    sim_final = []
    sim_mdd = []
    for _ in range(n_sim):
        sample = np.random.choice(arr, size=n_trades, replace=True)
        equity = np.cumprod(1 + sample)
        sim_final.append((equity[-1] - 1) * 100)
        peak = np.maximum.accumulate(equity)
        dd = (equity / peak - 1).min() * 100
        sim_mdd.append(dd)

    sim_final = sorted(sim_final)
    sim_mdd = sorted(sim_mdd)
    return {
        "final_p5":  sim_final[int(n_sim * 0.05)],
        "final_p25": sim_final[int(n_sim * 0.25)],
        "final_p50": sim_final[int(n_sim * 0.50)],
        "final_p75": sim_final[int(n_sim * 0.75)],
        "final_p95": sim_final[int(n_sim * 0.95)],
        "mdd_p50":   sim_mdd[int(n_sim * 0.50)],
        "mdd_p95":   sim_mdd[int(n_sim * 0.95)],
        "prob_profit": sum(1 for x in sim_final if x > 0) / n_sim * 100,
        "n_sim": n_sim,
    }
