import numpy as np
import pandas as pd

CAPITAL = 100_000.0
STEP = pd.Timedelta("5min")

def round_trip_cost(fee_bps_round_trip=6.0, slippage_bps_per_side=2.0):
    return (fee_bps_round_trip + 2 * slippage_bps_per_side) / 1e4

def signals_from_proba(p, base_rate, delta):
    s = np.zeros(len(p), dtype=int)
    s[p > base_rate + delta] = 1
    s[p < base_rate - delta] = -1
    return s

def apply_hold(index, signal, hold):
    """One position at a time: skip signals that arrive while a trade is still open."""
    sig = np.asarray(signal).copy()
    free_at = None
    for i, ts in enumerate(index):
        if sig[i] == 0:
            continue
        if free_at is not None and ts < free_at:
            sig[i] = 0
            continue
        free_at = ts + hold * STEP
    return sig

def run_backtest(df, signal, fee_bps_round_trip=6.0, slippage_bps_per_side=2.0,
                 capital=CAPITAL, hold=1):
    """df needs columns 'day' and 'fwd_ret' (return from next open to close of bar t+hold)."""
    assert df.index.is_monotonic_increasing, "sort by time first"
    out = df[["day", "fwd_ret"]].copy()
    out["signal"] = apply_hold(out.index, signal, hold)
    traded = out["signal"] != 0
    out["gross_ret"] = out["signal"] * out["fwd_ret"]
    out["cost"] = np.where(traded, round_trip_cost(fee_bps_round_trip, slippage_bps_per_side), 0.0)
    out["net_ret"] = out["gross_ret"] - out["cost"]
    out["equity"] = capital * (1 + out["net_ret"]).cumprod()
    return out

def _sharpe(daily):
    sd = daily.std(ddof=1)
    return daily.mean() / sd * np.sqrt(252) if pd.notna(sd) and sd > 0 else np.nan

def summarize(bt, capital=CAPITAL):
    eq = np.r_[capital, bt["equity"].values]
    max_dd = (eq / np.maximum.accumulate(eq) - 1).min()
    daily = (1 + bt["net_ret"]).groupby(bt["day"]).prod() - 1
    tr = bt["signal"] != 0
    n = int(tr.sum())
    return {
        "total_return_pct": (eq[-1] / capital - 1) * 100,
        "max_drawdown_pct": max_dd * 100,
        "sharpe": _sharpe(daily),
        "trades": n,
        "hit_rate_pct": (bt.loc[tr, "net_ret"] > 0).mean() * 100 if n else np.nan,
        "avg_net_bps_per_trade": bt.loc[tr, "net_ret"].mean() * 1e4 if n else np.nan,
        "gross_hit_rate_pct": (bt.loc[tr, "gross_ret"] > 0).mean() * 100 if n else np.nan,
        "avg_gross_bps_per_trade": bt.loc[tr, "gross_ret"].mean() * 1e4 if n else np.nan,
    }

def buy_and_hold_summary(df):
    """Hold from the first to the last bar, overnight included, no costs."""
    c = df["close"]
    eq = c / c.iloc[0]
    daily = (1 + c.pct_change().fillna(0)).groupby(df["day"]).prod() - 1
    return {
        "total_return_pct": (eq.iloc[-1] - 1) * 100,
        "max_drawdown_pct": (eq / eq.cummax() - 1).min() * 100,
        "sharpe": _sharpe(daily),
        "trades": 1, "hit_rate_pct": np.nan, "avg_net_bps_per_trade": np.nan,
        "gross_hit_rate_pct": np.nan, "avg_gross_bps_per_trade": np.nan,
    }