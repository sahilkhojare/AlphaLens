import numpy as np, pandas as pd
from src.backtest import run_backtest, summarize

def frame(rets):
    idx = pd.date_range("2026-01-05 09:20", periods=len(rets), freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame({"day": "2026-01-05", "fwd_ret": rets}, index=idx)

def test_no_signal_means_no_change():
    bt = run_backtest(frame([0.01, -0.02]), [0, 0])
    assert (bt["net_ret"] == 0).all() and bt["equity"].iloc[-1] == 100_000

def test_long_trade_pays_cost():
    bt = run_backtest(frame([0.01]), [1], fee_bps_round_trip=6, slippage_bps_per_side=2)
    assert np.isclose(bt["net_ret"].iloc[0], 0.01 - 0.0010)

def test_short_trade_gains_when_price_falls():
    bt = run_backtest(frame([-0.01]), [-1], fee_bps_round_trip=0, slippage_bps_per_side=0)
    assert np.isclose(bt["net_ret"].iloc[0], 0.01)

def test_drawdown_and_total_return():
    bt = run_backtest(frame([0.10, -0.50]), [1, 1], fee_bps_round_trip=0, slippage_bps_per_side=0)
    s = summarize(bt)
    assert np.isclose(s["max_drawdown_pct"], -50.0) and np.isclose(s["total_return_pct"], -45.0)