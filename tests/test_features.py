import numpy as np
import pandas as pd
import pytest
from src.features import build_dataset, feature_columns

def make_df(days=3, bars=40, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for d in pd.bdate_range("2026-01-05", periods=days):
        idx = pd.date_range(f"{d.date()} 09:15", periods=bars, freq="5min", tz="Asia/Kolkata")
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, bars)))
        op = np.r_[close[0], close[:-1]]
        frames.append(pd.DataFrame({
            "open": op, "high": np.maximum(op, close) * 1.0005,
            "low": np.minimum(op, close) * 0.9995, "close": close,
            "volume": rng.integers(1000, 5000, bars).astype(float)}, index=idx))
    return pd.concat(frames)

def test_features_ignore_the_future():
    df = make_df()
    cutoff = df.index[60]
    changed = df.copy()
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close", "volume"]] *= 1.5
    a, b = build_dataset(df), build_dataset(changed)
    feats = feature_columns(a)
    common = a.index[a.index <= cutoff]
    assert len(common) > 0
    pd.testing.assert_frame_equal(a.loc[common, feats], b.loc[common, feats])

@pytest.mark.parametrize("h", [1, 3])
def test_no_signal_near_the_end_of_a_day(h):
    df = make_df()
    ds = build_dataset(df, horizon=h)
    last_bars = df.groupby(df.index.date).tail(h).index
    assert not ds.index.isin(last_bars).any()

@pytest.mark.parametrize("h", [1, 3])
def test_target_matches_the_trade_return(h):
    df = make_df()
    ds = build_dataset(df, horizon=h)
    ts = ds.index[0]
    pos = df.index.get_loc(ts)
    expected = df["close"].iloc[pos + h] / df["open"].iloc[pos + 1] - 1
    assert np.isclose(ds.loc[ts, "fwd_ret"], expected)
    assert ds.loc[ts, "y"] == int(expected > 0)