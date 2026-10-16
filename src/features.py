import numpy as np
import pandas as pd

FEATURES_BASE = ["ret_1", "ret_3", "ret_6", "ema9_ratio", "ema20_ratio",
                 "rsi_14", "vol_12", "minute_of_day"]
WARMUP = 14          # bars skipped at the start of each day
STEP = pd.Timedelta("5min")

def feature_columns(d):
    return FEATURES_BASE + (["rel_vol"] if "rel_vol" in d.columns else [])

def build_dataset(df):
    d = df.copy()
    d["day"] = d.index.date
    g = d.groupby("day", sort=False)
    d["bar_no"] = g.cumcount()

    # Past-only features, computed inside each day
    for n in (1, 3, 6):
        d[f"ret_{n}"] = g["close"].pct_change(n)
    for span in (9, 20):
        ema = g["close"].transform(lambda s, sp=span: s.ewm(span=sp, adjust=False).mean())
        d[f"ema{span}_ratio"] = d["close"] / ema - 1
    delta = g["close"].diff()
    gain = delta.clip(lower=0).groupby(d["day"]).transform(lambda s: s.rolling(14).mean())
    loss = (-delta.clip(upper=0)).groupby(d["day"]).transform(lambda s: s.rolling(14).mean())
    d["rsi_14"] = 100 - 100 / (1 + gain / loss)
    d["vol_12"] = d.groupby("day")["ret_1"].transform(lambda s: s.rolling(12).std())
    d["minute_of_day"] = d.index.hour * 60 + d.index.minute
    if "volume" in d.columns and d["volume"].sum() > 0:
        m = d.groupby("day")["volume"].transform(lambda s: s.rolling(12).mean())
        d["rel_vol"] = d["volume"] / m.replace(0, np.nan)

    # Target: signal at close of bar t -> enter open of t+1 -> exit close of t+1
    d["fwd_ret"] = g["close"].shift(-1) / g["open"].shift(-1) - 1
    d["y"] = (d["fwd_ret"] > 0).astype(int)

    # Gap safety
    ts = d.index.to_series()
    next_ts = ts.groupby(d["day"]).shift(-1)
    next_ok = (next_ts - ts) == STEP
    gap = (ts.groupby(d["day"]).diff() > STEP).astype(int)
    recent_gap = gap.groupby(d["day"]).transform(lambda s: s.rolling(12, min_periods=1).max())

    feats = feature_columns(d)
    d = d.replace([np.inf, -np.inf], np.nan)
    mask = ((d["bar_no"] >= WARMUP) & d["fwd_ret"].notna() & next_ok
            & (recent_gap == 0) & d[feats].notna().all(axis=1))
    return d[mask]
