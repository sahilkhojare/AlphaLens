import pandas as pd

TZ = "Asia/Kolkata"

def load_ohlcv(path, tz=TZ):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    tcol = next((c for c in ("datetime", "date", "timestamp", "time") if c in df.columns), df.columns[0])
    ts = pd.to_datetime(df[tcol], errors="coerce", utc=True).dt.tz_convert(tz) \
        if df[tcol].astype(str).str.contains(r"[+-]\d\d:\d\d$").any() \
        else pd.to_datetime(df[tcol], errors="coerce").dt.tz_localize(tz)
    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df.columns)}")
    cols = need + (["volume"] if "volume" in df.columns else [])
    out = df[cols].apply(pd.to_numeric, errors="coerce")
    out.index = pd.DatetimeIndex(ts, name="ts")
    return out[out.index.notna()].sort_index()

def clean_and_report(df, name="data"):
    rep = {"raw_rows": len(df)}
    rep["duplicate_timestamps"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")]
    rep["nan_rows"] = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    bad = ((df.high < df[["open", "close"]].max(axis=1))
           | (df.low > df[["open", "close"]].min(axis=1))
           | (df[["open", "high", "low", "close"]] <= 0).any(axis=1))
    rep["bad_ohlc_rows"] = int(bad.sum())
    df = df[~bad]
    n_before = len(df)
    df = df.iloc[df.index.indexer_between_time("09:15", "15:25")]
    rep["outside_session_rows"] = n_before - len(df)
    ts = df.index.to_series()
    gap = (ts.groupby(ts.dt.date).diff() > pd.Timedelta("5min"))
    rep["gaps_inside_days"] = int(gap.sum())
    rep["clean_rows"] = len(df)
    rep["sessions"] = int(ts.dt.date.nunique())
    rep["date_range"] = f"{df.index.min()} -> {df.index.max()}"
    rep["has_volume"] = bool("volume" in df.columns and df["volume"].sum() > 0)
    print(f"\n=== Data report: {name} ===")
    for k, v in rep.items():
        print(f"{k:24s} {v}")
    return df, rep
