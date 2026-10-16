import yfinance as yf, pandas as pd

for sym in ["^NSEI", "RELIANCE.NS", "TCS.NS"]:
    df = yf.download(sym, interval="5m", period="60d", progress=False, auto_adjust=False)
    if df.empty:
        print(sym, "-> NO DATA"); continue
    df.columns = df.columns.get_level_values(0)
    df.index = df.index.tz_convert("Asia/Kolkata")
    days = df.index.normalize().nunique()
    print(f"\n{sym}: rows={len(df)}, sessions={days}")
    print("range:", df.index.min(), "->", df.index.max())
    print("bars/day:", df.groupby(df.index.date).size().value_counts().head(3).to_dict())
    print("first/last bar time:", df.index.time.min(), df.index.time.max())
    print("duplicates:", df.index.duplicated().sum(), "| NaNs:", int(df.isna().sum().sum()))
    print("zero-volume bars:", int((df["Volume"] == 0).sum()))
    print("bad OHLC:", int(((df.High < df[["Open","Close"]].max(axis=1)) | (df.Low > df[["Open","Close"]].min(axis=1))).sum()))
    df.to_csv(f"data/{sym.replace('^','')}_5m.csv")
