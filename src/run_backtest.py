import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.data_loader import load_ohlcv, clean_and_report
from src.features import build_dataset, feature_columns
from src.train import split_by_date, build_models
from src.backtest import (signals_from_proba, run_backtest, summarize,
                          buy_and_hold_summary, round_trip_cost)

DELTAS = [0.02, 0.04, 0.06, 0.08, 0.10]
MIN_TRADES = 30
FEE_BPS, SLIP_BPS = 6.0, 2.0

def pick_delta(part, p, base):
    """Choose the margin on the validation period only."""
    cost = round_trip_cost(FEE_BPS, SLIP_BPS)
    best = None
    for d in DELTAS:
        sig = signals_from_proba(p, base, d)
        tr = sig != 0
        if tr.sum() < MIN_TRADES:
            continue
        net = (sig * part["fwd_ret"].values - cost)[tr]
        if best is None or net.mean() > best[1]:
            best = (d, net.mean(), int(tr.sum()))
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--final", action="store_true",
                    help="evaluate on the untouched test period (use once, at the end)")
    args = ap.parse_args()

    parts = []
    for f in args.files:
        df, _ = clean_and_report(load_ohlcv(f), name=Path(f).stem)
        ds = build_dataset(df)
        ds["symbol"] = Path(f).stem
        parts.append(ds)
    feats = sorted(set.intersection(*[set(feature_columns(p)) for p in parts]),
                   key=lambda c: feature_columns(parts[0]).index(c))
    data = pd.concat(parts).sort_index()
    train, val, test = split_by_date(data)
    assert train["day"].max() < val["day"].min() < test["day"].min()
    target = test if args.final else val
    label = "TEST (final)" if args.final else "VALIDATION (margin tuned here, so optimistic)"
    base = train["y"].mean()
    print(f"\nEvaluating on: {label}")
    print(f"Costs: {round_trip_cost(FEE_BPS, SLIP_BPS) * 1e4:.1f} bps per trade (fees {FEE_BPS} round trip + slippage {SLIP_BPS} per side)")

    Path("results").mkdir(exist_ok=True)
    rows = []
    for name, model in build_models().items():
        model.fit(train[feats], train["y"])
        best = pick_delta(val, model.predict_proba(val[feats])[:, 1], base)
        if best is None:
            print(f"{name}: no margin gives at least {MIN_TRADES} validation trades, skipped")
            continue
        delta = best[0]
        print(f"{name}: margin chosen on validation = {delta} ({best[2]} trades)")
        p = model.predict_proba(target[feats])[:, 1]
        for sym, g in target.assign(p=p).groupby("symbol"):
            g = g.sort_index()
            bt = run_backtest(g, signals_from_proba(g["p"].values, base, delta), FEE_BPS, SLIP_BPS)
            rows.append({"symbol": sym, "strategy": name, **summarize(bt)})
            bt.to_csv(f"results/trades_{name}_{sym}.csv")

    for sym, g in target.groupby("symbol"):
        g = g.sort_index()
        rule = np.where(g["ema9_ratio"] < g["ema20_ratio"], 1, -1)   # EMA9 above EMA20 -> long
        rows.append({"symbol": sym, "strategy": "EMA rule", **summarize(run_backtest(g, rule, FEE_BPS, SLIP_BPS))})
        rows.append({"symbol": sym, "strategy": "Buy & hold (no costs)", **buy_and_hold_summary(g)})

    table = pd.DataFrame(rows).set_index(["symbol", "strategy"]).sort_index()
    print("\n" + table.round(2).to_string())

if __name__ == "__main__":
    main()