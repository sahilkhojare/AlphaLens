from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_ohlcv, clean_and_report
from src.features import build_dataset, feature_columns
from src.train import split_by_date, build_models
from src.evaluate import metrics
from src.backtest import (signals_from_proba, apply_hold, run_backtest, summarize,
                          buy_and_hold_summary, round_trip_cost, CAPITAL)

DELTAS = [0.02, 0.04, 0.06, 0.08, 0.10]
MIN_TRADES = 30

st.set_page_config(page_title="AlphaLens", layout="wide")


@st.cache_data(show_spinner=False)
def load_clean(path):
    return clean_and_report(load_ohlcv(path), name=Path(path).stem)


@st.cache_data(show_spinner="Training models...")
def train_all(paths, hold):
    parts = []
    for f in paths:
        df, _ = load_clean(f)
        ds = build_dataset(df, horizon=hold)
        ds["symbol"] = Path(f).stem
        parts.append(ds)
    feats = sorted(set.intersection(*[set(feature_columns(p)) for p in parts]),
                   key=lambda c: feature_columns(parts[0]).index(c))
    data = pd.concat(parts).sort_index()
    train, val, test = split_by_date(data)
    out = {}
    for name, model in build_models().items():
        model.fit(train[feats], train["y"])
        out[name] = {
            "p_val": model.predict_proba(val[feats])[:, 1],
            "p_test": model.predict_proba(test[feats])[:, 1],
            "m_val": metrics(val["y"], model.predict(val[feats])),
            "m_test": metrics(test["y"], model.predict(test[feats])),
        }
    return {"train": train, "val": val, "test": test,
            "base": float(train["y"].mean()), "models": out}


def choose_margin(val, p, base, hold, cost):
    """Pick the probability margin on the validation period only."""
    part = val.assign(p=p)
    best = None
    for d in DELTAS:
        nets = []
        for _, g in part.groupby("symbol"):
            g = g.sort_index()
            sig = apply_hold(g.index, signals_from_proba(g["p"].values, base, d), hold)
            tr = sig != 0
            nets.append((sig * g["fwd_ret"].values - cost)[tr])
        net = np.concatenate(nets)
        if len(net) < MIN_TRADES:
            continue
        if best is None or net.mean() > best[1]:
            best = (d, net.mean(), len(net))
    return best


def fmt(x, suffix=""):
    return "n/a" if pd.isna(x) else f"{x:.2f}{suffix}"


# ---------------------------------------------------------------- header
st.title("AlphaLens")
st.caption("Read the market. Test the signal.")
st.warning("Educational research prototype. Historical simulations only, not investment advice.")

files = sorted(str(p) for p in Path("data").glob("*_5m.csv"))
if not files:
    st.error("No data found. Run `python src/probe_data.py` first to download 5-minute data into data/.")
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    default = [f for f in files if ".NS_" in f] or files[:1]
    chosen = st.multiselect("Datasets (5-minute candles)", files, default=default,
                            format_func=lambda f: Path(f).stem)
    hold = st.select_slider("Holding period", options=[1, 3, 6], value=3,
                            format_func=lambda h: f"{h} bar(s) = {h * 5} min")
    model_name = st.selectbox("Model", ["LogReg", "RandomForest"])
    st.subheader("Cost assumptions")
    fee = st.number_input("Fees, round trip (bps)", 0.0, 50.0, 6.0, 0.5)
    slip = st.number_input("Slippage per side (bps)", 0.0, 50.0, 2.0, 0.5)
    st.caption(f"Total per trade: {round_trip_cost(fee, slip) * 1e4:.1f} bps")
    period = st.radio("Evaluate on", ["Validation", "Test"])
    manual = st.select_slider("Probability margin", options=["auto"] + DELTAS, value="auto")

if not chosen:
    st.info("Pick at least one dataset in the sidebar.")
    st.stop()

res = train_all(tuple(chosen), hold)
key = "p_val" if period == "Validation" else "p_test"
mkey = "m_val" if period == "Validation" else "m_test"
target = res["val"] if period == "Validation" else res["test"]
cost = round_trip_cost(fee, slip)

if period == "Validation":
    st.info("Validation period: the margin is tuned on these same sessions, so results are optimistic.")
else:
    st.warning("Test period: this data was already used for the reported results. "
               "Exploring settings here is fine, but do not report the best-looking setting.")

# margins (chosen on validation only)
deltas = {}
for name, r in res["models"].items():
    if manual != "auto":
        deltas[name] = manual
    else:
        best = choose_margin(res["val"], r["p_val"], res["base"], hold, cost)
        deltas[name] = best[0] if best else None
active = [n for n in res["models"] if deltas[n] is not None]
ok = model_name in active

tgt = target.assign(**{f"p_{n}": r[key] for n, r in res["models"].items()})
symbols = sorted(tgt["symbol"].unique())
sym = st.sidebar.selectbox("Stock to inspect", symbols)
g = tgt[tgt["symbol"] == sym].sort_index()

bts, rows = {}, {}
if ok:
    for n in active:
        sig = signals_from_proba(g[f"p_{n}"].values, res["base"], deltas[n])
        bts[n] = run_backtest(g, sig, fee, slip, hold=hold)
        rows[n] = summarize(bts[n])
    rule = np.where(g["ema9_ratio"] < g["ema20_ratio"], 1, -1)
    bts["EMA rule"] = run_backtest(g, rule, fee, slip, hold=hold)
    rows["EMA rule"] = summarize(bts["EMA rule"])
    rows["Buy & hold (no costs)"] = buy_and_hold_summary(g)

tab_data, tab_models, tab_bt, tab_trades, tab_notes = st.tabs(
    ["Data status", "Model comparison", "Trading results", "Trades & export", "Limitations"])

# ---------------------------------------------------------------- data
with tab_data:
    st.subheader("Data status")
    st.write("Source: Yahoo Finance via yfinance, 5-minute candles (personal-use data, not redistributed).")
    reps = {Path(f).stem: load_clean(f)[1] for f in chosen}
    st.dataframe(pd.DataFrame(reps).T.astype(str), use_container_width=True)
    split_rows = {}
    for label in ("train", "val", "test"):
        part = res[label]
        split_rows[label] = {"from": str(part["day"].min()), "to": str(part["day"].max()),
                             "rows": len(part), "share_of_up_bars": round(part["y"].mean(), 3)}
    st.subheader("Chronological split")
    st.dataframe(pd.DataFrame(split_rows).T, use_container_width=True)

# ---------------------------------------------------------------- models
with tab_models:
    st.subheader(f"Classification metrics ({period.lower()} period)")
    majority = int(res["base"] >= 0.5)
    base_acc = float((target["y"] == majority).mean())
    mrows = {"Always predict majority class": {"accuracy": base_acc}}
    for n, r in res["models"].items():
        m = r[mkey]
        mrows[n] = {k: m[k] for k in ("accuracy", "precision", "recall", "f1")}
    st.dataframe(pd.DataFrame(mrows).T.round(3), use_container_width=True)
    st.caption("Accuracy is not profitability. A model that always predicts the majority class "
               "already scores the baseline accuracy.")
    cols = st.columns(2)
    for col, (n, r) in zip(cols, res["models"].items()):
        fig = px.imshow(np.array(r[mkey]["confusion"]), text_auto=True,
                        x=["Predicted not up", "Predicted up"], y=["Actually not up", "Actually up"],
                        color_continuous_scale="Blues", title=f"{n}: confusion matrix")
        fig.update_layout(coloraxis_showscale=False)
        col.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- trading
with tab_bt:
    if not ok:
        st.warning(f"{model_name} produced fewer than {MIN_TRADES} validation trades at every margin, "
                   "so it is skipped. Try another holding period or a manual margin.")
    else:
        st.subheader(f"{sym}: {model_name}, hold {hold * 5} min, margin {deltas[model_name]}")
        s = rows[model_name]
        c = st.columns(5)
        c[0].metric("Total return", fmt(s["total_return_pct"], "%"))
        c[1].metric("Max drawdown", fmt(s["max_drawdown_pct"], "%"))
        c[2].metric("Sharpe (daily)", fmt(s["sharpe"]))
        c[3].metric("Trades", int(s["trades"]))
        c[4].metric("Hit rate", fmt(s["hit_rate_pct"], "%"))

        fig = go.Figure()
        for n in active + ["EMA rule"]:
            fig.add_scatter(x=bts[n].index.tz_localize(None), y=bts[n]["equity"], name=n,
                            line=dict(width=3 if n == model_name else 1.5))
        fig.add_scatter(x=g.index.tz_localize(None), y=CAPITAL * g["close"] / g["close"].iloc[0],
                        name="Buy & hold (no costs)", line=dict(dash="dot"))
        fig.update_layout(title="Equity curve", yaxis_title="Account value", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        eq = bts[model_name]["equity"]
        peak = np.maximum.accumulate(np.r_[CAPITAL, eq.values])[1:]
        dd = go.Figure(go.Scatter(x=eq.index.tz_localize(None), y=(eq.values / peak - 1) * 100,
                                  fill="tozeroy", name="Drawdown"))
        dd.update_layout(title=f"{model_name}: drawdown", yaxis_title="% below previous peak")
        st.plotly_chart(dd, use_container_width=True)

        st.dataframe(pd.DataFrame(rows).T.round(2), use_container_width=True)
        st.caption("Sharpe uses daily returns x sqrt(252). With only a few sessions it is not statistically "
                   "meaningful. Buy & hold holds overnight and pays no costs, so it is not a like-for-like comparison.")

# ---------------------------------------------------------------- trades
with tab_trades:
    if not ok:
        st.warning("No trades to show for this model and setting.")
    else:
        bt = bts[model_name]
        trades = bt[bt["signal"] != 0].copy()
        trades["side"] = np.where(trades["signal"] > 0, "LONG", "SHORT")
        trades["entry"] = "open of next bar"
        trades["exit"] = f"close of bar t+{hold}"
        trades = trades.rename_axis("signal_time")
        trades = trades[["side", "entry", "exit", "gross_ret", "cost", "net_ret"]]
        st.subheader(f"Trades: {sym}, {model_name} ({len(trades)})")
        st.dataframe(trades.round(5), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.download_button("Download trades (CSV)", trades.to_csv().encode(),
                           file_name=f"trades_{model_name}_{sym}_hold{hold}.csv", mime="text/csv")
        c2.download_button("Download full backtest (CSV)", bt.to_csv().encode(),
                           file_name=f"backtest_{model_name}_{sym}_hold{hold}.csv", mime="text/csv")

# ---------------------------------------------------------------- notes
with tab_notes:
    st.markdown("""
**What is predicted:** at the close of each 5-minute candle, whether a trade entered at the next candle's
open and exited at the close of the candle `hold` bars later would be profitable. No overnight positions.

**Limitations**
- Short sample (about 3 months, two stocks, a handful of test sessions), so results are noisy.
- Costs are assumptions, not exact broker rates. Taxes, spread, market impact and liquidity are not modelled in detail.
- The probability margin and holding periods were explored on validation data.
- Past simulated results say nothing about future performance.
- Feature importance, if shown, would not be causal evidence.
""")