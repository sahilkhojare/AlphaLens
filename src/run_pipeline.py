import argparse
from pathlib import Path
import pandas as pd
from src.data_loader import load_ohlcv, clean_and_report
from src.features import build_dataset, feature_columns
from src.train import split_by_date, build_models
from src.evaluate import metrics

def show(title, y, pred):
    m = metrics(y, pred)
    print(f"  {title:13s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f} f1={m['f1']:.3f}  [[TN,FP],[FN,TP]]={m['confusion']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--final", action="store_true", help="also score the untouched test set (use once, at the end)")
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
    for name, part in (("train", train), ("val", val), ("test", test)):
        print(f"{name:6s} {part['day'].min()} -> {part['day'].max()}  rows={len(part)}  up-share={part['y'].mean():.3f}")
    assert train["day"].max() < val["day"].min() < test["day"].min(), "split is not chronological!"

    majority = int(train["y"].mean() >= 0.5)
    base_acc = (val["y"] == majority).mean()
    print(f"\nFeatures: {feats}")
    print(f"Baseline (always predict {majority}, the train majority) accuracy on val: {base_acc:.3f}")
    for name, model in build_models().items():
        model.fit(train[feats], train["y"])
        print(f"\n{name}")
        show("validation", val["y"], model.predict(val[feats]))
        if args.final:
            show("TEST (final)", test["y"], model.predict(test[feats]))

if __name__ == "__main__":
    main()