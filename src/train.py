import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def split_by_date(data, train=0.6, val=0.2):
    days = pd.Index(sorted(set(data["day"])))
    i, j = int(len(days) * train), int(len(days) * (train + val))
    parts = [days[:i], days[i:j], days[j:]]
    return [data[data["day"].isin(p)] for p in parts]

def build_models():
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=50,
            random_state=42, n_jobs=-1),
    }