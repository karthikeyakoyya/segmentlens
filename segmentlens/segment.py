"""K-Means customer segmentation with silhouette-based choice of k."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM = ["recency_days", "frequency", "monetary"]


def prep(features: pd.DataFrame):
    X = features[RFM].copy()
    X["frequency"] = np.log1p(X["frequency"])
    X["monetary"] = np.log1p(X["monetary"])
    return StandardScaler().fit_transform(X)


def choose_k(Xs, k_range=range(2, 9), seed=42):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        rows.append({"k": k, "silhouette": silhouette_score(Xs, km.labels_, sample_size=3000, random_state=seed),
                     "inertia": km.inertia_})
    return pd.DataFrame(rows)


def fit_segments(features: pd.DataFrame, k: int, seed: int = 42):
    Xs = prep(features)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
    out = features.copy()
    out["segment"] = km.labels_
    return out


def profile(seg: pd.DataFrame) -> pd.DataFrame:
    return seg.groupby("segment").agg(customers=("segment", "size"),
                                      recency_days=("recency_days", "mean"),
                                      frequency=("frequency", "mean"),
                                      monetary=("monetary", "mean"),
                                      discount_share=("discount_share", "mean")).round(2)
