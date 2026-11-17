"""Before/after feature-engineering check: same model, baseline vs engineered features."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import BASELINE, ENGINEERED


def compare(features: pd.DataFrame, label: pd.Series, seed: int = 42) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for name, cols in [("baseline (2 raw features)", BASELINE),
                       ("engineered (12 features)", BASELINE + ENGINEERED)]:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        auc = cross_val_score(pipe, features[cols], label, cv=cv, scoring="roc_auc")
        acc = cross_val_score(pipe, features[cols], label, cv=cv, scoring="accuracy")
        rows.append({"feature_set": name, "cv_roc_auc_mean": auc.mean(), "cv_roc_auc_std": auc.std(),
                     "cv_accuracy_mean": acc.mean()})
    return pd.DataFrame(rows).round(4)
