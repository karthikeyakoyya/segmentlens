"""Feature engineering. Features use only orders strictly before the cutoff date."""
import numpy as np
import pandas as pd

BASELINE = ["total_orders", "total_spend"]
ENGINEERED = ["recency_days", "frequency", "monetary", "avg_gap_days", "gap_std_days",
              "discount_share", "return_rate", "category_diversity", "tenure_days",
              "recent_vs_prior_orders"]


def build_features(customers: pd.DataFrame, orders: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    past = orders[orders["order_date"] < cutoff].sort_values(["customer_id", "order_date"])
    g = past.groupby("customer_id")
    f = pd.DataFrame({
        "total_orders": g.size(),
        "total_spend": g["order_value"].sum(),
        "last_order": g["order_date"].max(),
        "discount_share": g["discount_used"].mean(),
        "return_rate": g["returned"].mean(),
        "category_diversity": g["category"].nunique(),
    })
    gaps = past.groupby("customer_id")["order_date"].diff().dt.days
    gap_df = pd.DataFrame({"customer_id": past["customer_id"], "gap": gaps})
    gg = gap_df.groupby("customer_id")["gap"]
    f["avg_gap_days"] = gg.mean()
    f["gap_std_days"] = gg.std()
    last90 = past[past["order_date"] >= cutoff - pd.Timedelta(days=90)].groupby("customer_id").size()
    prior90 = past[(past["order_date"] < cutoff - pd.Timedelta(days=90)) &
                   (past["order_date"] >= cutoff - pd.Timedelta(days=180))].groupby("customer_id").size()
    f["recent_vs_prior_orders"] = last90.reindex(f.index).fillna(0) - prior90.reindex(f.index).fillna(0)
    f["recency_days"] = (cutoff - f["last_order"]).dt.days
    f["frequency"] = f["total_orders"]
    f["monetary"] = f["total_spend"] / f["total_orders"]
    f = f.join(customers.set_index("customer_id")[["signup_date", "region", "channel"]], how="right")
    f["tenure_days"] = (cutoff - f["signup_date"]).dt.days
    # only customers who signed up before the cutoff
    f = f[f["tenure_days"] > 0].copy()
    # missing values: no past orders -> zeros; undefined recency/gaps -> tenure (documented choice)
    for c in ["total_orders", "total_spend", "frequency", "monetary", "discount_share",
              "return_rate", "category_diversity", "recent_vs_prior_orders"]:
        f[c] = f[c].fillna(0)
    f["recency_days"] = f["recency_days"].fillna(f["tenure_days"])
    f["avg_gap_days"] = f["avg_gap_days"].fillna(f["tenure_days"])
    f["gap_std_days"] = f["gap_std_days"].fillna(0)
    return f.drop(columns=["last_order"])


def build_label(orders: pd.DataFrame, index, cutoff: pd.Timestamp, horizon_days: int = 90) -> pd.Series:
    fut = orders[(orders["order_date"] >= cutoff) &
                 (orders["order_date"] < cutoff + pd.Timedelta(days=horizon_days))]
    buyers = set(fut["customer_id"])
    return pd.Series([int(c in buyers) for c in index], index=index, name="repeat_purchase")
