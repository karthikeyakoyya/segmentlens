"""Simulated e-commerce data. All data is synthetic and seeded; no real customers."""
import numpy as np
import pandas as pd

# Planted behaviour types (used only to generate data; the pipeline never sees them).
TYPES = {
    #            orders/yr, mean value, discount prob, return prob, share
    "loyal":      (14.0, 55.0, 0.15, 0.04, 0.20),
    "bargain":    (7.0,  28.0, 0.80, 0.08, 0.30),
    "big_spender":(3.0, 140.0, 0.20, 0.06, 0.15),
    "lapsing":    (1.5,  35.0, 0.40, 0.10, 0.35),
}
CATEGORIES = ["home", "fashion", "electronics", "grocery", "beauty"]
CHANNELS = ["web", "app", "marketplace"]
REGIONS = ["north", "south", "east", "west"]
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-12-31")


def generate(n_customers: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    names = list(TYPES)
    shares = np.array([TYPES[t][4] for t in names])
    kinds = rng.choice(names, size=n_customers, p=shares / shares.sum())
    total_days = (END - START).days
    customers, orders = [], []
    oid = 0
    for cid, kind in enumerate(kinds):
        per_year, mean_val, p_disc, p_ret, _ = TYPES[kind]
        signup = START + pd.Timedelta(days=int(rng.integers(0, int(total_days * 0.6))))
        active_days = (END - signup).days
        # lapsing customers stop buying part-way through
        if kind == "lapsing":
            active_days = int(active_days * rng.uniform(0.2, 0.7))
        n_orders = rng.poisson(per_year * active_days / 365)
        fav = rng.choice(CATEGORIES, size=2, replace=False)
        channel = rng.choice(CHANNELS, p=[0.5, 0.3, 0.2])
        customers.append((cid, signup, rng.choice(REGIONS), channel))
        if n_orders == 0:
            continue
        days = np.sort(rng.integers(0, max(active_days, 1), size=n_orders))
        for d in days:
            disc = rng.random() < p_disc
            val = rng.lognormal(np.log(mean_val), 0.4) * (0.85 if disc else 1.0)
            cat = fav[0] if rng.random() < 0.6 else rng.choice(CATEGORIES)
            orders.append((oid, cid, signup + pd.Timedelta(days=int(d)), round(val, 2),
                           cat, bool(disc), bool(rng.random() < p_ret)))
            oid += 1
    cust = pd.DataFrame(customers, columns=["customer_id", "signup_date", "region", "channel"])
    ords = pd.DataFrame(orders, columns=["order_id", "customer_id", "order_date",
                                         "order_value", "category", "discount_used", "returned"])
    truth = pd.Series(kinds, name="planted_type")
    return cust, ords, truth
