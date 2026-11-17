"""Run the full pipeline and write outputs/ (CSVs, charts, summary)."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from segmentlens.data import generate
from segmentlens.features import build_features, build_label
from segmentlens.segment import prep, choose_k, fit_segments, profile
from segmentlens.hypothesis import run_tests
from segmentlens.model import compare

OUT = pathlib.Path("outputs"); OUT.mkdir(exist_ok=True)
CUTOFF = pd.Timestamp("2024-10-01")

cust, orders, truth = generate()
orders.to_csv(OUT / "orders_simulated.csv", index=False)
feats = build_features(cust, orders, CUTOFF)
label = build_label(orders, feats.index, CUTOFF)
print(f"customers={len(feats)} orders={len(orders)} repeat_rate={label.mean():.3f}")

active = feats[feats["frequency"] >= 1]          # segment only customers with at least one past order
print(f"active_buyers={len(active)} never_purchased={len(feats) - len(active)}")
ks = choose_k(prep(active), k_range=range(3, 9)); ks.to_csv(OUT / "k_selection.csv", index=False)
best_k = int(ks.loc[ks["silhouette"].idxmax(), "k"])
seg = fit_segments(active, best_k)
prof = profile(seg); prof.to_csv(OUT / "segment_profile.csv")
tests = run_tests(seg, label.loc[seg.index]); tests.round(5).to_csv(OUT / "hypothesis_tests.csv", index=False)
fe = compare(feats, label); fe.to_csv(OUT / "feature_engineering_comparison.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(ks["k"], ks["silhouette"], marker="o"); ax[0].set_title("Silhouette by k"); ax[0].set_xlabel("k")
prof["customers"].plot.bar(ax=ax[1]); ax[1].set_title("Customers per segment")
plt.tight_layout(); plt.savefig(OUT / "segments.png", dpi=120)

print("best_k", best_k); print(ks.round(3)); print(prof); print(tests.round(4).to_string()); print(fe)
