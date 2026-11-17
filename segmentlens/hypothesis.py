"""Hypothesis tests with effect sizes and a Bonferroni correction."""
import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table)[0]
    n = table.values.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def run_tests(seg: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    df = seg.join(label)
    rows = []
    # H1: average order value differs between highest- and lowest-spend segments (Welch t-test)
    order = df.groupby("segment")["monetary"].mean().sort_values()
    lo, hi = order.index[0], order.index[-1]
    a, b = df[df.segment == hi]["monetary"], df[df.segment == lo]["monetary"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    rows.append(("H1 avg order value: top vs bottom segment (Welch t)", t, p, cohens_d(a, b), "Cohen d"))
    # H2: repeat-purchase rate depends on segment (chi-square)
    tab = pd.crosstab(df["segment"], df["repeat_purchase"])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    rows.append(("H2 repeat purchase depends on segment (chi-square)", chi2, p, cramers_v(tab), "Cramer V"))
    # H3: repeat-purchase rate depends on acquisition channel (chi-square)
    tab = pd.crosstab(df["channel"], df["repeat_purchase"])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    rows.append(("H3 repeat purchase depends on channel (chi-square)", chi2, p, cramers_v(tab), "Cramer V"))
    # H4: discount share differs by region (Kruskal-Wallis, non-parametric)
    groups = [g["discount_share"].values for _, g in df.groupby("region")]
    h, p = stats.kruskal(*groups)
    rows.append(("H4 discount share differs by region (Kruskal-Wallis)", h, p, np.nan, "n/a"))
    out = pd.DataFrame(rows, columns=["hypothesis", "statistic", "p_value", "effect_size", "effect_type"])
    out["p_bonferroni"] = (out["p_value"] * len(out)).clip(upper=1.0)
    out["significant_at_0.05"] = out["p_bonferroni"] < 0.05
    return out
