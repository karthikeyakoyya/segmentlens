import pandas as pd
import pytest
from segmentlens.data import generate
from segmentlens.features import build_features, build_label, BASELINE, ENGINEERED
from segmentlens.segment import prep, choose_k, fit_segments
from segmentlens.hypothesis import run_tests, cohens_d

CUTOFF = pd.Timestamp("2024-10-01")


@pytest.fixture(scope="module")
def data():
    cust, orders, _ = generate(n_customers=600, seed=7)
    feats = build_features(cust, orders, CUTOFF)
    label = build_label(orders, feats.index, CUTOFF)
    return cust, orders, feats, label


def test_generation_is_seeded():
    a = generate(200, seed=1)[1]
    b = generate(200, seed=1)[1]
    assert a.equals(b)


def test_no_future_leakage(data):
    _, orders, feats, _ = data
    past = orders[orders["order_date"] < CUTOFF].groupby("customer_id").size()
    got = feats.loc[feats["frequency"] > 0, "frequency"]
    assert (got == past.reindex(got.index)).all()


def test_no_missing_values_in_features(data):
    feats = data[2]
    assert not feats[BASELINE + ENGINEERED].isna().any().any()


def test_label_is_binary(data):
    assert set(data[3].unique()) <= {0, 1}


def test_choose_k_returns_valid_silhouette(data):
    feats = data[2]
    ks = choose_k(prep(feats[feats["frequency"] >= 1]), k_range=range(3, 5))
    assert ks["silhouette"].between(-1, 1).all()


def test_segments_cover_all_active_customers(data):
    feats = data[2]
    active = feats[feats["frequency"] >= 1]
    seg = fit_segments(active, 3)
    assert len(seg) == len(active) and seg["segment"].nunique() == 3


def test_hypothesis_table_shape_and_bonferroni(data):
    feats, label = data[2], data[3]
    seg = fit_segments(feats[feats["frequency"] >= 1], 3)
    t = run_tests(seg, label.loc[seg.index])
    assert len(t) == 4
    assert (t["p_bonferroni"] >= t["p_value"]).all()


def test_cohens_d_zero_for_identical_groups():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert cohens_d(s, s) == pytest.approx(0.0)
