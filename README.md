# SegmentLens

Customer segmentation, hypothesis testing and a before/after feature-engineering check, in Python.

**Data is simulated** (5,000 customers, about 39,800 orders, seeded, Jan 2023 - Dec 2024). Four behaviour types are planted in the generator, so the numbers below show that the pipeline works, not a finding about real customers.

## What it does
1. `features.py` builds 12 customer features using only orders before a cutoff (2024-10-01): recency, frequency, monetary value, average gap and gap spread between orders, discount share, return rate, category diversity, tenure, and recent-vs-prior order trend. Missing values are handled explicitly (customers with no orders get zeros, undefined recency/gaps use tenure).
2. `segment.py` runs K-Means on log-scaled RFM features for the 4,258 customers with at least one past order. k is picked by silhouette over k = 3..8 (best: k = 3, silhouette 0.505). 742 customers with no past orders are reported separately, not clustered.
3. `hypothesis.py` runs four tests with effect sizes and a Bonferroni correction:
   - H1 Welch t-test, average order value, top vs bottom segment: significant, Cohen d = 4.31
   - H2 chi-square, repeat purchase vs segment: significant, Cramer V = 0.69
   - H3 chi-square, repeat purchase vs channel: not significant (p = 0.86)
   - H4 Kruskal-Wallis, discount share vs region: not significant (p = 0.76)
4. `model.py` compares the same logistic regression (5-fold stratified CV) on 2 raw features vs 12 engineered features, predicting a purchase in the 90 days after the cutoff:
   - baseline: ROC AUC 0.899, accuracy 0.821
   - engineered: ROC AUC 0.910, accuracy 0.845

## Run
```
pip install -r requirements.txt
python run_all.py      # writes outputs/*.csv and outputs/segments.png
python -m pytest -q    # 8 tests
```

## Limits
- Simulated data; segment recovery is partly by design.
- The engineered-feature gain is modest (about +0.011 AUC) because total orders and spend already carry most of the signal.
- One cutoff date, one horizon; no time-based train/test split beyond the cutoff itself.
- Python only (no R version).
