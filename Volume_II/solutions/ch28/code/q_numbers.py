# Numbers quoted in the model answers to the four questions of Chapter 28
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.ensemble._hist_gradient_boosting.binning import _find_binning_thresholds
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

rng = np.random.default_rng(0)
print("== Q1  990 values in (0,1), ten in (1000, 2000)")
x = np.r_[rng.uniform(0.001, 1, 990), rng.uniform(1000, 2000, 10)]
def occupied(t, v): b = np.searchsorted(t, v); return len(np.unique(b)), np.bincount(b, minlength=len(t) + 1)
tq = _find_binning_thresholds(x, 255)
print("quantile thresholds:", len(tq), " bins", len(tq) + 1, " thresholds above 1:", int((tq > 1).sum()),
      " first 3:", tq[:3].round(4), " largest:", tq[-3:].round(1))
b = np.searchsorted(tq, x); print("  bins holding the 990 small values:", len(np.unique(b[:990])), " the ten large:", len(np.unique(b[990:])))
for name, v in (("x", x), ("log x", np.log(x)), ("log1p x", np.log1p(x))):
    e = np.linspace(v.min(), v.max(), 256)[1:-1]            # 255 equal-width bins
    bb = np.searchsorted(e, v)
    print(f"equal width on {name:8s}: width {(v.max()-v.min())/255:.4f}  bins used by small values {len(np.unique(bb[:990]))}"
          f"  by large {len(np.unique(bb[990:]))}  largest bin holds {np.bincount(bb).max()}")
tl = _find_binning_thresholds(np.log(x), 255)
print("quantile bins of log x identical to those of x:", np.array_equal(np.searchsorted(tl, np.log(x)), b))

print("\n== Q2  absolute error: sign gradients, median leaves")
X, y0 = make_friedman1(3000, n_features=6, noise=0.0, random_state=0)
r = np.random.default_rng(1)
y = y0 + 2.0 * r.standard_t(2, 3000)                        # heavy tails
Xt, yt0 = make_friedman1(3000, n_features=6, noise=0.0, random_state=5)
res = {}
for loss in ("squared_error", "absolute_error"):
    m = HGBR(loss=loss, random_state=0).fit(X[:2000], y[:2000])
    res[loss] = np.abs(m.predict(Xt) - yt0).mean()          # error against the noiseless truth
    print(f"{loss:15s} mean abs error against the true function: {res[loss]:.3f}")
for lam in (0.0, 10.0):
    m = HGBR(loss="absolute_error", max_iter=1, max_leaf_nodes=2, learning_rate=1.0, l2_regularization=lam, early_stopping=False).fit(X, y)
    nd = m._predictors[0][0].nodes
    print(f"absolute_error, l2 = {lam:4.1f}: leaf values {nd['value'][1]:.4f} {nd['value'][2]:.4f}")
for lam in (0.0, 10.0):
    m = HGBR(max_iter=1, max_leaf_nodes=2, learning_rate=1.0, l2_regularization=lam, early_stopping=False).fit(X, y)
    nd = m._predictors[0][0].nodes
    print(f"squared_error,  l2 = {lam:4.1f}: leaf values {nd['value'][1]:.4f} {nd['value'][2]:.4f}")
print("tiny example: residuals y - F in a node, split between the first three and the last three rows")
for resid in ([-1, -1, -1, 1, 1, 1], [-1, -1, -1, 100, 100, 100]):
    r_ = np.array(resid, float); g = -np.sign(r_); h = np.ones(6)
    GL, HL, GR, HR = g[:3].sum(), 3.0, g[3:].sum(), 3.0
    print(" residuals", resid, " gain", GL**2 / HL + GR**2 / HR - (g.sum())**2 / 6, " medians", np.median(r_[:3]), np.median(r_[3:]),
          " squared-error gain", r_[:3].sum()**2/3 + r_[3:].sum()**2/3 - r_.sum()**2/6)

print("\n== Q3  rows touched per level with and without the subtraction trick")
Xc, yc = make_friedman1(12000, n_features=8, random_state=0); yc = (yc > np.median(yc)).astype(int)
m = HGBC(max_iter=1, max_leaf_nodes=31, early_stopping=False).fit(Xc, yc)
nd = m._predictors[0][0].nodes
cnt = nd["count"]; split = nd["is_leaf"] == 0
small = large = 0
for i in np.where(split)[0]:
    a, b = cnt[nd["left"][i]], cnt[nd["right"][i]]; small += min(a, b); large += max(a, b)
print(f"31 leaves: {split.sum()} splits; rows in all children: {small + large}  in the smaller children: {small} ({small/(small+large):.3f})")
print(f"  rows passed with the trick: root {cnt[0]} + smaller children {small} = {cnt[0] + small};  without: root + all children = {cnt[0] + small + large}")
print(f"  saving on the children of one split of n rows: one row left: {(1-1/12000)*100:.2f}% of the larger child's pass; halving: 50% of the level")
for n_, ns in ((1000, 1), (1000, 500), (1000, 100)):
    print(f"  n = {n_}, smaller child {ns}: passes {ns} instead of {n_}: saves {1 - ns/n_:.3f} of the level")

print("\n== Q4  8 per cent missing, mostly class 1")
def make(n, seed):
    r = np.random.default_rng(seed); y = (r.random(n) < 0.4).astype(int)
    x = r.normal(1.2 * y, 1.0)                                  # moderate signal
    miss = r.random(n) < np.where(y == 1, 0.16, 0.027)
    return np.where(miss, np.nan, x).reshape(-1, 1), y, miss
Xa, ya, ma = make(20000, 0); Xb, yb, mb = make(20000, 1)
print(f"missing share {ma.mean():.3f}; share of class 1 among the missing {ya[ma].mean():.3f} (base rate {ya.mean():.3f})")
nanm = HGBC(max_iter=1, max_leaf_nodes=2, learning_rate=1.0, early_stopping=False).fit(Xa, ya)
pm = nanm.predict_proba(np.array([[np.nan]]))[0, 1]
print(f"NaN as it is: P(class 1 | NaN) = {pm:.3f};  accuracy {nanm.score(Xb, yb):.4f}")
imp = SimpleImputer(strategy="median").fit(Xa); med = imp.statistics_[0]
mi = HGBC(max_iter=1, max_leaf_nodes=2, learning_rate=1.0, early_stopping=False).fit(imp.transform(Xa), ya)
print(f"median imputation (median {med:.3f}): accuracy {mi.score(imp.transform(Xb), yb):.4f}; P(1 | x = median) = {mi.predict_proba([[med]])[0,1]:.3f}")
Xc_ = np.c_[imp.transform(Xa), np.isnan(Xa)]; Xd_ = np.c_[imp.transform(Xb), np.isnan(Xb)]
lr = LogisticRegression().fit(Xc_, ya); print(f"median + indicator column, logistic: accuracy {lr.score(Xd_, yb):.4f}")
cc = ~ma
ntr = HGBC(max_iter=1, max_leaf_nodes=2, learning_rate=1.0, early_stopping=False).fit(Xa[cc], ya[cc])
print(f"trained on complete rows only: P(1 | NaN) = {ntr.predict_proba(np.array([[np.nan]]))[0,1]:.3f};  accuracy on all test rows {ntr.score(Xb, yb):.4f}")
nd = ntr._predictors[0][0].nodes
print(f"  stored split: threshold {nd['num_threshold'][0]:.3f}, missing_go_to_left {nd['missing_go_to_left'][0]}, left count {nd['count'][1]}, right count {nd['count'][2]}")
print("  accuracy of always saying class 0:", 1 - yb.mean())
