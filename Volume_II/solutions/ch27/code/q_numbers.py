# Numbers quoted in the model answers to Questions 1 - 4.  Run: python3 code/q_numbers.py
import time
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, load_diabetes, make_friedman1
from sklearn.ensemble import GradientBoostingClassifier as GC, GradientBoostingRegressor as GR, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.utils import check_random_state

def inv_median(e): return float(np.percentile(e, 50, method="inverted_cdf"))   # lower median, as scikit-learn

print("== Q1  absolute error: leaf values")
x = np.arange(1, 7.); y = np.array([3, 5, 4, 10, 14, 12.]); med = np.median(y); r = y - med
print("lemonade: median", med, " residuals", r.tolist(), " signs", np.sign(r).tolist(), " leaf medians", np.median(r[:3]), np.median(r[3:]))
Xd, yd = load_diabetes(return_X_y=True); Xt, Xe, yt, ye = train_test_split(Xd, yd, test_size=0.25, random_state=0)
print("diabetes training targets: median %.1f, std %.1f, min %.0f, max %.0f" % (np.median(yt), yt.std(), yt.min(), yt.max()))
def boost_abs(X, y, leaf, n=100, lr=0.1, d=2):
    rng = check_random_state(0); F = np.full(len(y), np.median(y)); trees = []
    for _ in range(n):
        res = y - F; t = DTR(max_depth=d, random_state=rng).fit(X, np.sign(res)); lf = t.apply(X); g = t.tree_.value[:, 0, 0].copy()
        if leaf == "median":
            for l in np.unique(lf): g[l] = inv_median(res[lf == l])
        F += lr * g[lf]; trees.append((t, g))
    return lambda Z: np.median(y) + lr * sum(g[t.apply(Z)] for t, g in trees)
for leaf in ("sign", "median"):
    f = boost_abs(Xt, yt, leaf); pe, pt = f(Xe), f(Xt)
    r2 = 1 - ((ye - pe) ** 2).sum() / ((ye - ye.mean()) ** 2).sum()
    print(f"leaf values = {leaf:6s}: train MAE {np.abs(yt-pt).mean():6.2f}  test MAE {np.abs(ye-pe).mean():6.2f}  test R2 {r2:7.4f}   largest |F - median| {np.abs(pt-np.median(yt)).max():.1f}")
print("largest possible shift with leaf values of +-1, 100 trees, eta 0.1: ", 100 * 0.1 * 1)

print("\n== Q2  learning rate, Friedman data, depth 3, 1000 trees")
Xf, yf = make_friedman1(600, noise=1.0, random_state=0); Xft, Xfe, yft, yfe = train_test_split(Xf, yf, test_size=0.5, random_state=0)
print(" eta   best test MSE  trees  train MSE there  trees within 5% of best  fit time (s)")
for eta in (1.0, 0.3, 0.1, 0.03):
    t0 = time.time(); g = GR(n_estimators=1000, learning_rate=eta, max_depth=3, random_state=0).fit(Xft, yft); dt = time.time() - t0
    te = np.array([mean_squared_error(yfe, p) for p in g.staged_predict(Xfe)]); k = int(te.argmin())
    print(f"{eta:5.2f}  {te[k]:9.3f}    {k+1:5d}  {g.train_score_[k]:10.3f}      {int((te <= 1.05*te[k]).sum()):9d}            {dt:5.2f}")
print("noise variance of the targets = 1.0 (the test MSE cannot go below it)")

print("\n== Q3  bias and variance by simulation (40 training sets of 200, 500 fixed test points, Friedman function without noise)")
def f_true(X): return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
Xs, _ = make_friedman1(500, noise=0, random_state=99); ft = f_true(Xs)
mods = {"one stump": lambda s: DTR(max_depth=1, random_state=s), "one tree, depth 3": lambda s: DTR(max_depth=3, random_state=s),
        "one full tree": lambda s: DTR(random_state=s), "forest of 50 full trees": lambda s: RandomForestRegressor(50, random_state=s),
        "boosting, 100 stumps": lambda s: GR(n_estimators=100, max_depth=1, random_state=s), "boosting, 100 trees of depth 3": lambda s: GR(n_estimators=100, max_depth=3, random_state=s)}
P = {k: [] for k in mods}
for s in range(40):
    Xa, ya = make_friedman1(200, noise=1.0, random_state=s)
    for k, mk in mods.items(): P[k].append(mk(s).fit(Xa, ya).predict(Xs))
print(f"{'model':32s} bias^2  variance  sum")
for k, v in P.items():
    v = np.array(v); b2 = ((v.mean(0) - ft) ** 2).mean(); va = v.var(0).mean(); print(f"{k:32s} {b2:6.2f}  {va:7.2f}  {b2+va:6.2f}")

print("\n== Q4  probabilities of boosting, breast cancer (the split of Section 27.12)")
cd = load_breast_cancer(); Xc, Xce, yc, yce = train_test_split(cd.data, cd.target, test_size=0.25, random_state=0, stratify=cd.target)
lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xc, yc); gb = GC(random_state=0).fit(Xc, yc)
for name, m in (("logistic regression", lr), ("boosting, defaults", gb)):
    p = m.predict_proba(Xce)
    print(f"{name:30s} accuracy {m.score(Xce, yce):.4f}  log loss {log_loss(yce, p):.4f}")
p = gb.predict_proba(Xce)[:, 1]; wrong = (p > .5) != yce; li = -np.log(np.where(yce == 1, p, 1 - p))
print("boosting: wrong patients", int(wrong.sum()), "; their share of the total log loss %.2f; mean probability given to the true class by the wrong ones %.4f; by the right ones %.4f"
      % (li[wrong].sum() / li.sum(), np.where(yce == 1, p, 1 - p)[wrong].mean(), np.where(yce == 1, p, 1 - p)[~wrong].mean()))
print("probabilities of the right patients above 0.99 or below 0.01: %.2f" % (((p > .99) | (p < .01))[~wrong]).mean())
ll = [log_loss(yce, q) for q in gb.staged_predict_proba(Xce)]; print("test log loss at 1, 10, 31, 100 trees: ", [round(ll[k-1], 4) for k in (1, 10, 31, 100)], " minimum %.4f at %d trees" % (min(ll), int(np.argmin(ll)) + 1))
for m in ("sigmoid", "isotonic"):
    cal = CalibratedClassifierCV(GC(random_state=0), method=m, cv=5).fit(Xc, yc); q = cal.predict_proba(Xce)
    print(f"boosting calibrated ({m:8s}): accuracy {cal.score(Xce, yce):.4f}  log loss {log_loss(yce, q):.4f}")
