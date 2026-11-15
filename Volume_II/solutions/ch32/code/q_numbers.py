# Numbers quoted in the model answers to the four questions of Chapter 32.
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression, isotonic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

print("== Q1  multipliers lambda_j of Eq. 32.3-3")
def lam(y, w=None):
    y = np.asarray(y, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    f = (w * y).sum() / w.sum(); return f, np.cumsum(w * (y - f))
for name, yb in (("dose block, doses 2-5", [5, 3, 4, 1]), ("start low, end high (4, 1, 6)", [4, 1, 6])):
    f, l = lam(yb); print(f"{name}: f = {f:.4f}  lambda_j = {l.round(4)}")
print("PAVA on (4, 1, 6):", isotonic_regression(np.array([4, 1, 6.])).round(4))
print("PAVA on (1, 5)   :", isotonic_regression(np.array([1, 5.])).round(4))
print("PAVA on (5, 3)   :", isotonic_regression(np.array([5, 3.])).round(4))
f, l = lam([4, 1, 6]); print("leading part (4, 1) mean:", 2.5, " block mean:", round(f, 4))

print("\n== Q2  a calibrated forecaster with AUC 0.5")
rng = np.random.default_rng(0)
Z = rng.normal(size=(25000, 3)); X = np.hstack([Z] + [Z + 0.3 * rng.normal(size=Z.shape) for _ in range(3)])
pt = 1 / (1 + np.exp(-Z @ [1.6, 1.28, 0.96])); y = (rng.random(25000) < pt).astype(int)
Xtr, ytr, Xte, yte = X[:5000], y[:5000], X[15000:], y[15000:]
rate = yte.mean(); const = np.full(len(yte), rate)
print(f"share of events in the test set: {rate:.4f};  rate*(1-rate) = {rate * (1 - rate):.4f}")
lr = LogisticRegression().fit(Xtr, ytr)
for name, p in (("constant forecast", const), ("logistic regression", lr.predict_proba(Xte)[:, 1]), ("true probability", pt[15000:])):
    print(f"{name:20s} Brier {brier_score_loss(yte, p):.4f}  log loss {log_loss(yte, p):.4f}  AUC {roc_auc_score(yte, p):.4f}")
print("ln 2 =", round(np.log(2), 4))
cal = IsotonicRegression(out_of_bounds="clip").fit(const, yte)
print("isotonic calibration of the constant forecast gives:", np.unique(cal.predict(const)).round(4))

print("\n== Q3  Platt shift against temperature on an imbalanced problem")
rng = np.random.default_rng(2); n = 10000
Z = rng.normal(size=(n, 3)); X = np.hstack([Z] + [Z + 0.3 * rng.normal(size=Z.shape) for _ in range(3)])
pt = 1 / (1 + np.exp(-(Z @ [1.6, 1.28, 0.96] - 3.5))); y = (rng.random(n) < pt).astype(int)
Xa, ya, Xb, yb, ptb = X[:3000], y[:3000], X[3000:], y[3000:], pt[3000:]
print(f"share of events {y.mean():.4f};  test cases with true P > 1/2: {(ptb > .5).sum()} of {len(yb)}")
def line(name, pred): print(f"{name:24s} predicted positives {pred.sum():4d}, recall {(pred[yb == 1]).mean():.3f}, accuracy {(pred == yb).mean():.4f}")
line("true probability > 1/2", (ptb > .5).astype(int))
line("LinearSVC, f > 0", LinearSVC(max_iter=20000).fit(Xa, ya).predict(Xb))
for meth in ("sigmoid", "temperature"):
    m = CalibratedClassifierCV(LinearSVC(max_iter=20000), method=meth, cv=5).fit(Xa, ya)
    line(f"calibrated, {meth}", m.predict(Xb))
    if meth == "sigmoid":
        c = m.calibrated_classifiers_[0].calibrators[0]; A, B = c.a_, c.b_
        print(f"   first fold: A = {A:.3f}, B = {B:.3f};  P = 1/2 where f = -B/A = {-B / A:.3f}")

print("\n== Q4  gradient boosting, calibration sets of 300 and 30000 cases")
rng = np.random.default_rng(1); n = 5000 + 30000 + 10000
Z = rng.normal(size=(n, 3)); X = np.hstack([Z] + [Z + 0.3 * rng.normal(size=Z.shape) for _ in range(3)])
y = (rng.random(n) < 1 / (1 + np.exp(-Z @ [1.6, 1.28, 0.96]))).astype(int)
Xtr, ytr, Xca, yca, Xte, yte = X[:5000], y[:5000], X[5000:35000], y[5000:35000], X[35000:], y[35000:]
gb = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
print(f"raw gradient boosting: log loss {log_loss(yte, gb.predict_proba(Xte)[:, 1]):.4f}  Brier {brier_score_loss(yte, gb.predict_proba(Xte)[:, 1]):.4f}")
for size, reps in ((300, 20), (30000, 1)):
    res = {m: [] for m in ("sigmoid", "isotonic", "temperature")}; nzero = []
    for r in range(reps):
        idx = np.random.default_rng(r).choice(30000, size, replace=False)
        for m in res:
            c = CalibratedClassifierCV(FrozenEstimator(gb), method=m).fit(Xca[idx], yca[idx])
            p = c.predict_proba(Xte)[:, 1]; res[m].append((log_loss(yte, p), brier_score_loss(yte, p)))
            if m == "isotonic": nzero.append(int(((p == 0) | (p == 1)).sum()))
    print(f"calibration set {size} ({reps} draws): " + "; ".join(f"{m} {np.mean(v, 0)[0]:.4f}/{np.mean(v, 0)[1]:.4f}" for m, v in res.items()) + "   (log loss/Brier)")
    print(f"   exact 0 or 1 among the 10 000 isotonic forecasts: {np.mean(nzero):.0f}")
