# Problem 32C.3: judging on the calibration cases, and what to do with 2000 samples.
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB

rng = np.random.default_rng(0)                     # the data of Section 32.4
Z = rng.normal(size=(25000, 3)); X = np.hstack([Z] + [Z + 0.3 * rng.normal(size=Z.shape) for _ in range(3)])
y = (rng.random(25000) < 1 / (1 + np.exp(-Z @ [1.6, 1.28, 0.96]))).astype(int)
nb = GaussianNB().fit(X[:5000], y[:5000]); Xca, yca, Xte, yte = X[5000:15000], y[5000:15000], X[15000:], y[15000:]

print("== (a) the same calibration cases used for fitting and for judging (frozen naive Bayes)")
print("method        n_cal   on the calibration cases   on 10 000 fresh cases   optimism")
for meth in ("sigmoid", "isotonic", "temperature"):
    for n in (50, 200, 1000, 10000):
        a, b = [], []
        for r in range(10 if n < 10000 else 1):
            idx = np.random.default_rng(100 * n + r).choice(10000, n, replace=False)
            c = CalibratedClassifierCV(FrozenEstimator(nb), method=meth).fit(Xca[idx], yca[idx])
            a.append(log_loss(yca[idx], c.predict_proba(Xca[idx])[:, 1])); b.append(log_loss(yte, c.predict_proba(Xte)[:, 1]))
        print(f"{meth:12s} {n:6d} {np.mean(a):14.4f} {np.mean(b):22.4f} {np.mean(b) - np.mean(a):12.4f}")

print("\n== (b) only 2000 samples in all: cv=5 inside, judged by an outer 5-fold cross-validation")
X2, y2 = X[:2000], y[:2000]; outer = StratifiedKFold(5, shuffle=True, random_state=0)
def judge(est):
    p = cross_val_predict(est, X2, y2, cv=outer, method="predict_proba")[:, 1]; return log_loss(y2, p)
print(f"naive Bayes, raw                          {judge(GaussianNB()):.4f}")
for meth in ("sigmoid", "temperature", "isotonic"):
    for ens in (True, False):
        print(f"cv=5, {meth:12s} ensemble={str(ens):5s}      {judge(CalibratedClassifierCV(GaussianNB(), method=meth, cv=5, ensemble=ens)):.4f}")
c = CalibratedClassifierCV(GaussianNB(), cv=5, ensemble=False).fit(X2, y2)
print("fits of the classifier: ensemble=True 5, ensemble=False 6;  calibrated_classifiers_ lengths:",
      len(CalibratedClassifierCV(GaussianNB(), cv=5, ensemble=True).fit(X2, y2).calibrated_classifiers_), len(c.calibrated_classifiers_))
print("calibrator of ensemble=False trained on", len(y2), "out-of-fold scores; each of 5 calibrators of ensemble=True on", len(y2) // 5)
