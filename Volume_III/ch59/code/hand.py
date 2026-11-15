# Numbers of the worked examples of Chapter 59 (not printed in the book).
import json
import numpy as np
from scipy.stats import binomtest, loguniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (RandomizedSearchCV, cross_val_score,
                                     train_test_split, KFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT = {}
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
svc = make_pipeline(StandardScaler(), SVC(C=128.0, gamma=2.0 ** -11))
model = CalibratedClassifierCV(svc, method="sigmoid", cv=5).fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 0]
mal = yte == 0

# Example 59.4-1: thresholds and costs on the test patients
cFN, cFP = 10, 1
t_star = cFP / (cFP + cFN)
tab = []
for name, cut in (("0.5", 0.5), ("0.176", 0.17572836843167813),
                  ("t*", t_star)):
    tn, fp, fn, tp = confusion_matrix(mal, p >= cut).ravel()
    tab.append([name, round(cut, 4), int(fn), int(fp),
                int(cFN * fn + cFP * fp)])
    print(name, round(cut, 4), "FN", fn, "FP", fp, "cost", cFN * fn + cFP * fp)
OUT["thresholds"] = tab
OUT["t_star"] = t_star
OUT["implied_ratio_0.176"] = (1 - 0.17572836843167813) / 0.17572836843167813
print("t*", t_star, "implied cost ratio at 0.176:", OUT["implied_ratio_0.176"])
ps = [0.02, 0.06, 0.09, 0.12, 0.30, 0.85]
rows = [[q, round(cFN * q, 2), round(cFP * (1 - q), 2),
         "flag" if cFN * q > cFP * (1 - q) else "pass"] for q in ps]
OUT["six"] = rows
print(rows)

# Example 59.7-1: how precise is one test accuracy?
errs = int((model.predict(Xte) != yte).sum()); n = len(yte)
acc = 1 - errs / n
se = np.sqrt(acc * (1 - acc) / n)
wil = binomtest(n - errs, n).proportion_ci(confidence_level=0.95, method="wilson")
rng = np.random.RandomState(0)
correct = (model.predict(Xte) == yte).astype(float)
boot = np.array([correct[rng.randint(0, n, n)].mean() for _ in range(5000)])
OUT["acc"] = dict(errors=errs, n=n, acc=acc, se=se, lo=acc - 1.96 * se,
                  hi=acc + 1.96 * se, wilson=[wil.low, wil.high],
                  boot=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
print(OUT["acc"])
for k, v in {"1 patient": 1 / n}.items():
    print(k, v)

# Example 59.7-2: the bill of a nested search, counted by a fit counter
COUNT = [0]
class CountingRidge(RegressorMixin, BaseEstimator):
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, X, y):
        COUNT[0] += 1; self.w_ = np.linalg.solve(X.T @ X + self.alpha * np.eye(X.shape[1]), X.T @ y); return self
    def predict(self, X): return X @ self.w_
Xr, yr = make_regression(200, 5, noise=1.0, random_state=0)
inner = RandomizedSearchCV(CountingRidge(), {"alpha": loguniform(1e-3, 1e3)},
                           n_iter=30, cv=5, random_state=0)
inner.fit(Xr, yr); single = COUNT[0]
COUNT[0] = 0
cross_val_score(inner, Xr, yr, cv=KFold(5))
nested = COUNT[0]
OUT["fits"] = dict(single=single, nested=nested, families3=3 * nested)
print(OUT["fits"])
json.dump(OUT, open("verify/hand.json", "w"), indent=1, default=float)
