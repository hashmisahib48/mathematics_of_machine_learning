# Toy B: eight scores, isotonic and Platt calibration.
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import isotonic_regression
from sklearn.linear_model import LogisticRegression

f = np.array([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
lab = np.array([0, 0, 1, 0, 1, 0, 1, 1])
print("isotonic values:", isotonic_regression(lab.astype(float)))

n1, n0 = lab.sum(), (1 - lab).sum()     # Platt targets, Eq. 32.4-5
t = np.where(lab == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
print("targets:", t.round(4))
sg = lambda z: 1 / (1 + np.exp(-z))
s = 0.0                                  # B = 0 by symmetry, A = -s
for k in range(5):
    p = sg(s * f)
    g = np.sum(f * (p - t))              # dJ/ds
    h = np.sum(f ** 2 * p * (1 - p))     # d2J/ds2
    print(f"step {k}: s = {s:.5f}  g = {g:+.5f}  h = {h:.4f}")
    s -= g / h
print(f"Newton: A = {-s:.5f}, B = 0")

base = LogisticRegression().fit(f[:, None], lab)
base.coef_, base.intercept_ = np.array([[1.0]]), np.array([0.0])
for meth in ("sigmoid", "isotonic", "temperature"):
    cal = CalibratedClassifierCV(FrozenEstimator(base), method=meth,
                                 cv=2).fit(f[:, None], lab)
    print(meth, cal.predict_proba(f[:, None])[:, 1].round(4))
    g_ = cal.calibrated_classifiers_[0].calibrators[0]
    if meth == "sigmoid":
        print("  sklearn A, B:", round(g_.a_, 5),
              round(g_.b_, 8) + 0.0)
    if meth == "temperature":
        print("  beta:", round(g_.beta_, 5),
              " 2 beta:", round(2 * g_.beta_, 5))
