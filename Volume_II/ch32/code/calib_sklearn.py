# CalibratedClassifierCV on the data of Section 32.4.
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.naive_bayes import GaussianNB

rng = np.random.default_rng(0)            # the same data as before
Z = rng.normal(size=(25000, 3))
X = np.hstack([Z] + [Z + 0.3 * rng.normal(size=Z.shape)
                     for _ in range(3)])
y = (rng.random(25000) < 1 / (1 + np.exp(-Z @ [1.6, 1.28, 0.96])))
y = y.astype(int)
X_tr, y_tr = X[:5000], y[:5000]            # fit the classifier
X_ca, y_ca = X[5000:15000], y[5000:15000]  # fit the calibrator
X_te, y_te = X[15000:], y[15000:]          # judge the result

def report(label, cal):
    p = cal.predict_proba(X_te)[:, 1]
    print(f"{label:34s} {log_loss(y_te, p):.4f}"
          f"  {brier_score_loss(y_te, p):.4f}")

nb = GaussianNB().fit(X_tr, y_tr)
report("GaussianNB, uncalibrated", nb)
for meth in ("sigmoid", "isotonic", "temperature"):
    cal = CalibratedClassifierCV(FrozenEstimator(nb), method=meth)
    report(f"frozen NB + {meth}", cal.fit(X_ca, y_ca))
for ens in (True, False):
    cal = CalibratedClassifierCV(GaussianNB(), cv=5, ensemble=ens)
    report(f"cv=5, ensemble={ens}", cal.fit(X_tr, y_tr))
    print("   calibrated_classifiers_:",
          len(cal.calibrated_classifiers_))

rf = RandomForestClassifier(random_state=0).fit(X_tr, y_tr)
report("forest, uncalibrated", rf)
for where, (Xc, yc) in (("training", (X_tr, y_tr)),
                        ("held-out", (X_ca, y_ca))):
    cal = CalibratedClassifierCV(FrozenEstimator(rf))   # sigmoid
    report(f"forest + sigmoid on {where} data", cal.fit(Xc, yc))
