# Probabilities from an SVC in release 1.9.1.
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

plain = make_pipeline(StandardScaler(), SVC()).fit(X_tr, y_tr)
print("predict_proba on SVC():", hasattr(plain, "predict_proba"))

cal = make_pipeline(
    StandardScaler(),
    CalibratedClassifierCV(SVC(), method="sigmoid", cv=5,
                           ensemble=False),
).fit(X_tr, y_tr)
p = cal.predict_proba(X_te)[:, 1]
acc = plain.score(X_te, y_te), cal.score(X_te, y_te)
print("accuracy, SVC             :", round(acc[0], 4))
print("accuracy, calibrated SVC  :", round(acc[1], 4))
print("log loss, calibrated SVC  :", round(log_loss(y_te, p), 4))
sig = cal[-1].calibrated_classifiers_[0].calibrators[0]
print("Platt A, B (P = 1/(1+exp(A f + B))):",
      round(float(sig.a_), 3), round(float(sig.b_), 3))
for f in (-1, 0, 1):
    print(f"P(f = {f:2d}) = "
          f"{1 / (1 + np.exp(sig.a_ * f + sig.b_)):.3f}")
