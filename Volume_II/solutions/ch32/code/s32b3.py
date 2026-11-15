# Problem 32B.3: which predictions does calibration change?
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
plain = make_pipeline(StandardScaler(), SVC()).fit(Xtr, ytr)
base = plain.predict(Xte)
f = plain.decision_function(Xte)        # SVC score of each patient
print(f"uncalibrated SVC: accuracy {plain.score(Xte, yte):.4f}")
for meth in ("sigmoid", "isotonic", "temperature"):
    cal = CalibratedClassifierCV(SVC(), method=meth, cv=5,
                                 ensemble=False)
    cal = make_pipeline(StandardScaler(), cal).fit(Xtr, ytr)
    pred = cal.predict(Xte)
    idx = np.flatnonzero(pred != base)
    print(f"{meth:12s} changed {len(idx)} of {len(yte)};"
          f" accuracy {cal.score(Xte, yte):.4f}")
    for i in idx:
        print(f"   patient {i:3d}: score f = {f[i]:+.3f}, true class "
              f"{yte[i]}, plain {base[i]}, calibrated {pred[i]}")
