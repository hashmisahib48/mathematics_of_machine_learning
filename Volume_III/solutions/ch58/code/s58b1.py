# Problem 58B.1: thresholds for a recall target (cross-validated)
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import (cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)        # 0 = malignant
Xa, Xb, ya, yb = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression())
p = cross_val_predict(model, Xa, ya, cv=5,
                      method="predict_proba")[:, 0]   # P(malignant)
pos = ya == 0
# drop_intermediate=False keeps every threshold of the staircase
fpr, tpr, thr = roc_curve(pos, p, drop_intermediate=False)
print("malignant", pos.sum(), " benign", (~pos).sum())
print("target  threshold  recall  misses  false alarms  precision")
for target in (0.95, 0.98, 0.99):
    k = np.flatnonzero(tpr >= target)[0]   # first point reaching it
    yes = p >= thr[k]
    miss, alarm = (~yes & pos).sum(), (yes & ~pos).sum()
    prec = (yes & pos).sum() / yes.sum()
    print(f"{target:6.2f} {thr[k]:10.4f} {tpr[k]:7.4f} {miss:7d}"
          f" {alarm:13d} {prec:10.4f}")
k = np.flatnonzero(tpr >= 0.99)[0]         # one step up the staircase
print("0.99: threshold", round(thr[k], 4), " next one up",
      round(thr[k - 1], 4), " recall there", round(tpr[k - 1], 4))
yes = p >= 0.5
print("cut 0.5: recall", round((yes & pos).sum() / pos.sum(), 4),
      " false alarms", (yes & ~pos).sum())
