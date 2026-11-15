# Section 58.1: the default rule on the breast-cancer screen.
import numpy as np
from bcdata import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Xa, Xb, ya, yb, names = load()
print("malignant rows: train", (ya == 0).sum(), "of", len(ya),
      " test", (yb == 0).sum(), "of", len(yb))
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(Xa, ya)
pm = model.predict_proba(Xb)[:, 0]          # P(malignant)
pred = np.where(pm >= 0.5, 0, 1)
assert (pred == model.predict(Xb)).all()
(hit, miss), (alarm, ok) = confusion_matrix(yb, pred)
print("test: caught", hit, "missed", miss, "false alarms", alarm,
      "recall", round(hit / (hit + miss), 4),
      "precision", round(hit / (hit + alarm), 4))
cv = cross_val_predict(model, Xa, ya, cv=5, method="predict_proba")[:, 0]
yp = (ya == 0)
for t in (0.5, 0.2, 0.1, 0.05):
    yes = cv >= t
    print(f"CV threshold {t:4.2f}: recall {(yes & yp).sum() / yp.sum():.4f}"
          f"  false alarms {(yes & ~yp).sum()}")
fpr, tpr, thr = roc_curve(yp, cv)
k = np.flatnonzero(tpr >= 0.99)[0]
print("largest CV threshold with recall >= 0.99:", round(thr[k], 4),
      " recall", round(tpr[k], 4), " false alarm rate", round(fpr[k], 4))
print("CV scores below 0.05:", (cv < 0.05).sum(), " above 0.95:",
      (cv > 0.95).sum(), " between:", ((cv >= 0.05) & (cv <= 0.95)).sum())
