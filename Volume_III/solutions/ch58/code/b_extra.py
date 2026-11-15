# Extra numbers for Problems 58B.1 and 58B.2 quoted in the solutions
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)          # 0 = malignant
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression())
p = cross_val_predict(model, Xa, ya, cv=5, method="predict_proba")[:, 0]; pos = ya == 0
fpr, tpr, thr = roc_curve(pos, p, drop_intermediate=False)
print("== 58B.1  marginal price of recall on the training rows")
prev = None
for target in (0.95, 0.98, 0.99):
    k = np.flatnonzero(tpr >= target)[0]; yes = p >= thr[k]
    caught, alarms = (yes & pos).sum(), (yes & ~pos).sum()
    if prev: print("  %.2f -> %.2f: %d more tumours found for %d more false alarms: %.2f false alarms per extra tumour; equivalent t = 1/(1+%.2f) = %.4f"
                   % (prev[2], target, caught - prev[0], alarms - prev[1], (alarms - prev[1]) / (caught - prev[0]), (alarms - prev[1]) / (caught - prev[0]), 1 / (1 + (alarms - prev[1]) / (caught - prev[0]))))
    prev = (caught, alarms, target)
print("  from 0.5 (150 found, 2 alarms) to 0.99: %d more tumours for %d more false alarms" % (prev[0] - 150, prev[1] - 2))
print("\n== the same three thresholds on the 143 test rows (model fitted on all training rows)")
model.fit(Xa, ya); pt = model.predict_proba(Xb)[:, 0]; tp_ = yb == 0
for t in (0.4654, 0.247, 0.0817):
    yes = pt >= t
    print("  threshold %.4f: caught %d of %d, missed %d, false alarms %d" % (t, (yes & tp_).sum(), tp_.sum(), (~yes & tp_).sum(), (yes & ~tp_).sum()))
print("\n== the lowest malignant scores decide the 0.99 threshold (training, out-of-fold)")
low = np.sort(p[pos])[:4]; print("  four smallest P(malignant) among the malignant:", low.round(4).tolist())
print("  benign patients with a score at least 0.0817:", ((p >= 0.0817) & ~pos).sum(), " at least 0.0904:", ((p >= 0.0904) & ~pos).sum())
