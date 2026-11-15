# Section 59.4 (continued): probabilities, threshold, test set.
import numpy as np
from sklearn.calibration import (CalibratedClassifierCV,
                                 calibration_curve)
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (brier_score_loss, confusion_matrix,
                             log_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import (cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
svc = make_pipeline(StandardScaler(),
                    SVC(C=128.0, gamma=2.0 ** -11))
model = CalibratedClassifierCV(svc, method="sigmoid", cv=5)

# out-of-fold P(malignant); malignant is class 0
oof = cross_val_predict(model, Xtr, ytr, cv=5,
                        method="predict_proba")[:, 0]
mal = ytr == 0
frac, mean_p = calibration_curve(mal, oof, n_bins=5,
                                 strategy="quantile")
print("forecast", *[f"{v:.3f}" for v in mean_p])
print("observed", *[f"{v:.3f}" for v in frac])
fpr, tpr, thr = roc_curve(mal, oof)
t = thr[np.argmax(tpr >= 0.99)]
print(f"threshold for recall 0.99 out of fold: {t:.3f}, "
      f"false alarms {fpr[np.argmax(tpr >= 0.99)]:.3f}")

model.fit(Xtr, ytr)                  # the one look at the test set
p = model.predict_proba(Xte)[:, 0]
mal_te = yte == 0
print(f"test AUC {roc_auc_score(mal_te, p):.4f}  "
      f"log loss {log_loss(mal_te, p):.4f}  "
      f"Brier {brier_score_loss(mal_te, p):.4f}")
for label, cut in (("0.5", 0.5), (f"{t:.3f}", t)):
    tn, fp, fn, tp = confusion_matrix(mal_te, p >= cut).ravel()
    print(f"cut {label:6s} caught {tp} of {tp + fn}, "
          f"false alarms {fp} of {tn + fp}, "
          f"precision {tp / (tp + fp):.3f}")
