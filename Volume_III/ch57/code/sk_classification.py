# Section 57.8: classification scores in scikit-learn.
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, confusion_matrix,
                             classification_report, log_loss,
                             d2_log_loss_score, matthews_corrcoef,
                             roc_auc_score, top_k_accuracy_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
ytr, yte = 1 - ytr, 1 - yte      # 1 = malignant (positive)
lr = make_pipeline(StandardScaler(), LogisticRegression())
lr.fit(Xtr, ytr)
p, yh = lr.predict_proba(Xte)[:, 1], lr.predict(Xte)
print(confusion_matrix(yte, yh))   # rows true, columns predicted
print(classification_report(yte, yh, digits=3,
      target_names=["benign", "malignant"]))
print(f"MCC {matthews_corrcoef(yte, yh):.4f}  "
      f"AUC {roc_auc_score(yte, p):.4f}  "
      f"log loss {log_loss(yte, p):.4f}  "
      f"Brier {brier_score_loss(yte, p):.4f}  "
      f"D2 {d2_log_loss_score(yte, p):.4f}")

Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd / 16, yd, test_size=0.25,
                                      random_state=0, stratify=yd)
lg = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
P = lg.predict_proba(Xte)
top = [top_k_accuracy_score(yte, P, k=k) for k in (1, 2, 3)]
print("top-1, top-2, top-3 accuracy:", np.round(top, 4))
for kw in (dict(multi_class="ovr"),
           dict(multi_class="ovr", average="weighted"),
           dict(multi_class="ovo")):
    print(kw, round(roc_auc_score(yte, P, **kw), 5))
print("multiclass log loss", round(log_loss(yte, P), 4),
      " Brier", round(brier_score_loss(yte, P), 4))
