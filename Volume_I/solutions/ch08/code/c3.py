# Problem 8C.3: which class is positive?
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_validate, train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print("training part:", len(ytr), " benign (1):", int(ytr.sum()), " malignant (0):", int((1 - ytr).sum()), " share of benign: %.4f" % ytr.mean())
for lab, yy in (("benign = 1   ", ytr), ("malignant = 1", 1 - ytr)):
    p = DummyClassifier().fit(Xtr, yy).predict(Xtr)       # the majority class for every row
    tn, fp, fn, tp = confusion_matrix(yy, p).ravel()
    pr = tp / (tp + fp) if tp + fp else float("nan"); rc = tp / (tp + fn)
    print(f"{lab}: TP {tp} FP {fp} FN {fn} TN {tn}  accuracy {accuracy_score(yy, p):.4f}  precision {pr:.3f}  recall {rc:.1f}  F1 {f1_score(yy, p, zero_division=0):.4f}")
s = ytr.mean()
print("F1 of 'always positive' with share s = %.4f of positives: 2s/(1+s) = %.4f" % (s, 2 * s / (1 + s)))
for lab, yy in (("benign = 1   ", ytr), ("malignant = 1", 1 - ytr)):
    r = cross_validate(DummyClassifier(), Xtr, yy, cv=5, scoring=["accuracy", "f1"])
    print(f"{lab}: 5-fold accuracy {r['test_accuracy'].mean():.3f}  F1 {r['test_f1'].mean():.3f}  (F1 per fold {r['test_f1'].round(3)})")
