# Section 57.7: AUC, ROC curve and average precision from scratch.
import numpy as np
from scipy.stats import rankdata
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def auc_rank(y, s):               # Eq. 57.3-7, mid-ranks
    r = rankdata(s)               # ranks 1..n, ties averaged
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def curve(y, s):                  # one point per distinct score
    o = np.argsort(-s, kind="mergesort")
    y, s = y[o], s[o]
    end = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[end]; fp = 1 + end - tp
    return fp / fp[-1], tp / tp[-1], tp / (tp + fp)

def average_precision(y, s):      # Eq. 57.3-8, steps
    _, tpr, prec = curve(y, s)
    return np.sum(np.diff(np.r_[0, tpr]) * prec)

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
ytr, yte = 1 - ytr, 1 - yte      # 1 = malignant (positive)
lr = make_pipeline(StandardScaler(), LogisticRegression())
s = lr.fit(Xtr, ytr).decision_function(Xte)
for name, sc in (("continuous", s), ("rounded", s.round(1))):
    fpr, tpr, _ = curve(yte, sc)
    x, v = np.r_[0, fpr], np.r_[0, tpr]
    area = np.sum(np.diff(x) * (v[1:] + v[:-1]) / 2)   # trapezoids
    ref = roc_auc_score(yte, sc)
    assert np.isclose(auc_rank(yte, sc), ref)
    assert np.isclose(area, ref)
    ap = average_precision(yte, sc)
    assert np.isclose(ap, average_precision_score(yte, sc))
    print(f"{name:11s} AUC {ref:.4f}  AP {ap:.4f}  "
          f"thresholds {len(fpr)}")
print("roc_curve keeps", len(roc_curve(yte, s)[0]) - 1, "of them")
