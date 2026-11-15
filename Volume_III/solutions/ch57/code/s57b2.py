# Problem 57B.2: two models compared by the F2 score of malignant
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)        # 0 = malignant
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
f2 = make_scorer(fbeta_score, beta=2, pos_label=0)
f2_wrong = make_scorer(fbeta_score, beta=2)   # pos_label=1: benign
models = {"logistic": make_pipeline(StandardScaler(),
                                    LogisticRegression()),
          "forest": RandomForestClassifier(random_state=0)}
res = {}
for name, m in models.items():
    s = res[name] = cross_val_score(m, Xtr, ytr, cv=cv, scoring=f2)
    print(f"{name:8s} F2 per fold {np.round(s, 4)}")
    print(f"         mean {s.mean():.4f}  sd {s.std(ddof=1):.4f}")
    w = cross_val_score(m, Xtr, ytr, cv=cv, scoring=f2_wrong).mean()
    p = cross_val_predict(m, Xtr, ytr, cv=cv)
    tp = ((p == 0) & (ytr == 0)).sum()
    fn = ((p == 1) & (ytr == 0)).sum()
    fp = ((p == 0) & (ytr == 1)).sum()
    print(f"         pooled TP {tp} FN {fn} FP {fp};"
          f" F2 without pos_label {w:.4f}")
d = res["logistic"] - res["forest"]
print("per-fold difference", np.round(d, 4))
print(f"mean {d.mean():.4f}  sd {d.std(ddof=1):.4f}  "
      f"logistic better in {(d > 0).sum()} folds of 5")
