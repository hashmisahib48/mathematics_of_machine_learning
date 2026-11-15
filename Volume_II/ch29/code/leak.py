# Leakage: a meta-model trained on in-sample predictions.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(600, 10, n_informative=3, n_redundant=0,
                           flip_y=0.15, random_state=1)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y)

def members():
    return [("lr", make_pipeline(StandardScaler(),
                                 LogisticRegression())),
            ("rf", RandomForestClassifier(100, random_state=0)),
            ("nn1", KNeighborsClassifier(1))]

fitted = [(n, m.fit(Xtr, ytr)) for n, m in members()]
for n, m in fitted:
    print(f"{n:4s} train {m.score(Xtr, ytr):.4f}"
          f"  test {m.score(Xte, yte):.4f}")

honest = StackingClassifier(members()).fit(Xtr, ytr)   # 5-fold
leaky = StackingClassifier(fitted, cv="prefit").fit(Xtr, ytr)
for name, s in (("out-of-fold", honest), ("in-sample", leaky)):
    f = s.final_estimator_
    p = s.predict_proba(Xte)
    print(f"{name:12s} weights {f.coef_.ravel().round(2)}"
          f"  test {s.score(Xte, yte):.4f}"
          f"  log loss {log_loss(yte, p):.4f}")
