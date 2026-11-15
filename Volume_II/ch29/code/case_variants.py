# Stacking variants on the cancer data: passthrough and leakage.
import warnings
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler

def members():
    return [("lr", make_pipeline(sc(), LogisticRegression())),
            ("svc", make_pipeline(sc(), CalibratedClassifierCV(
                SVC(), method="sigmoid", cv=5, ensemble=False))),
            ("rf", RandomForestClassifier(100, random_state=0)),
            ("knn", make_pipeline(sc(), KNeighborsClassifier()))]

def show(name, s):
    p = s.predict_proba(Xte)
    print(f"{name:34s} test {s.score(Xte, yte):.4f}"
          f"  log loss {log_loss(yte, p):.4f}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    raw = StackingClassifier(members(), passthrough=True)
    show("passthrough, default final", raw.fit(Xtr, ytr))
kinds = {str(x.category.__name__) for x in w}
print("  warnings:", len(w), sorted(kinds), "-",
      raw.final_estimator_.n_iter_, "iterations, 30 raw columns")
fin = make_pipeline(sc(), LogisticRegression())
show("passthrough, scaled final", StackingClassifier(
    members(), fin, passthrough=True).fit(Xtr, ytr))

fitted = [(n, m.fit(Xtr, ytr)) for n, m in members()]
leaky = StackingClassifier(fitted, cv="prefit").fit(Xtr, ytr)
show("in-sample level-1 (cv='prefit')", leaky)
print("  weights in-sample:", leaky.final_estimator_.coef_.round(2))
honest = StackingClassifier(members()).fit(Xtr, ytr)
print("  weights out-of-fold:",
      honest.final_estimator_.coef_.round(2))
