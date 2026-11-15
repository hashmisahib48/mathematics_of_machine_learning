# Case study: breast cancer, AdaBoost against its relatives.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DTC

cd = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(
    cd.data, cd.target, test_size=0.25, random_state=0,
    stratify=cd.target)
ada50 = AdaBoostClassifier(random_state=0)
ada200 = AdaBoostClassifier(n_estimators=200, random_state=0)
models = {
    "one stump": DTC(max_depth=1, random_state=0),
    "full tree": DTC(random_state=0),
    "AdaBoost, 50 stumps": ada50,
    "AdaBoost, 200 stumps": ada200,
    "AdaBoost, 50 trees of depth 3": AdaBoostClassifier(
        DTC(max_depth=3), random_state=0),
    "logistic regression": make_pipeline(
        StandardScaler(), LogisticRegression()),
}
print(f"{'model':31s} 5-fold    test")
for name, m in models.items():
    cv = cross_val_score(m, Xtr, ytr, cv=5).mean()
    test = m.fit(Xtr, ytr).score(Xte, yte)
    print(f"{name:31s} {cv:.4f}  {test:.4f}")

tr = list(ada200.staged_score(Xtr, ytr))
te = list(ada200.staged_score(Xte, yte))
print("stumps  train    test")
for m in (1, 5, 10, 25, 50, 100, 200):
    print(f"{m:6d}  {tr[m-1]:.4f}  {te[m-1]:.4f}")
for name, m in (("AdaBoost", ada200), ("logistic", models[
        "logistic regression"])):
    p = m.predict_proba(Xte)
    print(f"{name:9s} largest probability {p.max():.4f}"
          f"  log loss {log_loss(yte, p):.4f}")
