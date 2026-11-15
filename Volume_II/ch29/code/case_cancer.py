# Case study: four models and their committees on the cancer data.
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)   # 0 malignant, 1 benign
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler
lr = make_pipeline(sc(), LogisticRegression())
svc = make_pipeline(sc(), CalibratedClassifierCV(
    SVC(), method="sigmoid", cv=5, ensemble=False))
rf = RandomForestClassifier(n_estimators=100, random_state=0)
knn = make_pipeline(sc(), KNeighborsClassifier())
est = [("lr", lr), ("svc", svc), ("rf", rf), ("knn", knn)]
models = dict(est)
models.update({
    "hard vote": VotingClassifier(est),
    "soft vote": VotingClassifier(est, voting="soft"),
    "soft, lr x 2": VotingClassifier(est, voting="soft",
                                     weights=[2, 1, 1, 1]),
    "stack": StackingClassifier(est)})

print("model                 5-fold    test  errors  log loss")
for name, m in models.items():
    cv = cross_val_score(m, Xtr, ytr, cv=5).mean()
    m.fit(Xtr, ytr)
    wrong = int((m.predict(Xte) != yte).sum())
    ll = "       -"
    if hasattr(m, "predict_proba"):
        ll = f"{log_loss(yte, m.predict_proba(Xte)):8.4f}"
    print(f"{name:20s}  {cv:.4f}  {m.score(Xte, yte):.4f}  "
          f"{wrong:6d}  {ll}")
