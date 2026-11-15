# Case study: calibrating classifiers on the breast-cancer data.
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV as CCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
models = {                                 # estimator, scaled?
    "GaussianNB": (GaussianNB(), False),
    "LinearSVC": (LinearSVC(), True),
    "SVC": (SVC(), True),
    "RandomForest": (RandomForestClassifier(random_state=0), False),
    "AdaBoost": (AdaBoostClassifier(n_estimators=200,
                                    random_state=0), False),
    "LogisticRegression": (LogisticRegression(), True)}
METHODS = ("raw", "sigmoid", "isotonic", "temperature")
rows = {}
for name, (est, scaled) in models.items():
    for meth in METHODS:
        if meth == "raw" and not hasattr(est, "predict_proba"):
            continue                    # SVC has no proba
        m = clone(est) if meth == "raw" else CCV(
            clone(est), method=meth, cv=5, ensemble=False)
        m = make_pipeline(StandardScaler(), m) if scaled else m
        p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        rows[name, meth] = (log_loss(y_te, p), m.score(X_te, y_te))
for k, title in enumerate(("log loss", "accuracy")):
    print(f"{title:19s}" + "".join(f"{s:>12s}" for s in METHODS))
    for name in models:
        print(f"{name:19s}" + "".join(
            f"{rows[name, s][k]:12.4f}" if (name, s) in rows
            else f"{'-':>12s}" for s in METHODS))
