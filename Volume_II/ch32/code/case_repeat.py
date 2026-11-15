# Not printed: three more splits of the breast-cancer data.
import json
import time
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV as CCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

t0 = time.time()
X, y = load_breast_cancer(return_X_y=True)
models = {
    "GaussianNB": (GaussianNB(), False),
    "LinearSVC": (LinearSVC(), True),
    "SVC": (SVC(), True),
    "RandomForest": (RandomForestClassifier(random_state=0), False),
    "AdaBoost": (AdaBoostClassifier(n_estimators=200,
                                    random_state=0), False),
    "LogisticRegression": (LogisticRegression(), True)}
METHODS = ("raw", "sigmoid", "isotonic", "temperature")
acc = {}
for seed in range(1, 4):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    for name, (est, scaled) in models.items():
        for meth in METHODS:
            if meth == "raw" and not hasattr(est, "predict_proba"):
                continue
            m = clone(est) if meth == "raw" else CCV(
                clone(est), method=meth, cv=5, ensemble=False)
            m = make_pipeline(StandardScaler(), m) if scaled else m
            p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
            acc.setdefault(f"{name}|{meth}", []).append(
                (log_loss(y_te, p), brier_score_loss(y_te, p)))
out = {}
for k, v in acc.items():
    v = np.array(v)
    out[k] = {"logloss_mean": float(v[:, 0].mean()),
              "logloss_min": float(v[:, 0].min()),
              "logloss_max": float(v[:, 0].max()),
              "brier_mean": float(v[:, 1].mean())}
    print(f"{k:32s} log loss {v[:, 0].mean():.4f} "
          f"({v[:, 0].min():.4f}-{v[:, 0].max():.4f})  "
          f"Brier {v[:, 1].mean():.4f}")
json.dump(out, open("../verify/case_repeat.json", "w"), indent=1)
print("seconds", round(time.time() - t0))
