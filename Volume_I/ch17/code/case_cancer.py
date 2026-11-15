# Case study: malignant or benign? 569 patients, linear classifiers.
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
rs = dict(random_state=0)
alphas = np.logspace(-2, 3, 12)
models = {
    "Perceptron": Perceptron(**rs),
    "SGD hinge": SGDClassifier(**rs),
    "SGD log_loss": SGDClassifier(loss="log_loss", **rs),
    "SGD modified_huber": SGDClassifier(loss="modified_huber",
                                        **rs),
    "SGD hinge, average": SGDClassifier(average=True, **rs),
    "RidgeClassifierCV": RidgeClassifierCV(alphas=alphas),
    "LogisticRegression": LogisticRegression(),
}
print("model                features   test    5-fold CV mean, sd")
for scaled in (False, True):
    for name, est in models.items():
        if not scaled and name.startswith(("Ridge", "Logis")):
            continue                       # these two: scaled only
        pipe = clone(est)
        if scaled:
            pipe = make_pipeline(StandardScaler(), pipe)
        test = pipe.fit(Xtr, ytr).score(Xte, yte)
        cvs = cross_val_score(pipe, Xtr, ytr, cv=cv)
        kind = "scaled" if scaled else "raw"
        print(f"{name:20s} {kind:8s} {test:6.4f}  "
              f"{cvs.mean():.4f} +- {cvs.std():.4f}")
