# Case study: benign or malignant, with support vector classifiers.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
p = X.shape[1]
print("gamma = 1/(p Var X): raw", f"{1 / (p * X_tr.var()):.2e}",
      " scaled", round(1 / p, 5))

def report(name, m):
    m.fit(X_tr, y_tr)
    print(f"{name:24s} train {m.score(X_tr, y_tr):.4f}"
          f"  test {m.score(X_te, y_te):.4f}")
    return m

report("SVC, raw features", SVC())
sc = report("SVC, scaled", make_pipeline(StandardScaler(), SVC()))
cv = cross_val_score(sc, X_tr, y_tr, cv=5)
print("  n_support_", sc[-1].n_support_, " 5-fold mean",
      cv.mean().round(4))
report("SVC linear, scaled", make_pipeline(
    StandardScaler(), SVC(kernel="linear")))
report("LinearSVC, scaled", make_pipeline(
    StandardScaler(), LinearSVC()))

grid = {"svc__C": 2.0 ** np.arange(-5, 10, 2),
        "svc__gamma": 2.0 ** np.arange(-15, 2, 2)}
pipe = make_pipeline(StandardScaler(), SVC())
gs = GridSearchCV(pipe, grid, cv=5).fit(X_tr, y_tr)
b = gs.best_estimator_
bp = gs.best_params_
print("best C =", float(bp["svc__C"]), " gamma =",
      float(bp["svc__gamma"]), " CV", round(gs.best_score_, 4))
print(f"tuned: train {b.score(X_tr, y_tr):.4f}"
      f"  test {b.score(X_te, y_te):.4f}"
      f"  n_support_ {b[-1].n_support_}")
print(confusion_matrix(y_te, b.predict(X_te)))
