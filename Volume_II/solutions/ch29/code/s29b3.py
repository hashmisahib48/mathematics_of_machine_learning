# Problem 29B.3: the final estimator of a stack
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler as sc
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
est = [("lr", make_pipeline(sc(), LogisticRegression())),
       ("rf", RF(100, random_state=0)),
       ("knn", make_pipeline(sc(), KNeighborsClassifier()))]
finals = {"LogisticRegression()": LogisticRegression(),
          "LogisticRegression(C=0.01)": LogisticRegression(C=0.01),
          "DecisionTreeClassifier": DecisionTreeClassifier(
              random_state=0),
          "RandomForestClassifier": RF(100, random_state=0)}
print("final estimator              5-fold   test  errors  weights")
for name, fe in finals.items():
    m = StackingClassifier(est, final_estimator=fe)
    cv = cross_val_score(m, Xtr, ytr, cv=5).mean()
    m.fit(Xtr, ytr)
    f = m.final_estimator_
    w = np.round(f.coef_.ravel(), 2) if hasattr(f, "coef_") else ""
    wrong = int((m.predict(Xte) != yte).sum())
    print(f"{name:27s} {cv:.4f}  {m.score(Xte, yte):.4f}"
          f"  {wrong:5d}  {w}")
