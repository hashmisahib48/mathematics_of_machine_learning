# Growing a forest tree by tree: out-of-bag error against test error.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

forest = RandomForestClassifier(
    n_estimators=25, warm_start=True, oob_score=True, random_state=0)
print("trees   OOB error   test error")
for B in (25, 50, 100, 200, 300, 400):
    forest.set_params(n_estimators=B).fit(X_tr, y_tr)
    oob, test = 1 - forest.oob_score_, 1 - forest.score(X_te, y_te)
    print(f"{B:5d}   {oob:9.4f}   {test:10.4f}")
