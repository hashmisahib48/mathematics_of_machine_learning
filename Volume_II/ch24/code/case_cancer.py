# Case study: one tree on the breast-cancer data, and its depth.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)   # 0 malignant, 1 benign
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
base = DummyClassifier().fit(X_tr, y_tr)
print("baseline test accuracy:", round(base.score(X_te, y_te), 4))

tree = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
cv = cross_val_score(DecisionTreeClassifier(random_state=0),
                     X_tr, y_tr, cv=5)
print("full tree: depth", tree.get_depth(), " leaves",
      tree.get_n_leaves(), " train", tree.score(X_tr, y_tr),
      " test", round(tree.score(X_te, y_te), 4),
      " 5-fold", round(cv.mean(), 4))

print("depth  train   5-fold  test")
for d in (1, 2, 3, 4, 5):
    m = DecisionTreeClassifier(max_depth=d, random_state=0)
    s = cross_val_score(m, X_tr, y_tr, cv=5).mean()
    m.fit(X_tr, y_tr)
    print(f"{d:5d}  {m.score(X_tr, y_tr):.4f}  {s:.4f}  "
          f"{m.score(X_te, y_te):.4f}")

pipe = make_pipeline(StandardScaler(), LogisticRegression())
s = cross_val_score(pipe, X_tr, y_tr, cv=5).mean()
pipe.fit(X_tr, y_tr)
print("logistic regression (Chapter 16, Volume I): test",
      round(pipe.score(X_te, y_te), 4), " 5-fold", round(s, 4))
