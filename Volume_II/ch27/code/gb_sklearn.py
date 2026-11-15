# Gradient boosting in scikit-learn: the toy data of Chapter 27.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)

x = np.array([[1], [2], [3], [4], [5], [6.]])
y = np.array([3, 5, 4, 10, 14, 12.])
reg = GradientBoostingRegressor(
    n_estimators=2, max_depth=1, learning_rate=0.5, random_state=0)
reg.fit(x, y)
print("init_ constant   ", reg.init_.constant_.ravel())
for m, tree in enumerate(reg.estimators_[:, 0], 1):
    t = tree.tree_
    print(f"tree {m}: x <= {t.threshold[0]},",
          "leaves", t.value[1:3, 0, 0])
print("train_score_     ", reg.train_score_.round(4))
for m, p in enumerate(reg.staged_predict([[3.8]]), 1):
    print(f"prediction at 3.8 after {m} tree(s): {p[0]:.4f}")

# One round for the ten loan applicants (age, income, repaid).
age = [35, 27, 22, 48, 33, 24, 52, 29, 41, 58]
inc = [22, 40, 55, 28, 62, 18, 36, 26, 48, 75]
z = [0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
X = np.c_[age, inc]
clf = GradientBoostingClassifier(
    n_estimators=1, max_depth=1, learning_rate=0.5, random_state=0)
clf.fit(X, z)
t = clf.estimators_[0, 0].tree_
print("\nprior", clf.init_.class_prior_, " split on feature",
      t.feature[0], "at", t.threshold[0])
print("leaves", t.value[1:3, 0, 0].round(4))
print("decision_function", clf.decision_function(X[:3]).round(4))
print("predict_proba(1) ", clf.predict_proba(X[:3])[:, 1].round(4))

# Three classes: one tree per class in every stage.
Xi, yi = load_iris(return_X_y=True)
g = GradientBoostingClassifier(n_estimators=5, random_state=0)
g.fit(Xi, yi)
print("\nestimators_", g.estimators_.shape,
      " apply", g.apply(Xi).shape)
print("n_trees_per_iteration_", g.n_trees_per_iteration_)
