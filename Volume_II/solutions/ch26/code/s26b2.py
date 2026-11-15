# Problem 26B.2: a weak learner that is too strong ends the loop.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cd = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(
    cd.data, cd.target, test_size=0.25, random_state=0,
    stratify=cd.target)

print("depth  rounds  eps_1   alpha_1  tree1 train  test")
for d in (1, 2, 3, 5, None):
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=d),
                             random_state=0).fit(Xtr, ytr)
    k = len(ada.estimators_)
    tree1 = ada.estimators_[0]
    eps1 = ada.estimator_errors_[0]      # weighted error of round 1
    a1 = ada.estimator_weights_[0]
    print(f"{str(d):>5s} {k:6d}  {eps1:6.4f} {a1:8.4f}"
          f"   {tree1.score(Xtr, ytr):.4f}   "
          f"{ada.score(Xte, yte):.4f}")

ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),
                         random_state=0).fit(Xtr, ytr)
print("\nmax_depth=5: estimator_weights_[:3] =",
      ada.estimator_weights_[:3])
print("estimator_errors_[:3]  =", ada.estimator_errors_[:3])
tree = ada.estimators_[0]
print("depth of the tree:", tree.get_depth(), " leaves:",
      tree.get_n_leaves(), " training errors:",
      int((tree.predict(Xtr) != ytr).sum()), " of", len(ytr))
