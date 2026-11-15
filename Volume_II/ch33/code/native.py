# Which estimators take the three kinds of target without a wrapper?
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
big = X[:, 0] > 5.8
targets = {"multiclass": y,
           "multilabel": np.c_[y == 0, y == 1, big].astype(int),
           "multioutput": np.c_[y, big]}
models = [LogisticRegression(max_iter=300), LinearSVC(), SVC(),
          RidgeClassifier(), LinearDiscriminantAnalysis(),
          GaussianNB(), GB(n_estimators=5), KNeighborsClassifier(),
          DecisionTreeClassifier(), RF(n_estimators=5),
          MLPClassifier(max_iter=3000, random_state=0)]
print(f"{'estimator':28s}" + "".join(f"{t:>12s}" for t in targets))
for m in models:
    row = ""
    for t in targets.values():
        try:
            clone(m).fit(X, t)
            row += f"{'yes':>12s}"
        except ValueError:
            row += f"{'no':>12s}"
    print(f"{type(m).__name__:28s}" + row)
