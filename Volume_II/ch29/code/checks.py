# Details of the scikit-learn 1.9.1 implementation, checked by running.
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import check_cv
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(60, noise=0.3, random_state=0)
Xi, yi = load_iris(return_X_y=True)

# 1. ties in hard voting go to the smaller class label
a = ("zero", DummyClassifier(strategy="constant", constant=0))
b = ("one", DummyClassifier(strategy="constant", constant=1))
for w in (None, [1, 3], [3, 1]):
    p = VotingClassifier([a, b], weights=w).fit(X, y).predict(X[:3])
    print("two voters, always 0 and always 1, weights", w, "->", p)
print("hard voting has predict_proba:",
      hasattr(VotingClassifier([a, b]), "predict_proba"))

# 2. soft voting needs probabilities from every member
try:
    VotingClassifier([("svc", SVC()), ("lr", LogisticRegression())],
                     voting="soft").fit(X, y).predict(X)
except AttributeError as e:
    print("soft voting with an SVC:", type(e).__name__, "-", e)

# 3. what stack_method="auto" picks, the folds and the final models
est = [("lr", LogisticRegression(max_iter=2000)), ("svc", SVC()),
       ("tree", DecisionTreeClassifier(max_depth=2))]
s = StackingClassifier(est).fit(X, y)
print("stack_method_", s.stack_method_)
print("default cv:", check_cv(None, y, classifier=True))
print("final estimators:", type(s.final_estimator_).__name__, "and",
      type(StackingRegressor([("ols", LinearRegression())])
           .fit(X, y).final_estimator_).__name__)

# 4. width of the level-1 matrix
fin = LogisticRegression(max_iter=2000)
for name, XX, yy in (("two classes", X, y),
                     ("three classes", Xi, yi)):
    for pt in (False, True):
        s = StackingClassifier(est, fin, passthrough=pt).fit(XX, yy)
        print(f"{name:14s} passthrough={pt!s:5s} width"
              f" {s.transform(XX).shape[1]}")

# 5. number of fits
class Count(LogisticRegression):
    n_fits = 0
    def fit(self, X, y, **kw):
        Count.n_fits += 1
        return super().fit(X, y, **kw)

StackingClassifier([(f"c{i}", Count()) for i in range(3)]).fit(X, y)
print("fits of three members with cv=5:", Count.n_fits)
