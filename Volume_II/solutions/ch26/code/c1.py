# Problem 26C.1: a stump never has weighted error above 1 - 1/K; a learner that does ends fit().
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as DTC

rng = np.random.default_rng(0)
for name, data, K in (("breast cancer", load_breast_cancer(), 2), ("iris", load_iris(), 3)):
    X, y = data.data, data.target; worst = 0.0
    for _ in range(2000):                       # 2000 random weightings of the samples
        w = rng.dirichlet(np.full(len(y), rng.choice([0.05, 0.3, 1.0, 5.0])))
        h = DTC(max_depth=1).fit(X, y, sample_weight=w)
        worst = max(worst, w[h.predict(X) != y].sum())
    print(f"{name:13s} K={K}: largest weighted error of a stump over 2000 weightings = {worst:.4f}   (limit 1-1/K = {1-1/K:.4f})")

# the bound is approached by AdaBoost itself, whose weights make each stump's task harder and harder
Xb, yb = load_breast_cancer(return_X_y=True)
big = AdaBoostClassifier(n_estimators=400, random_state=0).fit(Xb, yb)
e = big.estimator_errors_[:len(big.estimators_)]
print(f"400 rounds on breast cancer: eps_1 = {e[0]:.4f}, eps_100 = {e[99]:.4f}, eps_400 = {e[-1]:.4f}, largest = {e.max():.4f} (all below 0.5)")
xx = np.array([[1.0], [1.0]]); yy = np.array([0, 1])           # two samples that no question can separate
print("two samples with the same x, weights 1/2 each: eps =", 0.5 * (DTC(max_depth=1).fit(xx, yy, sample_weight=[.5, .5]).predict(xx) != yy).sum())

# A learner that is worse than guessing: always answers class 0, on three classes with shares 0.30, 0.35, 0.35
X = rng.normal(size=(100, 2)); y = np.repeat([0, 1, 2], [30, 35, 35])
h = DummyClassifier(strategy="constant", constant=0).fit(X, y)
print("\nconstant learner on three classes: weighted error with equal weights =", (h.predict(X) != y).mean())
try:
    AdaBoostClassifier(DummyClassifier(strategy="constant", constant=0), n_estimators=5).fit(X, y)
except ValueError as e:
    print("AdaBoostClassifier.fit raises ValueError:", e)
alpha = np.log(0.3 / 0.7) + np.log(2)
print(f"Eq. 26.3-2 would give alpha = {alpha:.4f} at eps = 0.7 (K = 3), negative; the threshold is 1-1/K = {1-1/3:.4f}")

# The same learner on two classes, with the labels reversed: eps = 0.8 becomes 0.2
y2 = np.repeat([0, 1], [20, 80]); X2 = rng.normal(size=(100, 2))
h0 = DummyClassifier(strategy="constant", constant=0).fit(X2, y2); h1 = DummyClassifier(strategy="constant", constant=1).fit(X2, y2)
e0, e1 = (h0.predict(X2) != y2).mean(), (h1.predict(X2) != y2).mean()
print(f"\ntwo classes: always-0 has eps = {e0:.2f}, alpha = {np.log((1-e0)/e0):+.4f}; always-1 has eps = {e1:.2f}, alpha = {np.log((1-e1)/e1):+.4f}")
