# Problem 25B.2: bagging stable and unstable trees on the moons
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.33, random_state=0, stratify=y)
print("train %d, test %d" % (len(y_tr), len(y_te)))
print("depth  one tree  bagged  mean of 100  spread  leaves")
for d in (1, 3, None):
    one = DecisionTreeClassifier(max_depth=d, random_state=0)
    one.fit(X_tr, y_tr)
    bag = BaggingClassifier(DecisionTreeClassifier(max_depth=d),
                            n_estimators=100, random_state=0)
    bag.fit(X_tr, y_tr)
    trees = bag.estimators_
    # P(class 1) of every tree at every test point
    P = np.array([t.predict_proba(X_te)[:, 1] for t in trees])
    spread = P.var(axis=0).mean()           # sigma^2 of Eq. 25.3-8
    mean_acc = np.mean([t.score(X_te, y_te) for t in trees])
    leaves = np.mean([t.get_n_leaves() for t in trees])
    print("%5s %9.4f %7.4f %11.4f %7.4f %7.1f" % (
        d, one.score(X_te, y_te), bag.score(X_te, y_te),
        mean_acc, spread, leaves))
