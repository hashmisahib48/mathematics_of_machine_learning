# Leave-one-out error is at most (number of support vectors) / n.
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.svm import SVC

X, y = make_moons(200, noise=0.3, random_state=1)
print("     C   n_SV/n   LOO error   bound holds")
for C in (0.1, 1, 10, 100, 1000):
    m = SVC(C=C, gamma=1.0).fit(X, y)
    ratio = len(m.support_) / len(y)
    loo = 1 - cross_val_score(SVC(C=C, gamma=1.0), X, y,
                              cv=LeaveOneOut()).mean()
    ok = loo <= ratio + 1e-12
    print(f"{C:6g} {ratio:8.3f} {loo:11.3f}   {ok}")
