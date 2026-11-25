# Numbers printed in the text of Chapter 6 (slates, Example 6.4-1, notes).
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

X = np.array([[1, 1], [2, 2], [1, 3], [4, 4], [5, 2], [3.5, 1]])
t = np.array([12, 15, 13, 24, 27, 20.])
q = np.array([3, 2.5])
d = np.sqrt(((X - q) ** 2).sum(1))
o = np.argsort(d, kind="stable")[:3]
v = 1 / d[o]
print("distances A..F", d.round(4))
print("neighbors", "ABCDEF"[o[0]] + "ABCDEF"[o[1]] + "ABCDEF"[o[2]])
print("weights", v.round(3), "sum v", v.sum().round(3))
print("sum v y", (v * t[o]).sum().round(2), "yhat", ((v * t[o]).sum() / v.sum()).round(3))
Xi, yi = load_iris(return_X_y=True)
for f, (a, b) in enumerate(StratifiedKFold(5).split(Xi, yi), 1):
    p = KNeighborsClassifier(100).fit(Xi[a], yi[a]).predict(Xi[b])
    print("k=100 fold", f, "train per class", np.bincount(yi[a]),
          "predicted per class", np.bincount(p, minlength=3),
          "accuracy", round((p == yi[b]).mean(), 3))
print("5/6 =", round(100 / 120, 4))
print("3np for n=331, p=10:", 3 * 331 * 10)
