# OneVsRestClassifier and OneVsOneClassifier on the six points.
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.multiclass import OneVsOneClassifier as OvO
from sklearn.multiclass import OneVsRestClassifier as OvR

X = np.array([[1.], [2], [4], [5], [7], [8]])
y = np.array([0, 0, 1, 1, 2, 2])
base = RidgeClassifier(alpha=1e-8)     # fits targets -1 and +1

def slopes(m):
    return [round(float(e.coef_.item()), 4) + 0.0
            for e in m.estimators_]

ovr = OvR(base).fit(X, y)
print("OvR classes_:", ovr.classes_, " fitted:",
      len(ovr.estimators_))
print("  slopes     :", slopes(ovr))
print("  scores x=4 :", ovr.decision_function([[4.]]).round(4))
print("  predict    :", ovr.predict(X))
native = RidgeClassifier(alpha=1e-8).fit(X, y)
same = np.allclose(native.decision_function(X),
                   ovr.decision_function(X))
print("  native RidgeClassifier agrees:", same)

ovo = OvO(base).fit(X, y)
print("OvO fitted   :", len(ovo.estimators_), "pairs")
print("  slopes     :", slopes(ovo))
print("  scores x=4 :", ovo.decision_function([[4.]]).round(4))
print("  predict    :", ovo.predict(X))

try:                         # a label matrix is not a target
    LogisticRegression().fit(X, np.eye(3)[y])
except ValueError as e:
    print("ValueError:", e)
