# Test of MySGD against SGDClassifier on the breast-cancer data.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sgd_scratch import MySGD

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
kw = dict(alpha=1e-3, eta0=0.01, power_t=0.5)

print("fixed order, 3 epochs: largest |w - w_sk| over 9 settings")
worst = 0.0
for loss in ("hinge", "log_loss", "perceptron"):
    for sch in ("constant", "invscaling", "optimal"):
        mine = MySGD(loss, schedule=sch, epochs=3, shuffle=False,
                     **kw).fit(Xtr, ytr)
        ref = SGDClassifier(loss=loss, learning_rate=sch, tol=None,
                            max_iter=3, shuffle=False,
                            **kw).fit(Xtr, ytr)
        worst = max(worst, np.abs(mine.coef_ - ref.coef_[0]).max())
assert worst < 1e-10
print(f"  {worst:.1e}  (assertion passed)")

print("shuffled, 20 epochs: the random orders differ")
mine = MySGD("hinge", epochs=20, seed=1, **kw).fit(Xtr, ytr)
ref = SGDClassifier(alpha=1e-3, max_iter=20, tol=None,
                    random_state=1).fit(Xtr, ytr)
for name, m in (("scratch", mine), ("scikit-learn", ref)):
    acc = np.mean(m.predict(Xte) == yte)
    print(f"  {name:12s} test accuracy {acc:.4f}")
print("  largest |w - w_sk| =",
      np.abs(mine.coef_ - ref.coef_[0]).max().round(3))
