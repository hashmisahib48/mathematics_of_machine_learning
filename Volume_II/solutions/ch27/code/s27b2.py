# Problem 27B.2: exponential-loss boosting against AdaBoost.
import numpy as np
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
kw = dict(n_estimators=100, random_state=0)
gbe = GradientBoostingClassifier(loss="exponential", max_depth=1,
                                 **kw).fit(Xtr, ytr)
gbl = GradientBoostingClassifier(max_depth=1, **kw).fit(Xtr, ytr)
ada = AdaBoostClassifier(**kw).fit(Xtr, ytr)

for name, m in (("gradient boosting, exponential", gbe),
                ("AdaBoost, SAMME", ada),
                ("gradient boosting, log loss", gbl)):
    print(f"{name:31s} test accuracy {m.score(Xte, yte):.4f}")
same = (gbe.predict(Xte) == ada.predict(Xte)).mean()
print("same prediction on", round(same, 4), "of the test set")

p = gbe.predict_proba(Xte)[:, 1]
q = expit(2 * gbe.decision_function(Xte))
print("predict_proba == expit(2 F):", np.allclose(p, q))
pa = ada.predict_proba(Xte)[:, 1]
print("probability range, exponential:", p.min().round(4),
      p.max().round(4))
print("probability range, AdaBoost    :", pa.min().round(4),
      pa.max().round(4))

s = 2 * ytr - 1                           # labels +1 / -1
F_ada = 0.5 * sum(a * (2 * h.predict(Xtr) - 1) for h, a in
                  zip(ada.estimators_, ada.estimator_weights_))
F_gb = gbe.decision_function(Xtr)
print("training mean of exp(-s F): gradient boosting",
      round(np.exp(-s * F_gb).mean(), 4), " AdaBoost",
      round(np.exp(-s * F_ada).mean(), 4))
