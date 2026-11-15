# Problem 26B.1: after the update the last stump has error 1/2.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

cd = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(
    cd.data, cd.target, test_size=0.25, random_state=0,
    stratify=cd.target)
ada = AdaBoostClassifier(n_estimators=10, random_state=0)
ada.fit(Xtr, ytr)

n = len(ytr)
w = np.full(n, 1 / n)                    # round-1 weights
print(" m   eps_m  alpha_m  error under next weights")
dev = []
for m, (h, a) in enumerate(zip(ada.estimators_,
                               ada.estimator_weights_), 1):
    wrong = h.predict(Xtr) != ytr
    eps = w[wrong].sum()                 # Eq. 26.3-1
    assert np.isclose(a, np.log((1 - eps) / eps))   # Eq. 26.3-2
    w = w * np.exp(a * wrong)            # Eq. 26.3-3 ...
    w = w / w.sum()                      # ... and normalise
    err = w[wrong].sum()                 # same stump, new weights
    dev.append(abs(err - 0.5))
    print(f"{m:2d} {eps:7.4f} {a:8.4f}  {err:.16f}")
print("largest |error - 1/2|:", max(dev))
print("weights add up to", w.sum(), " largest weight x n:",
      round(w.max() * n, 2))
