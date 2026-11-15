# Facts quoted in the text, checked against the installed release.
import inspect
import numpy as np
import sklearn
from sklearn.linear_model import ARDRegression, BayesianRidge

print("scikit-learn", sklearn.__version__)
for cls in (BayesianRidge, ARDRegression):
    sig = inspect.signature(cls)
    print(cls.__name__, {k: v.default for k, v in sig.parameters.items()})
try:
    BayesianRidge(alpha=1.0)
except TypeError as e:
    print("TypeError:", e)
print("fit params BR :", list(inspect.signature(BayesianRidge.fit).parameters))
print("fit params ARD:", list(inspect.signature(ARDRegression.fit).parameters))
X = np.random.default_rng(0).normal(size=(30, 3))
y = X @ [1.0, 0.0, -1.0]
br = BayesianRidge().fit(X, y)
print("BR attrs :", sorted(a for a in vars(br) if a.endswith("_")))
ard = ARDRegression().fit(X, y)
print("ARD attrs:", sorted(a for a in vars(ard) if a.endswith("_")))
print("lambda_ shapes:", np.shape(br.lambda_), np.shape(ard.lambda_))
print("sigma_ shape:", br.sigma_.shape, " scores_ empty:", br.scores_)
m, s = br.predict(X[:2], return_std=True)
print("predict return_std shapes:", m.shape, s.shape)
# does predicted std include intercept uncertainty? compare formula
Xc = X[:2] - br.X_offset_
print("std = sqrt(1/alpha + x'Sx):",
      np.allclose(s, np.sqrt(1 / br.alpha_ + np.sum(Xc @ br.sigma_ * Xc, 1))))
w = np.ones(30)
w[:10] = 3.0
print("sample_weight accepted by BR:",
      BayesianRidge().fit(X, y, sample_weight=w).n_iter_ > 0)
