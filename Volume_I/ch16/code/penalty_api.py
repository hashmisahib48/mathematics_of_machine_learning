# The four penalties in scikit-learn 1.9.1, with warnings as errors.
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)

settings = {
    "L2 (default)": dict(),
    "L2, C = 0.1": dict(C=0.1, l1_ratio=0),
    "L1": dict(C=0.1, l1_ratio=1, solver="liblinear"),
    "elastic net": dict(C=0.1, l1_ratio=0.5, solver="saga"),
    "none": dict(C=np.inf),
}
warnings.simplefilter("error")                # any warning -> error
for name, kw in settings.items():
    m = LogisticRegression(max_iter=5000, random_state=0, **kw)
    m.fit(X, y)
    nz = int((m.coef_ != 0).sum())
    print(f"{name:13s} nonzero weights {nz:2d}   "
          f"|w|_2 = {np.linalg.norm(m.coef_):6.3f}")

warnings.simplefilter("always")
with warnings.catch_warnings(record=True) as rec:
    LogisticRegression(penalty="l2").fit(X, y)
print(rec[0].category.__name__ + ":", str(rec[0].message)[:52])
for bad in (dict(solver="lbfgs", l1_ratio=1),
            dict(solver="liblinear", l1_ratio=0.5)):
    try:
        LogisticRegression(**bad).fit(X, y)
    except ValueError as e:
        print("ValueError:", str(e)[:56])
