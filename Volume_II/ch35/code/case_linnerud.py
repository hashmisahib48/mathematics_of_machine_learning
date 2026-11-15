# Case study: Linnerud, three exercises and three body measures.
import numpy as np
from sklearn.cross_decomposition import (CCA, PLSCanonical,
                                         PLSRegression, PLSSVD)
from sklearn.datasets import load_linnerud
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = load_linnerud()
X, Y = d.data, d.target
cv = KFold(5, shuffle=True, random_state=0)   # as in Chapter 33
models = {"mean of the training part": DummyRegressor(),
          "Ridge, alpha = 5": make_pipeline(StandardScaler(),
                                            Ridge(alpha=5.0)),
          "PLSRegression, A = 1": PLSRegression(1),
          "PLSRegression, A = 2": PLSRegression(2),
          "PLSRegression, A = 3": PLSRegression(3)}
heads = "".join(f"{t:>8s}" for t in d.target_names)
print(f"{'5-fold R2':27s}{heads}{'all':>8s}")
for name, m in models.items():
    r2 = r2_score(Y, cross_val_predict(m, X, Y, cv=cv),
                  multioutput="raw_values")
    print(f"{name:27s}" + "".join(f"{v:8.3f}" for v in r2)
          + f"{r2.mean():8.3f}")

print("\nfirst pair of directions (scaled data)")
z = lambda a: (a - a.mean(0)) / a.std(0, ddof=1)    # as scale=True
Xs, Ys = z(X), z(Y)
for est in (PLSRegression(3), PLSCanonical(3), CCA(3), PLSSVD(3)):
    est.fit(X, Y)
    w, c = est.x_weights_[:, 0], est.y_weights_[:, 0]
    s, u = Xs @ w, Ys @ (c / np.linalg.norm(c))
    r, cv_ = np.corrcoef(s, u)[0, 1], np.cov(s, u)[0, 1]
    print(f"{type(est).__name__:15s} w = {w.round(3)}"
          f"  corr = {r:.3f}  cov = {cv_:.3f}")
