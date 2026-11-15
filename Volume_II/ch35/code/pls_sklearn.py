# PLSRegression on the four samples of Section 35.1 (raw readings).
import numpy as np
from sklearn.cross_decomposition import PLSRegression

Xc = np.array([[-3., 1], [-2, -1], [2, 1], [3, -1]])
yc = np.array([0., -1, 2, -1])
X, y = Xc + [10, 5], yc + 4        # readings before centring

for A in (1, 2):
    pls = PLSRegression(n_components=A, scale=False).fit(X, y)
    print(f"A={A}  coef_{pls.coef_.shape} = {pls.coef_.round(4)[0]}"
          f"  intercept_ = {pls.intercept_}")
for name in ("x_weights_", "x_loadings_", "x_rotations_",
             "x_scores_"):
    print(f"{name:13s}", getattr(pls, name).T.round(4).tolist())
print(f"{'y_loadings_':13s}", pls.y_loadings_.round(4).tolist())
print("predict([[11, 4]]) =", pls.predict([[11, 4]]).round(4))
auto = PLSRegression(n_components=1).fit(X, y)        # scale=True
print("scale=True, A=1: coef_ =", auto.coef_.round(4)[0],
      " n_iter_ =", auto.n_iter_)
try:
    PLSRegression(n_components=3).fit(X, y)
except ValueError as err:
    print("ValueError:", err)
