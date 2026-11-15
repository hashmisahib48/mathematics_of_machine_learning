# Lasso, ElasticNet and lasso_path on the delivery data.
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, lasso_path

X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]], float)
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])   # km and lights -> min

lasso = Lasso(alpha=1.0).fit(X, y)              # Eq. 11.3-1
print("Lasso(alpha=1.0)")
print("  coef_      =", np.round(lasso.coef_, 4))
print("  intercept_ =", round(lasso.intercept_, 4))
print("  n_iter_    =", lasso.n_iter_)
print("  dual_gap_  =", f"{lasso.dual_gap_:.2e}")
Xc, yc = X - X.mean(axis=0), y - y.mean()    # centre
n, w = len(y), lasso.coef_
r = yc - Xc @ w                              # residual
s = min(1.0, 1.0 / np.abs(Xc.T @ r / n).max())
gap = (r @ r / (2 * n) + np.abs(w).sum()
       - (s * r @ yc - 0.5 * s * s * r @ r) / n)     # Eq. 11.6-1
print("  Eq. 11.6-1 =", f"{gap:.2e}")
print("  predict    =", np.round(lasso.predict([[6, 6]]), 3))
print("  score (R2) =", round(lasso.score(X, y), 4))

big = Lasso(alpha=3.4).fit(X, y)                # alpha = alpha_max
print("Lasso(alpha=3.4) coef_ =", big.coef_, " b =", big.intercept_)

en = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X, y)   # Eq. 11.4-1
print("ElasticNet(1.0, 0.5) coef_ =", np.round(en.coef_, 4))

grid = [3.4, 2.4, 1.4, 1.0, 0.5, 0.1]
alphas, coefs, gaps = lasso_path(Xc, yc, alphas=grid)
print("\nlasso_path on centred data")
print("  alpha     w1      w2")
for a, (w1, w2) in zip(alphas, coefs.T):
    print(f"  {a:4.1f}  {w1:7.4f} {w2:7.4f}")
