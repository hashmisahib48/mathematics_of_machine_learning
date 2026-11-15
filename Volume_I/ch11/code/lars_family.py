# The LARS family on the data of Listing 11.4.
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import (Lars, Lasso, LassoLars,
                                  LassoLarsCV, LassoLarsIC)

X, y, w_true = make_regression(
    n_samples=120, n_features=20, n_informative=4, noise=25.0,
    coef=True, random_state=2)
print("true non-zero features:", np.flatnonzero(w_true))

lars = Lars(n_nonzero_coefs=4).fit(X, y)     # plain LARS, 4 steps
print("Lars     order of entry:", [int(j) for j in lars.active_])
print("         alphas_ =", np.round(lars.alphas_, 2))

ll = LassoLars(alpha=2.3904).fit(X, y)       # Eq. 11.3-1 by LARS
cd = Lasso(alpha=2.3904, tol=1e-12).fit(X, y)  # coord. descent
gap = np.abs(ll.coef_ - cd.coef_).max()
print("LassoLars vs Lasso, max |difference| =", f"{gap:.1e}")
print("         path nodes:", ll.alphas_.shape[0])

cv = LassoLarsCV(cv=5).fit(X, y)
print("LassoLarsCV alpha_ =", round(cv.alpha_, 4))
print("   selected:", np.flatnonzero(cv.coef_))

for crit in ("aic", "bic"):                     # Eq. 11.10-3
    ic = LassoLarsIC(criterion=crit).fit(X, y)
    print(f"LassoLarsIC {crit}: alpha_ = {ic.alpha_:.4f}")
    print("   selected:", np.flatnonzero(ic.coef_))
print("noise_variance_ =", round(ic.noise_variance_, 2),
      "  criterion_ has", ic.criterion_.shape[0], "values")
