# Orthogonal matching pursuit: a hand-sized run, then scikit-learn.
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import (OrthogonalMatchingPursuit,
                                  OrthogonalMatchingPursuitCV)

X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]], float)
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
Xc, yc = X - X.mean(axis=0), y - y.mean()


def omp(Xc, yc, k):
    chosen, r = [], yc.copy()
    for step in range(1, k + 1):
        score = np.abs(Xc.T @ r)                   # Eq. 11.11-2
        score[chosen] = 0.0
        chosen.append(int(score.argmax()))
        w = np.linalg.lstsq(Xc[:, chosen], yc, rcond=None)[0]
        r = yc - Xc[:, chosen] @ w                 # Eq. 11.11-3
        print(f"step {step}: scores {np.round(score, 2)}"
              f" -> feature {chosen[-1] + 1}")
        print("  w =", np.round(w, 4), f" |r|^2 = {r @ r:.4f}")
        print("  residual", np.round(r, 2))
    return chosen, w


omp(Xc, yc, 2)
one = OrthogonalMatchingPursuit(n_nonzero_coefs=1).fit(X, y)
print("scikit-learn, 1 term:", one.coef_, round(one.intercept_, 3))

X, y, w_true = make_regression(
    n_samples=120, n_features=20, n_informative=4, noise=25.0,
    coef=True, random_state=2)
m = OrthogonalMatchingPursuit(n_nonzero_coefs=4).fit(X, y)
print("OMP, 4 terms selected:", np.flatnonzero(m.coef_))
cv = OrthogonalMatchingPursuitCV(cv=5).fit(X, y)
print("OMPCV: n_nonzero_coefs_ =", cv.n_nonzero_coefs_,
      " selected:", np.flatnonzero(cv.coef_))
