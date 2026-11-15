import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import nan_euclidean_distances
from clinic import X
warnings.simplefilter("ignore", ConvergenceWarning)  # max_iter < 10

# Example 51.6-1: distances from patient 2 (Eq. 51.4-1), k = 2
d = nan_euclidean_distances(X[1:2], X)[0]
donors = [0, 2, 3, 4, 5]                  # bp seen: not patient 2
near = sorted(donors, key=lambda j: d[j])[:2]
print("d(2, j), j = 1, 3, 4, 5, 6:", d[donors].round(2))
print("nearest:", [j + 1 for j in near], "-> bp", X[near, 1].mean())
assert KNNImputer(n_neighbors=2).fit_transform(X)[1, 1] == 137.5

# Example 51.6-2: chained least squares on a 4 x 2 table
T = np.array([[1, 1], [2, 2], [3, np.nan], [np.nan, 3]])
def fit(x, y):                            # least-squares line
    xc = x - x.mean()
    b = (xc * (y - y.mean())).sum() / (xc ** 2).sum()
    return y.mean() - b * x.mean(), b
F = np.where(np.isnan(T), np.nanmean(T, axis=0), T)   # mean start
for rnd in (1, 2):
    a, b = fit(F[:3, 1], F[:3, 0]); F[3, 0] = a + b * F[3, 1]
    r = [0, 1, 3]
    a, b = fit(F[r, 0], F[r, 1]); F[2, 1] = a + b * F[2, 0]
    print(f"round {rnd}: x1[4] = {F[3, 0]:.4f}, "
          f"x2[3] = {F[2, 1]:.4f}")
    S = IterativeImputer(estimator=LinearRegression(), max_iter=rnd,
                         tol=0, random_state=0).fit_transform(T)
    assert np.allclose(S, F)
