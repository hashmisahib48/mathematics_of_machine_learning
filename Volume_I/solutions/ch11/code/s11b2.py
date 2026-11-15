# Problem 11B.2: OMP and the units of the columns
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from sklearn.preprocessing import StandardScaler

dist = np.array([1., 2, 3, 4, 5])
lights = np.array([2., 1, 4, 3, 5])
t = np.array([3., 4, 6, 7, 10])
data = {"raw": np.c_[dist, lights],
        "lights x 10": np.c_[dist, 10 * lights]}
data["standardised"] = StandardScaler().fit_transform(
    data["lights x 10"])

for name, X in data.items():
    omp = OMP(n_nonzero_coefs=1).fit(X, t)
    Xc, yc = X - X.mean(0), t - t.mean()
    score = np.abs(Xc.T @ yc)                # Eq. 11.11-2, step 1
    j = score.argmax()
    x = Xc[:, j]
    w = x @ yc / (x @ x)                     # Eq. 11.11-3
    print(name)
    print("  scores", np.round(score, 3), "-> feature", j + 1)
    print("  coef_", np.round(omp.coef_, 4), " by hand", round(w, 4),
          " b =", round(omp.intercept_, 4))
print("correlations with the time:",
      round(np.corrcoef(dist, t)[0, 1], 4),
      round(np.corrcoef(lights, t)[0, 1], 4))
