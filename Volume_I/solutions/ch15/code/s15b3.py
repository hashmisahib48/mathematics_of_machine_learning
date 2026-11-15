# Problem 15B.3: the tol threshold of LinearRegression, raw powers.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(0)
year = np.arange(1990, 2020.0)[:, None]           # as Listing 15.5
t = year[:, 0] - 1990
y = 5 + 0.3 * t + 0.02 * t ** 2 + rng.normal(0, 0.5, 30)
tols = [1e-6, 1e-9, 1e-12, 1e-15]

print("degree   tol    rank_    R2    prediction for 2025")
for d in (2, 3, 4):
    pf = PolynomialFeatures(d, include_bias=False)
    for tol in tols:
        m = make_pipeline(pf, LinearRegression(tol=tol))
        m.fit(year, y)
        p25 = m.predict([[2025.0]])[0]
        print(f"{d:6d} {tol:7.0e} {m[-1].rank_:5d} "
              f"{m.score(year, y):8.4f} {p25:9.3f}")
    A = pf.fit_transform(year)
    s = np.linalg.svd(A - A.mean(0), compute_uv=False)  # centred
    print("   singular values / largest:",
          np.array2string(s / s[0], precision=1))
    ref = make_pipeline(StandardScaler(), pf, LinearRegression())
    ref.fit(year, y)                          # the cure: scale first
    print(f"   scaled, default tol: rank_ {ref[-1].rank_}, R2 "
          f"{ref.score(year, y):.4f}, 2025: "
          f"{ref.predict([[2025.]])[0]:.3f}")
