# Years as a feature: polynomial columns without and with scaling.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(0)
year = np.arange(1990, 2020.0)[:, None]          # 30 samples
t = year[:, 0] - 1990
y = 5 + 0.3 * t + 0.02 * t ** 2 + rng.normal(0, 0.5, 30)

print("deg  R2 raw  rank  R2 scaled  rank   cond(raw columns)")
for d in (2, 3, 4, 5):
    pf = PolynomialFeatures(d, include_bias=False)
    raw = make_pipeline(pf, LinearRegression()).fit(year, y)
    sc = make_pipeline(StandardScaler(), PolynomialFeatures(
        d, include_bias=False), LinearRegression()).fit(year, y)
    cond = np.linalg.cond(pf.fit_transform(year))
    print(f"{d:3d}  {raw.score(year, y):6.4f}  {raw[-1].rank_:4d}"
          f"  {sc.score(year, y):9.4f}  {sc[-1].rank_:4d}"
          f"  {cond:12.3e}")
