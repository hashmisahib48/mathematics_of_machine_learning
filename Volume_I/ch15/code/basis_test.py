# Test of the two classes against scikit-learn.
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import SplineTransformer
from basis_scratch import MyPoly, MySpline

rng = np.random.default_rng(0)
X = rng.normal(size=(6, 3))
for d in (1, 2, 3):
    ours = MyPoly(d).fit(X).transform(X)
    ref = PolynomialFeatures(d).fit_transform(X)
    assert ours.shape == ref.shape and np.allclose(ours, ref)
    print(f"degree {d}: {ours.shape[1]:2d} columns agree")

x = rng.uniform(0, 10, 40)
grid = np.linspace(-1, 11, 200)       # reaches outside the data
for n_knots, d in ((5, 3), (8, 3), (4, 2), (3, 1)):
    B = MySpline(n_knots, d).fit(x).transform(grid)
    ref = SplineTransformer(n_knots, d).fit(x[:, None])
    assert np.allclose(B, ref.transform(grid[:, None]))
    rs = B.sum(1)
    print(f"n_knots={n_knots} degree={d}: {B.shape[1]:2d} columns "
          f"agree; row sums {rs.min():.1f}..{rs.max():.1f}")

# Ridge on the spline columns, Eq. 15.2-3 with alpha = 0.1
y = np.sin(x) + 0.3 * rng.normal(size=40)
Phi = MySpline(8, 3).fit(x).transform(x)[:, :-1]  # drop one column
P, ym = Phi - Phi.mean(0), y.mean()
A = P.T @ P + 0.1 * np.eye(P.shape[1])
w = np.linalg.solve(A, P.T @ (y - ym))
b = ym - Phi.mean(0) @ w
sk = make_pipeline(SplineTransformer(8, 3, include_bias=False),
                   Ridge(alpha=0.1)).fit(x[:, None], y)
mine = MySpline(8, 3).fit(x).transform(grid)[:, :-1] @ w + b
gap = np.abs(mine - sk.predict(grid[:, None])).max()
assert gap < 1e-10
print(f"ridge on spline columns agrees, max difference {gap:.1e}")
