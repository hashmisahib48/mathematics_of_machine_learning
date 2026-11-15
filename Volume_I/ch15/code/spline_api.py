# SplineTransformer: columns, partition of unity, extrapolation.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

x = np.linspace(0, 10, 25)[:, None]
y = np.sin(x[:, 0] / 2) + 0.1 * np.cos(7 * x[:, 0])
sp = SplineTransformer(n_knots=5, degree=3).fit(x)
Z = sp.transform(x)
print("columns:", sp.n_features_out_, "= 5 + 3 - 1")
print("knots incl. extra:", sp.bsplines_[0].t)
nz = sorted(set((Z > 0).sum(1).tolist()))
print("row sums:", np.round(Z.sum(1)[:3], 6), "  non-zero:", nz)
for d in (0, 1):
    s = SplineTransformer(n_knots=5, degree=d).fit(x)
    print(f"degree {d}: {s.n_features_out_} columns")
s = SplineTransformer(include_bias=False).fit(x)
print("include_bias=False:", s.n_features_out_, "columns")

new = np.array([[-2.0], [0.0], [10.0], [12.0]])
for mode in ("constant", "linear", "continue", "error"):
    sp = SplineTransformer(n_knots=5, extrapolation=mode).fit(x)
    fit = LinearRegression().fit(sp.transform(x), y)
    try:
        pred = fit.predict(sp.transform(new))
        print(f"{mode:9s}", np.round(pred, 3))
    except ValueError as e:
        print(f"{mode:9s} ValueError:", str(e)[:40])
