# Hour of day: ordinary and periodic splines.
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

def demand(h):                       # two rush hours, period 24 h
    d = lambda c: np.minimum(abs(h - c), 24 - abs(h - c))
    return (2 + 5 * np.exp(-d(8) ** 2 / 4)
            + 3 * np.exp(-d(17.5) ** 2 / 6))

rng = np.random.default_rng(0)
h = rng.uniform(0, 24, 120)[:, None]
y = demand(h[:, 0]) + rng.normal(0, 0.4, 120)
knots = np.linspace(0, 24, 9)[:, None]       # period = 24 h
print("true demand at 0 h, 24 h:",
      np.round(demand(np.array([0.0, 24.0])), 3))
for mode in ("constant", "periodic"):
    sp = SplineTransformer(knots=knots, extrapolation=mode)
    m = make_pipeline(sp, Ridge(alpha=0.01)).fit(h, y)
    f0, f24 = m.predict([[0.0], [24.0]])
    print(f"{mode:9s} f(0) = {f0:.3f}  f(24) = {f24:.3f}"
          f"  jump = {abs(f0 - f24):.3f}")
