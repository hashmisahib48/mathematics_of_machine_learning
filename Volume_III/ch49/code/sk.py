# Section 49.4 by hand (NumPy) and in scikit-learn.
import numpy as np
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
    MaxAbsScaler, RobustScaler, QuantileTransformer)

x = np.array([1.0, 2.0, 3.0, 4.0, 10.0]).reshape(-1, 1)
new = np.array([[3.5], [20.0]])             # two later values
q1, med, q3 = np.percentile(x, [25, 50, 75])
hand = {                                    # Eq. 49.3-1
    "standard": lambda v: (v - x.mean()) / x.std(),
    "min-max": lambda v: (v - x.min()) / (x.max() - x.min()),
    "max-abs": lambda v: v / np.abs(x).max(),
    "robust": lambda v: (v - med) / (q3 - q1),
}
tf = {
    "standard": StandardScaler(), "min-max": MinMaxScaler(),
    "max-abs": MaxAbsScaler(), "robust": RobustScaler(),
    "quantile U": QuantileTransformer(n_quantiles=5),
    "quantile N": QuantileTransformer(
        n_quantiles=5, output_distribution="normal"),
}
row = lambda v: " ".join(f"{u:6.3f}" for u in v)
for name, t in tf.items():
    z, zn = t.fit_transform(x), t.transform(new)
    print(f"{name:>11} {row(z.ravel())} | {row(zn.ravel())}")
    if name in hand:                        # NumPy agrees
        assert np.allclose(z, hand[name](x))
        assert np.allclose(zn, hand[name](new))
s, m = tf["standard"], tf["min-max"]
print("standard: mean_", s.mean_, "var_", s.var_,
      "scale_", s.scale_)
print("min-max : scale_", m.scale_.round(4),
      "min_", m.min_.round(4))
print("by-hand formulas agree with scikit-learn")
