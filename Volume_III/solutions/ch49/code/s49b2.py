# Problem 49B.2: skewness of the breast-cancer features
import numpy as np
from scipy.stats import skew
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import (PowerTransformer,
                                   QuantileTransformer)

data = load_breast_cancer()
X, names = data.data, data.feature_names
pt = PowerTransformer()                    # Yeo-Johnson, standardised
qt = QuantileTransformer(n_quantiles=200,
                         output_distribution="normal")
S = {"raw": skew(X, axis=0),
     "Yeo-Johnson": skew(pt.fit_transform(X), axis=0),
     "quantile": skew(qt.fit_transform(X), axis=0)}

for k, s in S.items():
    a = np.abs(s)
    print(f"{k:12s} mean|skew| {a.mean():6.3f}  "
          f"max|skew| {a.max():6.3f}"
          f"  n(|skew|>1) {(a > 1).sum():2d}")

j = np.abs(S["raw"]).argmax()              # the most skewed feature
print(f"\nmost skewed: {names[j]}, skew {S['raw'][j]:.3f}")
print(f"  lambda {pt.lambdas_[j]:.3f}, after Yeo-Johnson "
      f"{S['Yeo-Johnson'][j]:.3f}")
print("Yeo-Johnson: any feature above 0.5 in |skew|?",
      bool((np.abs(S["Yeo-Johnson"]) > 0.5).any()))

Zq = qt.transform(X)                   # why the quantile map fails
print("\nstill above 1 after the quantile map:")
for k in np.flatnonzero(np.abs(S["quantile"]) > 1):
    col, z = X[:, k], Zq[:, k]
    tie = col == col.min()                 # rows at the minimum
    print(f"  {names[k]:21s}{S['quantile'][k]:7.3f}  "
          f"{tie.sum()} rows = {col.min():g} -> {z[tie][0]:.3f}")
print(f"skew of {names[k]} without those rows: {skew(z[~tie]):.3f}")
