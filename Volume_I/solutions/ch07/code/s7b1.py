# Problem 7B.1: the wine data and the sizes of its features
import numpy as np
from sklearn.datasets import load_wine

w = load_wine(as_frame=True)
X, y = w.data, w.target
print("shape:", X.shape)
print("wines per class:", np.bincount(y).tolist())

sd = X.std(ddof=0)                   # divide by n, like the scaler
big, small = sd.idxmax(), sd.idxmin()
print(f"largest  std: {big:22s} {sd[big]:9.4f}")
print(f"smallest std: {small:22s} {sd[small]:9.4f}")
print(f"ratio of the two: {sd[big] / sd[small]:.0f}")
print(f"ratio of the variances: {(sd[big] / sd[small]) ** 2:.3g}")
print("pandas default (n - 1):", round(X.std()[big], 4))

var = X.var(ddof=0)                  # share of the total variance
share = (var / var.sum()).sort_values(ascending=False)
print(f"share of total variance: {big} {share.iloc[0]:.4f},",
      f"next {share.index[1]} {share.iloc[1]:.4f}")
print("features with std > 10:", int((sd > 10).sum()), "of", len(sd))

# squared distance between two wines, raw and standardised
a, b = X.iloc[0], X.iloc[100]
z = (X - X.mean()) / sd
d_raw = ((a - b) ** 2)
d_std = (((z.iloc[0] - z.iloc[100]) ** 2))
print(f"wines 0 and 100: {big} gives {d_raw[big] / d_raw.sum():.4f}",
      f"of the squared distance raw, {d_std[big] / d_std.sum():.4f}",
      "standardised")
