# Sample mean, sample variance and the law of large numbers.
import numpy as np

d = np.array([2.0, 4, 4, 6])
n = len(d)
print(f"mean = {d.mean():.4f}")
print(f"SS   = {np.sum((d - d.mean())**2):.4f}")
print(f"var, divisor n   = {d.var(ddof=0):.4f}")      # Eq. 4.7-1
print(f"var, divisor n-1 = {d.var(ddof=1):.4f}")
s = d.std(ddof=1)
print(f"s = {s:.4f}   standard error = {s/np.sqrt(n):.4f}")  # 4.7-3

# Average of both variance estimators over 100,000 samples of size 5
# from a normal distribution whose true variance is 1.
rng = np.random.default_rng(0)
Z = rng.normal(0, 1, size=(100_000, 5))
print(f"mean of var(ddof=0) = {Z.var(axis=1, ddof=0).mean():.4f}")
print(f"mean of var(ddof=1) = {Z.var(axis=1, ddof=1).mean():.4f}")

# Law of large numbers: the mean of fair-coin flips.
flips = rng.integers(0, 2, size=100_000)
for m in (10, 100, 1000, 100_000):
    print(f"n = {m:>6}: mean = {flips[:m].mean():.4f}")

# Defaults differ between libraries.
import pandas as pd
from sklearn.preprocessing import StandardScaler
print("NumPy  var :", np.var(d))                     # divisor n
print("pandas var :", round(pd.Series(d).var(), 4))  # divisor n - 1
sc = StandardScaler().fit(d.reshape(-1, 1))
print("Scaler var_:", sc.var_[0])                    # divisor n
