# The ten lines through two of the five points (y5 = 20).
from itertools import combinations
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 4.0, 6.0, 7.0, 20.0])
t = np.median(np.abs(y - np.median(y)))    # RANSAC default
print("threshold t =", t)

slopes = []
print("pair     b       w     sum|r|  inliers")
for i, j in combinations(range(5), 2):
    w = (y[j] - y[i]) / (x[j] - x[i])      # Eq. 13.4-3
    b = y[i] - w * x[i]
    r = y - b - w * x
    slopes.append(w)
    print(f"({i+1},{j+1}) {b:7.3f} {w:7.3f} {np.abs(r).sum():7.2f}"
          f"  {np.sum(np.abs(r) <= t)}")

w_ts = np.median(slopes)                   # Theil-Sen slope
b_ts = np.median(y - w_ts * x)
print("sorted slopes:", np.round(np.sort(slopes), 3))
print(f"Theil-Sen: w = {w_ts:.3f}, b = {b_ts:.3f}")
