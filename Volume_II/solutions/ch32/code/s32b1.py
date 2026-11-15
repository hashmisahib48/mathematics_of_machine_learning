# Problem 32B.1: a decreasing isotonic curve, increasing="auto"
import numpy as np
from sklearn.isotonic import IsotonicRegression

x = np.arange(1, 11.0)
y = np.array([9, 7, 8, 6, 6.5, 3, 4, 2, 1, 2.0])
q = [0, 3.5, 7.5, 11]       # outside, between, between, outside

iso = IsotonicRegression(increasing="auto").fit(x, y)   # nan outside
clip = IsotonicRegression(increasing="auto",
                          out_of_bounds="clip").fit(x, y)
print("increasing_   :", iso.increasing_)
print("X_thresholds_ :", iso.X_thresholds_)
print("y_thresholds_ :", iso.y_thresholds_)
print("query x       :", q)
print("default (nan) :", iso.predict(q).round(4))
print("clip          :", clip.predict(q).round(4))
print("fit at x      :", iso.predict(x).round(4))
print("sum y, sum f  :", y.sum(), iso.predict(x).sum())
