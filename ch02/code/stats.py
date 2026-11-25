# Mean, variance, covariance and correlation, by formula and by NumPy.
import numpy as np

x = [1, 2, 3, 4, 5]                      # hours studied
y = [52, 57, 61, 68, 72]                 # test scores
n = len(x)
xm, ym = sum(x) / n, sum(y) / n
sxx = sum((a - xm) ** 2 for a in x)
syy = sum((b - ym) ** 2 for b in y)
sxy = sum((a - xm) * (b - ym) for a, b in zip(x, y))
print("means:", xm, ym)
print("variances:", sxx / (n - 1), syy / (n - 1))
print("covariance:", sxy / (n - 1))
print("correlation:", round(sxy / (sxx * syy) ** 0.5, 4))
print("NumPy r:", round(np.corrcoef(x, y)[0, 1], 4))
