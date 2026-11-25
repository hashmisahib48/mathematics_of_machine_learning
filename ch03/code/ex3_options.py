# Options: fit_intercept, sample_weight and positive
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]], dtype=float)
y = np.array([14, 23, 24, 35, 38], dtype=float)

m1 = LinearRegression(fit_intercept=False).fit(X, y)
print("fit_intercept=False :", m1.coef_, m1.intercept_)

w = np.array([1, 1, 2, 1, 1], dtype=float)   # third trip twice
m2 = LinearRegression().fit(X, y, sample_weight=w)
print("sample_weight       :", m2.coef_, m2.intercept_)

y_neg = np.array([14, 10, 24, 12, 38], dtype=float)
m3 = LinearRegression().fit(X, y_neg)
m4 = LinearRegression(positive=True).fit(X, y_neg)
print("positive=False      :", m3.coef_, m3.intercept_)
print("positive=True       :", m4.coef_, m4.intercept_)
