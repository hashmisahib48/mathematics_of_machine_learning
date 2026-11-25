# A column that copies another one.
import numpy as np
import linreg
from sklearn.linear_model import LinearRegression

X = np.array([[2, 1], [4, 2], [5, 1], [7, 3], [9, 2]], dtype=float)
y = np.array([14, 23, 24, 35, 38], dtype=float)
X3 = np.column_stack([X, 1000 * X[:, 0]])    # distance in meters

try:
    linreg.LinearRegression().fit(X3.tolist(), y.tolist())
except ValueError as err:
    print("ours   :", err)

sk = LinearRegression().fit(X3, y)
print("sklearn: rank_ =", sk.rank_, " coef_ =", sk.coef_.round(6))
p2 = LinearRegression().fit(X, y).predict(X)
print("same predictions?", np.allclose(sk.predict(X3), p2))
