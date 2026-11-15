# Six flats as a table: X holds the features, y the prices.
import numpy as np
X = np.array([[4.0, 1.0],      # area (10 m^2), rooms
              [5.0, 2.0],
              [6.0, 2.0],
              [7.0, 3.0],
              [8.0, 3.0],
              [9.0, 4.0]])
y = np.array([95.0, 120.0, 130.0, 160.0, 170.0, 200.0])  # thousands
n, p = X.shape                  # n samples (rows), p features
print("n =", n, "  p =", p, "  y shape =", y.shape)
print("x_32 =", X[2, 1], "  (Python counts from 0)")
w = np.array([15.0, 10.0])      # two weights, chosen by eye
b = 20.0                        # intercept
y_hat = X @ w + b               # Eq. 1.3-3: rows times w, plus b
r = y - y_hat                   # residuals, Eq. 1.4-1
print("y_hat =", y_hat)
print("r     =", r)
print("sum of squared residuals =", np.sum(r ** 2))
print("mean squared loss, Eq. 1.4-3 =", round(np.mean(r ** 2), 2))
