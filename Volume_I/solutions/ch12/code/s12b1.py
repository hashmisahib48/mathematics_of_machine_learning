# Problem 12B.1: more data, narrower band
import numpy as np
from sklearn.linear_model import BayesianRidge

x = np.array([-3., -1, 1, 3]); y = np.array([-2., -2, 1, 3])
print("copies  alpha_ lambda_  sd(w)  sd(y*)  36 sd(w)^2  1/alpha_")
for k in (1, 4, 16):
    X = np.tile(x, k)[:, None]                 # k copies of the data
    m = BayesianRidge().fit(X, np.tile(y, k))
    _, sd = m.predict([[6.0]], return_std=True)
    s2 = m.sigma_[0, 0]                        # variance of the slope
    print(f"{k:6d} {m.alpha_:8.3f} {m.lambda_:7.3f} {s2 ** 0.5:7.4f}"
          f" {sd[0]:7.3f} {36 * s2:10.4f} {1 / m.alpha_:9.4f}")
r = y - (x @ y) / (x @ x) * x              # LS residuals
print("limit of alpha_: n / SSE =", round(4 / (r @ r), 4),
      "  floor 1/sqrt(alpha_) =", round((r @ r / 4) ** 0.5, 4))
w = (x @ y) / (x @ x)                          # LS slope, 0.9
print("limit of lambda_: 1 / slope^2 =", round(1 / w ** 2, 4))
