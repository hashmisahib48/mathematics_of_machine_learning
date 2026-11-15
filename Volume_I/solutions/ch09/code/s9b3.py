# Problem 9B.3: a duplicated feature
import numpy as np
from sklearn.linear_model import LinearRegression
from ols_scratch import MyOLS   # the class of Listing 9.2

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
X = np.c_[x, x]                        # two identical columns
Xc = X - X.mean(axis=0)
print("X'X (centred) =", (Xc.T @ Xc).tolist())
sv = np.linalg.svd(Xc, compute_uv=False)
print("singular values of centred X:", sv)

for method in ("normal", "qr", "svd"):
    try:
        m = MyOLS(method).fit(X, y)
        print(f"MyOLS({method!r:8s}): w =", np.round(m.coef_, 4),
              " b =", round(m.intercept_, 4))
    except np.linalg.LinAlgError as err:
        print(f"MyOLS({method!r:8s}): LinAlgError:", err)

lr = LinearRegression().fit(X, y)
print("LinearRegression  : w =", np.round(lr.coef_, 4),
      " b =", round(lr.intercept_, 4), " rank_ =", lr.rank_)
print("sum of the weights:", round(lr.coef_.sum(), 4),
      " (one-feature slope of Chapter 9: 1.7)")
print("predictions equal to the one-feature fit:",
      np.allclose(lr.predict(X), 0.9 + 1.7 * x))
