# Problem 9B.1: a score method for MyOLS (R^2 of Eq. 9.3-6)
import numpy as np
from sklearn.linear_model import LinearRegression
from ols_scratch import MyOLS          # the class of Listing 9.2


class MyOLSScore(MyOLS):
    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        sse = np.sum((y - self.predict(X)) ** 2)   # squared residuals
        sst = np.sum((y - y.mean()) ** 2)          # about the mean y
        return 1.0 - sse / sst


X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
ref = LinearRegression().fit(X, y).score(X, y)
print(f"LinearRegression.score : {ref:.6f}")
for method in ("normal", "qr", "svd"):
    r2 = MyOLSScore(method).fit(X, y).score(X, y)
    print(f"MyOLS({method!r:8s}).score : {r2:.6f}"
          f"   equal: {np.isclose(r2, ref)}")

# on new data R^2 can be negative
m = MyOLSScore("svd").fit(X, y)
Xn = np.array([[6.0], [7.0], [8.0]])
lr = LinearRegression().fit(X, y)
for name, shift in (("on the line", 0.0), ("shifted by +10", 10.0)):
    yn = 0.9 + 1.7 * Xn[:, 0] + shift
    print(f"new data {name:14s}: R2 = {m.score(Xn, yn):8.4f}"
          f"   sklearn: {lr.score(Xn, yn):8.4f}")
