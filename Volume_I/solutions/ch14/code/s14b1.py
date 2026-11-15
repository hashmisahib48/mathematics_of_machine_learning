# Problem 14B.1: sample weights against repeated rows
import numpy as np
from sklearn.linear_model import PoissonRegressor

X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
y = np.array([1.0, 2.0, 2.0, 4.0, 6.0])          # the crack counts

def fit(X, y, w=None):
    m = PoissonRegressor(alpha=0.3, tol=1e-12, max_iter=500)
    return m.fit(X, y, sample_weight=w)

rows = [0, 1, 2, 2, 3, 4]                         # row 2 listed twice
fits = {
    "weights 1,1,2,1,1": fit(X, y, [1, 1, 2, 1, 1]),
    "row 3 listed twice": fit(X[rows], y[rows]),
    "weights 2,2,4,2,2": fit(X, y, [2, 2, 4, 2, 2]),
    "no weights": fit(X, y),
}
print("fit                     coef_      intercept_")
for name, m in fits.items():
    print(f"{name:20s} {m.coef_[0]:10.6f} {m.intercept_:10.6f}")

# the data term of Eq. 14.3-9 divides by the sum of the weights
for name, s in (("weights 1,1,2,1,1", 6), ("weights 2,2,4,2,2", 12)):
    print(f"{name}: sum of weights {s}, 1/(2*sum) = {1/(2*s):.4f}")
