# LinearSVC against SVC(kernel="linear"): the penalised intercept.
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC, LinearSVC

X = np.array([[1, 5], [2, 3], [3, 4], [4, 1],
              [7, 6], [8, 2], [4, 7], [6, 3]], float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
s, C = 2 * y - 1, 10.0

ref = SVC(kernel="linear", C=C, tol=1e-10).fit(X, y)
print("SVC linear   :", ref.coef_[0], ref.intercept_)
for c in (1, 100):
    m = LinearSVC(loss="hinge", C=C, intercept_scaling=c, dual=True,
                  tol=1e-10, max_iter=10**7).fit(X, y)
    print(f"hinge, c={c:3d}:", m.coef_[0].round(4),
          m.intercept_.round(4))

def J(v, c):                               # Eq. 22.10-1
    w, b = v[:2], v[2]
    z = 1 - s * (X @ w + b)
    pen = 0.5 * w @ w + 0.5 * (b / c)**2
    return pen + C * np.sum(np.maximum(0, z)**2)

for c in (1, 100):
    m = LinearSVC(C=C, intercept_scaling=c, dual=False,
                  tol=1e-10).fit(X, y)
    v = minimize(J, np.zeros(3), args=(c,), method="BFGS",
                 options={"gtol": 1e-10}).x
    got = np.r_[m.coef_[0], m.intercept_]
    print(f"squared hinge, c={c}:")
    print("   LinearSVC      ", got.round(4))
    print("   minimiser of J ", v.round(4))
d = LinearSVC().get_params()
print("defaults: dual =", d["dual"], "| loss =", d["loss"],
      "| max_iter =", d["max_iter"])
