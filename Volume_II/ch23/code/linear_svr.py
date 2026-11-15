# LinearSVR penalises the intercept, SVR(kernel="linear") does not.
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVR, LinearSVR

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
C, eps = 10.0, 1.0


def objective(p, s=1.0, sq=False):       # Eq. 23.4-3, p = (w, b)
    w, b = p                             # b = s * (weight of s)
    r = np.maximum(0, np.abs(y - w * X[:, 0] - b) - eps)
    loss = (r**2).sum() if sq else r.sum()
    return 0.5 * (w**2 + (b / s)**2) + C * loss


def show(name, w, b):
    print(f"{name:34s} w = {w:.4f}  b = {b:.4f}")


m = SVR(kernel="linear", C=C, epsilon=eps, tol=1e-10).fit(X, y)
show("SVR(kernel='linear')", m.coef_[0, 0], m.intercept_[0])
for s in (1.0, 10.0):
    m = LinearSVR(C=C, epsilon=eps, intercept_scaling=s, tol=1e-10,
                  max_iter=100000).fit(X, y)
    show(f"LinearSVR, intercept_scaling={s:g}", m.coef_[0],
         m.intercept_[0])
    q = minimize(objective, [1.0, 1.0], args=(s,),
                 method="Nelder-Mead", options={"xatol": 1e-9,
                                                "fatol": 1e-12})
    show("   minimiser of Eq. 23.4-3", *q.x)
m = LinearSVR(C=C, epsilon=eps, loss="squared_epsilon_insensitive",
              dual=False, tol=1e-10).fit(X, y)
show("LinearSVR, squared loss", m.coef_[0], m.intercept_[0])
q = minimize(objective, [1.0, 1.0], args=(1.0, True), method="BFGS")
show("   minimiser", *q.x)
d = LinearSVR()
print("defaults:", d.epsilon, d.loss, d.dual, d.tol, d.max_iter)
