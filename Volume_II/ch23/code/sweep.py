# Examples 23.5-2 and 23.5-3: delivery data, linear SVR.
import numpy as np
from sklearn.svm import SVR

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])


def fit(C, eps, yy=y):
    m = SVR(kernel="linear", C=C, epsilon=eps, tol=1e-10).fit(X, yy)
    r = yy - m.predict(X)
    bound = int((np.abs(m.dual_coef_) > C * (1 - 1e-6)).sum())
    out = int((np.abs(r) > eps + 1e-7).sum())
    return (m.coef_[0, 0], m.intercept_[0], len(m.support_), bound,
            out, m.predict([[6.0]])[0])


print("epsilon sweep, C = 10")
print("eps     w       b    SV  at C  outside  f(6)")
for eps in (0, 0.5, 1, 1.5, 2, 3, 3.5, 4):
    w, b, n, nb, no, p = fit(10, eps)
    print(f"{eps:3.1f} {w:7.4f} {b:7.4f} {n:4d} {nb:4d} "
          f"{no:6d} {p:8.3f}")

print("C sweep, epsilon = 1")
print("   C       w       b    SV  at C  f(6)")
for C in (0.05, 0.1, 0.2, 0.3, 0.4, 1, 10):
    w, b, n, nb, no, p = fit(C, 1.0)
    print(f"{C:5.2f} {w:7.4f} {b:7.4f} {n:4d} {nb:4d} {p:8.3f}")

y2 = y.copy()
y2[4] = 20.0                    # the wrong entry of Chapter 13
print("wrong entry y_5 = 20, epsilon = 1")
for C in (0.3, 1, 10):
    w, b, n, nb, no, p = fit(C, 1.0, y2)
    print(f"C = {C:4.1f}: w = {w:.4f} b = {b:.4f}  f(6) = {p:.3f}")
A = np.c_[np.ones(5), X[:, 0]]
b0, w0 = np.linalg.lstsq(A, y2, rcond=None)[0]
print(f"least squares: w = {w0:.4f} b = {b0:.4f} "
      f"f(6) = {b0 + 6 * w0:.3f}")
