# Numerical checks of statements made in Sections 22.3 to 22.6.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Z, s = StandardScaler().fit_transform(X), 2 * y - 1
C = 1.0
m = SVC(kernel="linear", C=C, tol=1e-10).fit(Z, y)
a = np.abs(m.dual_coef_[0])                      # alpha at the SVs
S = Z[m.support_]
t = m.dual_coef_[0]                              # alpha_i y_i
w, b = m.coef_[0], m.intercept_[0]
W = a.sum() - 0.5 * t @ (S @ S.T) @ t            # Eq. 22.4-6
hinge = lambda w, b: np.maximum(0, 1 - s * (Z @ w + b)).sum()
J = 0.5 * w @ w + C * hinge(w, b)                # Eq. 22.4-3
print(f"dual W = {W:.6f}   primal J = {J:.6f}   gap = {J - W:.1e}")
print("sum alpha_i y_i =", f"{t.sum():.1e}",
      "  w = sum alpha_i y_i x_i:", np.allclose(w, t @ S))
rng = np.random.default_rng(0)
worse = sum(0.5 * (w + d[:30]) @ (w + d[:30]) + C * hinge(
    w + d[:30], b + d[30]) >= J - 1e-9
            for d in 1e-3 * rng.normal(size=(2000, 31)))
print("perturbations of (w, b) that raise J:", worse, "of 2000")
print("alpha = C for", int((a > C - 1e-8).sum()), "of", len(a),
      "support vectors;  free:", int((a < C - 1e-8).sum()))

# slope of W along the SMO line, Eq. 22.6-2 (finite differences)
Xs = np.array([[1, 5], [2, 3], [3, 4], [4, 1],
               [7, 6], [8, 2], [4, 7], [6, 3]], float)
ys = np.array([-1, -1, -1, -1, 1, 1, 1, 1.])
K = Xs @ Xs.T
Wf = lambda al: al.sum() - 0.5 * (al * ys) @ K @ (al * ys)
al = np.array([0, 0, .3, .1, .1, .1, .05, .15])   # sum a y = 0
i, j, h = 4, 2, 1e-6
f = K @ (al * ys)
E = f - ys                       # b drops out of E_i - E_j
def along(u):
    q = al.copy()
    q[j] += u
    q[i] += ys[i] * ys[j] * (-u)     # keeps sum a y fixed
    return Wf(q)
slope = (along(h) - along(-h)) / (2 * h)
curv = (along(h) - 2 * along(0) + along(-h)) / h**2
eta = K[i, i] + K[j, j] - 2 * K[i, j]
pred = ys[j] * (E[i] - E[j])
print(f"slope {slope:.6f} vs y_j(E_i - E_j) = {pred:.6f}")
print(f"curvature {curv:.4f} vs -eta = {-eta:.4f}")

for name, A in (("raw", X), ("scaled", Z)):
    g = 1 / (A.shape[1] * A.var())
    print(f"{name:7s} Var(X) = {A.var():10.4f}  gamma 'scale' = "
          f"{g:.3e}  'auto' = {1 / A.shape[1]:.4f}")
    P = (A**2).sum(1)[:, None] + (A**2).sum(1) - 2 * A @ A.T
    print(f"         mean of gamma ||x - z||^2 over pairs: "
          f"{g * P.mean():.3f}")
