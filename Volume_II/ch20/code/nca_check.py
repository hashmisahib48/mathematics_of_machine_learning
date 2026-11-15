# Neighbourhood components analysis: objective and gradient.
import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA

x = np.array([[0.0], [1.0], [2.0], [4.0]])   # four points, one feature
y = np.array([0, 0, 1, 1])
same = y[:, None] == y[None, :]

def objective(A, X):
    Z = X @ A.T                               # the embedded points
    d2 = ((Z[:, None] - Z[None]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)              # p_ii = 0
    e = np.exp(-d2)
    P = e / e.sum(axis=1, keepdims=True)      # Eq. 20.6-3
    return P, (P * same).sum()                # Eq. 20.6-4

def gradient(A, X):
    P, _ = objective(A, X)
    G = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(X)):
        for j in range(len(X)):
            if i != j:
                u = np.outer(X[i] - X[j], X[i] - X[j])
                G += (P[i] * same[i]).sum() * P[i, j] * u
                G -= same[i, j] * P[i, j] * u
    return 2 * A @ G                          # Eq. 20.6-5

np.set_printoptions(suppress=True)
A = np.array([[1.0]])
P, f = objective(A, x)
print("p_ij for A = 1 (row i, column j):\n", np.round(P, 4))
print("p_i =", np.round((P * same).sum(axis=1), 4), " f =", round(f, 4))
h = 1e-6
num = (objective(A + h, x)[1] - objective(A - h, x)[1]) / (2 * h)
print("gradient: formula", np.round(gradient(A, x), 6),
      " finite difference", np.round(num, 6))

fit = NCA(n_components=1, init="identity").fit(x, y)
a = fit.components_
print("fitted A =", np.round(a, 4), " f =", round(objective(a, x)[1], 4))

rng = np.random.default_rng(0)
X, A = rng.normal(size=(30, 3)), rng.normal(size=(2, 3))
yy = rng.integers(0, 3, 30)
same = yy[:, None] == yy[None, :]
est = NCA(n_components=2)
est.n_iter_ = 1
f_sk, g_sk = est._loss_grad_lbfgs(A.ravel(), X, same, sign=1.0)
diff = np.abs(gradient(A, X) - g_sk.reshape(2, 3)).max()
print("random data: f", round(objective(A, X)[1], 6), "vs",
      round(f_sk, 6), " max |gradient difference|", f"{diff:.1e}")
