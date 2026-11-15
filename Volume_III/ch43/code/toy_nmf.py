# One multiplicative-update sweep of NMF on three paints
import numpy as np
from sklearn.decomposition import NMF

X = np.array([[4., 2.], [3., 3.], [2., 4.]])     # three paints
W0 = np.array([[2., 1.], [1., 1.], [1., 2.]])    # starting amounts
H0 = np.array([[1., 1.], [1., 1.]])              # starting pigments
err = lambda W, H: np.linalg.norm(X - W @ H)     # Frobenius norm

print("start: error", err(W0, H0).round(4))
W1 = W0 * (X @ H0.T) / (W0 @ H0 @ H0.T)    # W first, as in sklearn
print("W after update\n", W1.round(4))
H1 = H0 * (W1.T @ X) / (W1.T @ W1 @ H0)    # then H, new W
print("H after update\n", H1.round(4))
print("after one sweep: error", err(W1, H1).round(4))

# the same sweep inside scikit-learn (solver 'mu', one iteration)
m = NMF(2, solver="mu", init="custom", max_iter=1, tol=0.0)
Ws = m.fit_transform(X, W=W0.copy(), H=H0.copy())
print("agrees with NMF(max_iter=1):", np.allclose(Ws, W1),
      np.allclose(m.components_, H1))
# the exact factors are a fixed point of the update
Wt = np.array([[2., 0.], [1., 1.], [0., 2.]])    # true amounts
Ht = np.array([[2., 1.], [1., 2.]])              # true pigments
print("ratio X H^T/(W H H^T) at the truth is all ones:",
      np.allclose((X @ Ht.T) / (Wt @ Ht @ Ht.T), 1.0))
