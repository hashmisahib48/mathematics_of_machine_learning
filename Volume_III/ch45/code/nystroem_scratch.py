# The Nystroem feature map for the RBF kernel, from scratch.
import numpy as np
from sklearn.datasets import load_digits
from sklearn.kernel_approximation import Nystroem

def rbf(A, B, g):                           # Eq. 21.3-7 (Volume II)
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None] - 2 * A @ B.T
    return np.exp(-g * np.maximum(d2, 0))

def fit(X, g, m, seed):
    idx = np.random.RandomState(seed).permutation(len(X))[:m]
    L = X[idx]                              # the landmarks
    U, s, Vt = np.linalg.svd(rbf(L, L, g))  # W = U diag(s) V^T
    s = np.clip(s, 1e-12, None)             # as the library does
    return idx, L, (U / np.sqrt(s)) @ Vt    # W^(-1/2), Eq. 45.4-7

def transform(X, L, g, Wih):
    return rbf(X, L, g) @ Wih.T

# the test: the same landmarks and features as scikit-learn
X = load_digits(return_X_y=True)[0][:600] / 16
idx, L, Wih = fit(X, 0.11, 60, 0)
ref = Nystroem(gamma=0.11, n_components=60, random_state=0).fit(X)
assert (idx == ref.component_indices_).all()
Zm, Zr = transform(X, L, 0.11, Wih), ref.transform(X)
assert np.allclose(Zm, Zr)
K = rbf(X, X, 0.11)
lam = np.linalg.eigvalsh(K - Zm @ Zm.T)
print("features agree; max |dZ| =", np.abs(Zm - Zr).max())
print("relative error of the Gram matrix:",
      (np.linalg.norm(K - Zm @ Zm.T) / np.linalg.norm(K)).round(4))
print("error is PSD:", lam.min() > -1e-9)
