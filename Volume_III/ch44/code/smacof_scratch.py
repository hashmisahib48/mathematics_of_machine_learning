# Metric SMACOF (stress majorisation) from scratch.
import numpy as np
from scipy.spatial.distance import pdist, squareform


def my_smacof(D, Z0, max_iter=300, eps=1e-6):
    n = len(D)
    Z, old = Z0.copy(), None
    for it in range(max_iter):
        E = squareform(pdist(Z))                # current distances
        E[E == 0] = 1e-5
        R = D / E
        B = -R                                  # Eq. 44.4-4
        B[np.diag_indices(n)] += R.sum(axis=1)
        Z = B @ Z / n                           # Eq. 44.4-5
        E = squareform(pdist(Z))
        stress = ((E - D) ** 2).sum() / 2       # Eq. 44.4-1
        gap = (old - stress) / ((E ** 2).sum() / 2) if old else 1
        if old is not None and gap < eps:
            break
        old = stress
    return Z, stress, it + 1


# --- the test: the same start, the same stopping rule
from sklearn.datasets import load_digits
from sklearn.manifold import smacof

X = load_digits().data[:150] / 16
D = squareform(pdist(X))
Z0 = np.random.RandomState(0).uniform(size=(150, 2))
Z, s, it = my_smacof(D, Z0)
Zr, sr, itr = smacof(D, metric=True, n_components=2, init=Z0,
                     n_init=1, max_iter=300, eps=1e-6,
                     return_n_iter=True, normalized_stress=False)
print(f"mine: stress {s:.4f} after {it} iterations")
print(f"scikit-learn: stress {sr:.4f} after {itr} iterations")
print("max |dZ| =", f"{np.abs(Z - Zr).max():.1e}")
assert it == itr and np.allclose(Z, Zr)
