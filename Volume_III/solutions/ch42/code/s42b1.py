# Problem 42B.1: PCA of the digits by eigh on the covariance matrix
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

Xc = Xtr - Xtr.mean(axis=0)                  # centred data
S = Xc.T @ Xc / (len(Xtr) - 1)               # Eq. 42.3-1
lam, V = np.linalg.eigh(S)                   # ascending order
lam, V = lam[::-1], V[:, ::-1]               # largest first
Z = Xc @ V[:, :5]                            # scores, no sign rule

ref = PCA(5, svd_solver="full").fit(Xtr)
Zr = ref.transform(Xtr)
print("max |eigenvalue difference|:",
      f"{np.abs(lam[:5] - ref.explained_variance_).max():.1e}")
print("max |score difference|, raw:",
      f"{np.abs(Z - Zr).max():.3f}")
sign = np.sign((Z * Zr).sum(axis=0))         # +1 same, -1 opposite
print("sign of each column relative to PCA:", sign.astype(int))
print("max |score difference| after the flip:",
      f"{np.abs(Z * sign - Zr).max():.1e}")

# the rule of svd_flip: largest entry of each component positive
W = V[:, :5].T                               # rows are components
big = np.abs(W).argmax(axis=1)
W = W * np.sign(W[np.arange(5), big])[:, None]
print("max difference with the sign rule applied:",
      f"{np.abs(Xc @ W.T - Zr).max():.1e}")
