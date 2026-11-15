# IncrementalPCA in batches, and TruncatedSVD without centring.
import numpy as np
from scipy import sparse
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD

X, y = load_digits(return_X_y=True)
n = len(X)
pca = PCA(10, svd_solver="full").fit(X)
for bs in (100, 400, n):
    ip = IncrementalPCA(10, batch_size=bs).fit(X)
    cos = np.abs((ip.components_ * pca.components_).sum(axis=1))
    v = ip.explained_variance_ / pca.explained_variance_
    rel = np.abs(v - 1)
    print(f"batch {bs:4d}: min |cos| {cos.min():.4f}, "
          f"max variance error {rel.max():.1e}")
ip = IncrementalPCA(10)                       # a stream of 9 chunks
for chunk in np.array_split(X, 9):
    ip.partial_fit(chunk)
r1 = ip.explained_variance_ratio_.sum()
r2 = pca.explained_variance_ratio_.sum()
print("partial_fit: seen", int(ip.n_samples_seen_),
      "samples; ratio sum", r1.round(4), " PCA", r2.round(4))

svd = TruncatedSVD(2, algorithm="arpack").fit(X)
m = X.mean(axis=0) / np.linalg.norm(X.mean(axis=0))
print("TruncatedSVD: |cos(component 1, mean)| =",
      abs(svd.components_[0] @ m).round(5))
print("  ratio", svd.explained_variance_ratio_.round(4),
      " PCA", PCA(2).fit(X).explained_variance_ratio_.round(4))
Xc = X - X.mean(axis=0)                       # centred first: PCA
c = TruncatedSVD(10, algorithm="arpack").fit(Xc)
same = np.allclose(np.abs(c.components_), np.abs(pca.components_))
q = c.explained_variance_[0] / pca.explained_variance_[0]
print("centred: components agree", same)
print("  variance ratio to PCA", round(q, 6),
      " (n-1)/n =", round((n - 1) / n, 6))
sp = TruncatedSVD(2, algorithm="arpack").fit(sparse.csr_matrix(X))
print("sparse input = dense input:",
      np.allclose(np.abs(sp.components_), np.abs(svd.components_)))
