# Kernel PCA on two circles: gamma, centring by hand, the pre-image.
import numpy as np
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KernelCenterer

X, y = make_circles(n_samples=300, factor=0.3, noise=0.05,
                    random_state=0)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.33, random_state=0, stratify=y)
print("gamma  top 2 eigenvalues  accuracy, first component")
for g in (0.5, 2, 10, 50):
    kp = KernelPCA(2, kernel="rbf", gamma=g, random_state=0)
    kp.fit(Xtr)
    clf = LogisticRegression().fit(kp.transform(Xtr)[:, :1], ytr)
    acc = clf.score(kp.transform(Xte)[:, :1], yte)
    print(f"{g:5}  {kp.eigenvalues_.round(2)}        {acc:.3f}")

kp = KernelPCA(2, kernel="rbf", gamma=2, random_state=0).fit(Xtr)
K = KernelCenterer().fit_transform(rbf_kernel(Xtr, gamma=2))
lam, A = np.linalg.eigh(K)                    # Eq. 42.11-2
lam, A = lam[::-1][:2], A[:, ::-1][:, :2]
S = A * np.sqrt(lam)                          # scores of the data
print("by hand = KernelPCA:",
      np.allclose(np.abs(S), np.abs(kp.transform(Xtr))))

print("pre-image: n_components, alpha, RMS error on test points")
for k, a in ((2, 0.01), (5, 0.01), (5, 0.1), (5, 1.0)):
    kp = KernelPCA(k, kernel="rbf", gamma=2, alpha=a,
                   fit_inverse_transform=True).fit(Xtr)
    R = kp.inverse_transform(kp.transform(Xte))
    rms = np.sqrt(((R - Xte) ** 2).sum(axis=1).mean())
    print(f"{k:12d} {a:6} {rms:.4f}")
