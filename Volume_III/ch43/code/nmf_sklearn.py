# NMF on the digits: solvers, losses, initialisations
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xtr, Xte = train_test_split(X, test_size=0.25, random_state=0,
                            stratify=y)[:2]
runs = [("cd, Frobenius", dict()),
        ("cd, nndsvd start", dict(init="nndsvd")),
        ("cd, random start", dict(init="random")),
        ("mu, Frobenius", dict(solver="mu")),
        ("mu, KL", dict(solver="mu", beta_loss="kullback-leibler"))]
fro = lambda A, B: np.linalg.norm(A - B)       # Frobenius norm
print(f"{'k = 10':18s}{'iter':>5s}{'err_':>8s}{'train':>8s}"
      f"{'test':>8s}{'H=0':>7s}{'W=0':>7s}")
for name, kw in runs:
    m = NMF(10, max_iter=1000, random_state=0, **kw)
    W = m.fit_transform(Xtr)                   # scores, n x k
    H = m.components_                          # parts, k x p
    Wt = m.transform(Xte)
    print(f"{name:18s}{m.n_iter_:5d}{m.reconstruction_err_:8.1f}"
          f"{fro(Xtr, W @ H):8.1f}{fro(Xte, Wt @ H):8.1f}"
          f"{(H == 0).mean():7.3f}{(W == 0).mean():7.3f}")
p = PCA(10).fit(Xtr)                          # same k, contrast
rec = lambda A: p.inverse_transform(p.transform(A))
print(f"{'PCA, 10 components':18s}{'':13s}{fro(Xtr, rec(Xtr)):8.1f}"
      f"{fro(Xte, rec(Xte)):8.1f}")
