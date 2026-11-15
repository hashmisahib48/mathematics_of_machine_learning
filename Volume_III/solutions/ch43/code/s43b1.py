# Problem 43B.1: NMF test error and sparsity against k
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
Xtr, Xte = train_test_split(X, test_size=0.25, random_state=0,
                            stratify=y)[:2]      # split of 43.10
fro = lambda A, B: np.linalg.norm(A - B)         # Frobenius norm

print(f"{'k':>3s}{'train':>8s}{'test':>8s}{'drop':>7s}"
      f"{'test/PCA':>10s}{'H=0':>7s}{'W=0':>7s}{'iter':>6s}")
last = None
for k in (5, 10, 15, 20):
    nmf = NMF(k, max_iter=1000, random_state=0)
    W = nmf.fit_transform(Xtr)                   # codes of training
    H = nmf.components_                          # parts, k x 64
    Wt = nmf.transform(Xte)                      # codes, H fixed
    pca = PCA(k).fit(Xtr)                        # same k, no signs
    e_pca = fro(Xte, pca.inverse_transform(pca.transform(Xte)))
    e_te = fro(Xte, Wt @ H)
    drop = "" if last is None else f"{last - e_te:7.1f}"
    last = e_te                                  # gain of 5 parts
    print(f"{k:3d}{fro(Xtr, W @ H):8.1f}{e_te:8.1f}{drop:>7s}"
          f"{e_te / e_pca:10.3f}{(H == 0).mean():7.3f}"
          f"{(W == 0).mean():7.3f}{nmf.n_iter_:6d}")
