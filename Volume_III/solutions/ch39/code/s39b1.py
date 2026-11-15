# Problem 39B.1: the scale gamma of the rbf affinity on the two rings
import sys, os, warnings
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here + '/../../../ch39/code')
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel
from data import rings

X, y = rings()           # make_circles(300, factor=0.4, noise=0.05)
same = y[:, None] == y[None]
print("gamma    ARI pieces  weakest   leak/within"
      "     lam2      lam3")
for g in [0.3, 1, 3, 10, 30, 100, 300, 1000]:
    sc = SpectralClustering(2, gamma=g, random_state=0)
    lab = sc.fit_predict(X)
    W = rbf_kernel(X, gamma=g)
    np.fill_diagonal(W, 0)
    # pieces of the graph that keeps the weights above 1e-8
    pieces = connected_components(W > 1e-8)[0]
    weakest = W.max(axis=1).min()   # best link of the worst point
    # mean weight between the rings / mean non-zero weight within one
    leak = W[~same].mean() / W[same & (W > 0)].mean()
    d = W.sum(1)
    Ls = np.eye(300) - W / np.sqrt(np.outer(d, d))
    lam = np.abs(eigh(Ls, eigvals_only=True, subset_by_index=[0, 2]))
    print(f"{g:5g} {adjusted_rand_score(y, lab):6.3f} {pieces:5d}"
          f" {weakest:9.2e} {leak:9.2e} {lam[1]:9.2e} {lam[2]:9.2e}")
