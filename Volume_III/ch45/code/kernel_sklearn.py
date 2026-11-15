# RBFSampler and Nystroem on the five points of Chapter 45.
import numpy as np
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.metrics.pairwise import rbf_kernel

np.set_printoptions(precision=4, suppress=True)
t = np.arange(5.0)[:, None]                 # five points
K = rbf_kernel(t, gamma=0.5)

rff = RBFSampler(gamma=0.5, n_components=4, random_state=0).fit(t)
print("omega", rff.random_weights_[0])
print("b    ", rff.random_offset_)
F = rff.transform(t)                        # Eq. 45.4-3
est = (F[0] @ F[1]).round(4)
print("estimate", est, " exact", K[0, 1].round(4))
big = RBFSampler(gamma=0.5, n_components=2000,
                 random_state=0).fit(t)
Fb = big.transform(t)
print("D = 2000: max error", abs(K - Fb @ Fb.T).max().round(4))

for seed in range(100):                     # find landmarks {1, 3}
    nys = Nystroem(gamma=0.5, n_components=2,
                   random_state=seed).fit(t)
    if set(nys.component_indices_) == {1, 3}:
        break
print("\nseed", seed, " landmarks", nys.component_indices_)
print("normalization_\n", nys.normalization_)
Z = nys.transform(t)                        # Eq. 45.4-7
print("K approx, rows 0 and 2:\n", (Z @ Z.T)[[0, 2]])
