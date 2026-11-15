# Problem 46B.2: number of edges of the graphical lasso against alpha
import sys
import numpy as np
sys.path.insert(0, "../../../ch46/code")     # returns.py of the book
from returns import true_model, draw
from sklearn.covariance import GraphicalLasso, empirical_covariance
from sklearn.preprocessing import StandardScaler

Sigma, Theta, sd = true_model()
p = len(Sigma)
off = ~np.eye(p, dtype=bool)
true = (np.abs(Theta) > 1e-12) & off         # the 18 true edges
Z = StandardScaler().fit_transform(draw(150, seed=1))
S = empirical_covariance(Z)
amax = np.abs(S[off]).max()                  # Eq. 46.5-3
print("alpha_max = %.4f; true edges %d" % (amax, true.sum() // 2))
print("alpha/alpha_max  alpha  edges  true  false  n_iter")
for f in (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
    g = GraphicalLasso(alpha=f * amax, enet_tol=1e-6).fit(Z)
    E = (np.abs(g.precision_) > 1e-10) & off
    print(f"{f:10.1f} {f * amax:10.3f} {E.sum() // 2:6d}"
          f" {(E & true).sum() // 2:5d} {(E & ~true).sum() // 2:6d}"
          f" {g.n_iter_:7d}")
