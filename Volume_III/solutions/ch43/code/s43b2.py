# Problem 43B.2: FastICA and PCA against the noise level
import numpy as np
from sklearn.decomposition import FastICA, PCA

t = np.linspace(0, 8, 2000)
S = np.c_[np.sin(2 * t), 2 * (0.7 * t % 1) - 1]   # sine, sawtooth
S = (S - S.mean(axis=0)) / S.std(axis=0)          # unit variance
A = np.array([[1.0, 1.0], [0.5, 2.0]])            # as Listing 43.4
rng = np.random.RandomState(0)
N = rng.standard_normal(S.shape)                  # one set of draws

def worst(est):               # worst |corr| with the best source
    c = np.abs(np.corrcoef(est.T, S.T)[:2, 2:])
    return c.max(axis=1).min()

lam = np.linalg.eigvalsh(A @ A.T)   # variances on principal axes
print("eigenvalues of A A^T:", lam.round(3))
print(f"{'sigma':>6s}{'noise var':>10s}{'weak axis':>10s}"
      f"{'FastICA':>9s}{'PCA':>8s}")
for s in (0.05, 0.2, 0.5, 1.0, 2.0):
    X = S @ A.T + s * N                           # noisy mixtures
    r_ica = worst(FastICA(2, random_state=0).fit_transform(X))
    r_pca = worst(PCA(2).fit_transform(X))
    share = s**2 / (lam[0] + s**2)   # noise share, weak axis
    print(f"{s:6.2f}{s**2:10.4f}{share:10.3f}"
          f"{r_ica:9.4f}{r_pca:8.4f}")
