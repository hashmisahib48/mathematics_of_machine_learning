# Problem 30B.1: posterior samples by Cholesky (run from code/).
import sys
sys.path.insert(0, '../../../ch30/code')
import numpy as np
from gp_scratch import MyGP              # Listing 30.2

X = np.array([1., 2, 3, 5, 6, 8])[:, None]
y = np.array([0.2, 1.1, 1.5, 0.4, -0.6, 0.3])
gp = MyGP(sf2=0.7366, ell=1.3672, noise=0.01).fit(X, y)
grid = np.linspace(0, 10, 120)[:, None]
mean, cov = gp.predict(grid, return_cov=True)     # Eq. 30.3-9
sd = np.sqrt(np.diag(cov))
L = np.linalg.cholesky(cov + 1e-8 * np.eye(120))  # cov = L L^T

def errors(n, seed):
    Z = np.random.default_rng(seed).normal(size=(120, n))
    S = mean[:, None] + L @ Z                     # columns = samples
    return (np.abs(S.mean(axis=1) - mean).max(),
            np.abs(S.std(axis=1) - sd).max())

for n in (2000, 20000):                           # seed 0: one run
    em, es = errors(n, 0)
    print(f"{n:6d} draws: mean error {em:.4f}  std error {es:.4f}")
for n in (2000, 20000):                           # average of 30 runs
    r = np.mean([errors(n, s) for s in range(30)], axis=0)
    print(f"{n:6d} draws, 30 seeds: mean error {r[0]:.4f}"
          f"  std error {r[1]:.4f}")
w = np.linalg.eigvalsh(cov)
print(f"smallest eigenvalue {w.min():.2e}; eigenvalues < 1e-12:",
      int((w < 1e-12).sum()))
try:
    np.linalg.cholesky(cov)
    print("plain Cholesky: succeeded")
except np.linalg.LinAlgError:
    print("plain Cholesky: LinAlgError, jitter needed")
