# Problem 34C.2: the walker, Eq. 34.3-6 from the first step, and a weakly attached group.
import warnings
import numpy as np
from sklearn.semi_supervised import LabelPropagation
warnings.simplefilter('ignore')
np.set_printoptions(precision=4, suppress=True)

W = np.zeros((5, 5))
for i, w in enumerate([2, 4, 1, 1]): W[i, i + 1] = W[i + 1, i] = w
d = W.sum(1); T = W / d[:, None]; print("degrees:", d, "\nrow 3 of T:", T[2])
h = np.array([0, 2, 3, 7, 11]) / 11                       # harmonic solution of Example 34.5-1
print("h_3 from the first-step equation: %.4f * %.4f + %.4f * %.4f = %.4f   (3/11 = %.4f)"
      % (T[2, 1], h[1], T[2, 3], h[3], T[2, 1] * h[1] + T[2, 3] * h[3], 3 / 11))
rng = np.random.default_rng(1); hit, N = 0, 200000       # simulate the walker from node 3
for _ in range(N):
    i = 2
    while 0 < i < 4: i = rng.choice(5, p=T[i])
    hit += (i == 4)
print("simulated share of walkers from node 3 that reach node 5 first: %.4f (N = %d)" % (hit / N, N))

print("\n== a group of two unlabelled nodes (6, 7; link 3) joined to node 3 by a link of weight eps")
print("   eps     f_3      f_6      f_7     largest eigenvalue   n_iter (tol 1e-9)")
for eps in (1, 0.1, 0.01, 0.001):
    V = np.zeros((7, 7)); V[:5, :5] = W; V[2, 5] = V[5, 2] = eps; V[5, 6] = V[6, 5] = 3
    dv = V.sum(1); U = [1, 2, 3, 5, 6]; L = [0, 4]
    f = np.linalg.solve(np.diag(dv[U]) - V[np.ix_(U, U)], V[np.ix_(U, L)] @ np.array([0., 1.]))
    rho = max(abs(np.linalg.eigvals((V / dv[:, None])[np.ix_(U, U)])))
    y = np.array([0, -1, -1, -1, 1, -1, -1])
    m = LabelPropagation(kernel=lambda a, b: V.copy(), max_iter=10**6, tol=1e-9).fit(np.arange(7.).reshape(-1, 1), y)
    print(f"{eps:6.3f} {f[1]:8.4f} {f[3]:8.4f} {f[4]:8.4f} {rho:14.5f} {m.n_iter_:14d}")
