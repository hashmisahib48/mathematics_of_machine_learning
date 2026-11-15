# Problem 21B.3: adding kernels, Eq. 21.3-9
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

rng = np.random.default_rng(3)
x = np.sort(rng.uniform(0, 6, 30))
y = np.sin(x) + 0.5 * x + 0.1 * rng.normal(size=30)
xe = np.linspace(0, 8, 200)                    # grid: 0 to 8
truth = np.sin(xe) + 0.5 * xe
inside = xe <= 6
alpha = 0.1

rbf = lambda P, Q: rbf_kernel(P, Q, gamma=1.0)
lin = lambda P, Q: P @ Q.T
kernels = {"rbf": rbf, "linear": lin,
           "rbf + linear": lambda P, Q: rbf(P, Q) + lin(P, Q)}
X, Xe = x[:, None], xe[:, None]
rm = lambda e: np.sqrt(np.mean(e ** 2))
for name, k in kernels.items():
    G = k(X, X) + alpha * np.eye(30)
    a = np.linalg.solve(G, y)                      # Eq. 21.4-3
    f = k(Xe, X) @ a                               # Eq. 21.4-4
    err = f - truth
    print(f"{name:13s} inside {rm(err[inside]):.3f}"
          f"  beyond {rm(err[~inside]):.3f}"
          f"  f(8) = {f[-1]:.3f} (truth {truth[-1]:.3f})")
    if name != "rbf":                    # the linear part: slope
        slope = a @ x                    # w = sum a_i x_i
        e = (0.5 - slope) * xe[~inside]
        print(f"{'':13s} slope {slope:.3f} (true 0.5):"
              f" rms error {rm(e):.3f} beyond 6")
print("sin x alone beyond 6, rms: %.3f" % rm(np.sin(xe[~inside])))
