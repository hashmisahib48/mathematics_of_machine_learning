# Problem 34B.1: tolerance and error of LabelPropagation (chain).
import numpy as np
from sklearn.semi_supervised import LabelPropagation

W = np.zeros((5, 5))                 # weights of the chain, Sec. 34.1
for i, w in enumerate([2, 4, 1, 1]):
    W[i, i + 1] = W[i + 1, i] = w
x = np.arange(5.0).reshape(-1, 1)    # dummy features, not used
y = np.array([0, -1, -1, -1, 1])     # nodes 1 and 5 are labelled
exact = np.array([0, 2, 3, 7, 11]) / 11   # Eq. 34.3-6, class 1

T = W / W.sum(axis=1)[:, None]       # Eq. 34.3-3
U = [1, 2, 3]                        # the unlabelled nodes
rho = max(abs(np.linalg.eigvals(T[np.ix_(U, U)])))
print(f"largest eigenvalue of T_UU = {rho:.4f}")
print(f"iterations per factor 10 = {np.log(10) / -np.log(rho):.2f}")

print("     tol  n_iter  max error  error/tol  factor")
prev = None
for tol in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9, 1e-12]:
    m = LabelPropagation(kernel=lambda a, b: W.copy(),  # a copy
                         max_iter=100000, tol=tol).fit(x, y)
    err = abs(m.label_distributions_[:, 1] - exact).max()
    rate = "     -"
    if prev is not None:            # error ratio per iteration
        k = m.n_iter_ - prev[0]
        rate = f"{(err / prev[1]) ** (1 / k):.4f}"
    print(f"{tol:8.0e}  {m.n_iter_:5d}  {err:9.2e}  "
          f"{err / tol:9.3f}  {rate}")
    prev = (m.n_iter_, err)
