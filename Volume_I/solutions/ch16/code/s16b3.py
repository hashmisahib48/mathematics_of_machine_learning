# Problem 16B.3: from L2 to L1 with l1_ratio (Eq. 16.5-4).
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Z = StandardScaler().fit_transform(X)          # all 569 samples
C = 0.1
print("rho   zero  nonzero  max|C g_j| over zero weights (<= rho)")
for rho in (0, 0.25, 0.5, 0.75, 1.0):
    m = LogisticRegression(C=C, l1_ratio=rho, solver="saga",
                           max_iter=20000, tol=1e-6, random_state=0)
    m.fit(Z, y)
    w = m.coef_[0]
    p = m.predict_proba(Z)[:, 1]
    g = C * Z.T @ (p - y)                      # C X^T (p - y)
    zero = w == 0
    top = np.abs(g[zero]).max() if zero.any() else float("nan")
    print(f"{rho:4.2f} {zero.sum():5d} {(~zero).sum():8d}"
          f" {top:14.4f}")
    nz = ~zero                                 # Eq. 16.5-4, w_j != 0
    res = g[nz] + (1 - rho) * w[nz] + rho * np.sign(w[nz])
    print(f"      residual of Eq. 16.5-4: {np.abs(res).max():.1e}"
          f"   sum(p - y) = {np.sum(p - y):.1e}")
