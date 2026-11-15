# Eight seeds: how the soft margin changes with C (Example 22.7-2).
import numpy as np
from sklearn.svm import SVC

X = np.array([[1, 5], [2, 3], [3, 4], [4, 1],
              [7, 6], [8, 2], [4, 7], [6, 3]], float)
y = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

print("   C   n_SV  at bound  inside  |w|   margin")
for C in (0.1, 0.2, 0.4, 1.0):
    m = SVC(kernel="linear", C=C, tol=1e-10).fit(X, y)
    z = y * m.decision_function(X)
    ab = np.abs(m.dual_coef_[0])
    nw = np.linalg.norm(m.coef_)
    nb = (ab > C - 1e-9).sum()
    print(f"{C:5.1f} {len(m.support_):5d} {nb:8d}"
          f" {(z < 1 - 1e-6).sum():7d} {nw:5.3f} {2 / nw:6.3f}")
