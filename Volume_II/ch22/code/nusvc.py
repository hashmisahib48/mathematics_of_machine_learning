# NuSVC: nu bounds the margin errors and the support vectors.
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC

X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=0.9, flip_y=0.08,
    random_state=2)
Z = StandardScaler().fit_transform(X)
s, n = 2 * y - 1, len(y)
print("  nu   margin errors   support vectors   train acc")
for nu in (0.1, 0.2, 0.3, 0.5, 0.7):
    m = NuSVC(nu=nu, kernel="rbf", gamma=0.5, tol=1e-6).fit(Z, y)
    err = np.sum(s * m.decision_function(Z) < 1 - 1e-4) / n
    sv = len(m.support_) / n
    ok = "bounds hold" if err <= nu <= sv else "VIOLATED"
    print(f"{nu:5.2f} {err:12.3f} {sv:15.3f} {m.score(Z, y):12.3f}"
          f"   {ok}")
print("class sizes:", np.bincount(y), " largest feasible nu:",
      2 * np.bincount(y).min() / n)
try:
    NuSVC(nu=0.99).fit(Z, y)
except ValueError as e:
    print("nu = 0.99 ->", e)
