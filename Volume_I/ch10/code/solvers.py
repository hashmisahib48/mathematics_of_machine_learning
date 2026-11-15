# All Ridge solvers on one problem (diabetes data, alpha = 0.1).
import warnings
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

X, y = load_diabetes(return_X_y=True)
ref = Ridge(alpha=0.1, solver="cholesky").fit(X, y)

def gap(solver, **kw):                     # distance from cholesky
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = Ridge(alpha=0.1, solver=solver, random_state=0,
                  max_iter=100000, **kw).fit(X, y)
    return m, np.abs(m.coef_ - ref.coef_).max(), len(caught)

print("solver     solver_    tol 1e-4    tol 1e-10   warnings")
for name in ("auto", "svd", "cholesky", "lsqr", "sparse_cg",
             "sag", "saga"):
    m, g4, n4 = gap(name)
    _, g10, n10 = gap(name, tol=1e-10)
    print(f"{name:10s} {m.solver_:10s} {g4:.1e}    {g10:.1e}"
          f"     {n4 + n10}")

m = Ridge(alpha=0.1, positive=True).fit(X, y)   # needs lbfgs
print("positive=True ->", m.solver_, "  min w =",
      round(m.coef_.min(), 4))
m = Ridge(alpha=0.1).fit(sp.csr_matrix(X), y)   # sparse input
print("sparse input  ->", m.solver_)
wide = np.random.default_rng(0).normal(size=(20, 200))   # p > n
m = Ridge(alpha=1.0).fit(wide, np.arange(20.0))
print("wide X (20 x 200) ->", m.solver_, " coef_", m.coef_.shape)
