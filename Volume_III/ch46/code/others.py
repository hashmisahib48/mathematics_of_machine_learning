# The function forms, and one warning, of Chapter 46.
import warnings
import numpy as np
from returns import draw
from sklearn.covariance import (ledoit_wolf, oas, shrunk_covariance,
    graphical_lasso, empirical_covariance, LedoitWolf, OAS, MinCovDet,
    ShrunkCovariance, GraphicalLasso)
X = draw(30, seed=0)
S = empirical_covariance(X)
C, a = ledoit_wolf(X)
print("ledoit_wolf  ", a.round(4), np.allclose(C, LedoitWolf().fit(X).covariance_))
C, a = oas(X)
print("oas          ", a.round(4), np.allclose(C, OAS().fit(X).covariance_))
print("shrunk_cov   ", np.allclose(shrunk_covariance(S, 0.2),
      ShrunkCovariance(shrinkage=0.2).fit(X).covariance_))
W, T = graphical_lasso(S / np.outer(np.sqrt(np.diag(S)), np.sqrt(np.diag(S))), 0.3)
print("graphical_lasso returns (covariance, precision):", W.shape, T.shape)
mu = np.trace(S) / 20
print("shrinkage 1 gives mu I:", np.allclose(shrunk_covariance(S, 1.0), mu * np.eye(20)))
with warnings.catch_warnings(record=True) as w:     # n = 30 is p + 10
    warnings.simplefilter("always")
    m = MinCovDet(random_state=0).fit(X)
print("MinCovDet n=30, p=20: h =", int(m.raw_support_.sum()), "warnings", len(w))
D = X[:, :3]; D = np.c_[D, D[:, 0] + D[:, 1]]       # a redundant column
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    MinCovDet(random_state=0).fit(D)
print("redundant column ->", [str(x.message) for x in w])
