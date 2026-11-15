# Covariance estimators of scikit-learn on 30 days of 20 returns.
import numpy as np
from returns import true_model, draw
from sklearn.covariance import (EmpiricalCovariance, LedoitWolf,
    OAS, ShrunkCovariance, GraphicalLassoCV, log_likelihood)
from sklearn.covariance import empirical_covariance as emp_cov
from sklearn.preprocessing import StandardScaler

Sigma, Theta, sd = true_model()
X, Xte = draw(30, seed=0), draw(2000, seed=99)
p = X.shape[1]

def report(name, par, cov, prec, loc):
    err = ((cov - Sigma) ** 2).sum() / p    # = error_norm
    Ste = emp_cov(Xte - loc, assume_centered=True)
    ll = log_likelihood(Ste, prec)          # = score
    ev = np.linalg.eigvalsh(cov)
    print(f"{name:12s}{par:>8s}{err:8.2f}{ll:9.2f}"
          f"{ev[0]:8.3f}{ev[-1] / ev[0]:8.1f}")

print(f"{'estimator':12s}{'par':>8s}{'error':>8s}{'test ll':>9s}"
      f"{'lam_min':>8s}{'cond':>8s}")
report("true Sigma", "", Sigma, Theta, np.zeros(p))
for name, m in [("Empirical", EmpiricalCovariance()),
                ("Shrunk 0.1", ShrunkCovariance(shrinkage=0.1)),
                ("LedoitWolf", LedoitWolf()), ("OAS", OAS())]:
    m.fit(X)
    a = getattr(m, "shrinkage_", m.get_params().get("shrinkage", 0))
    report(name, f"{a:.3f}", m.covariance_, m.precision_,
           m.location_)

sc = StandardScaler().fit(X)                # glasso on z-scores
gl = GraphicalLassoCV(enet_tol=1e-6).fit(sc.transform(X))
s2 = np.outer(sc.scale_, sc.scale_)         # original units
report("GraphLassoCV", f"{gl.alpha_:.3f}", gl.covariance_ * s2,
       gl.precision_ / s2, sc.mean_)
edges = ((np.abs(gl.precision_) > 0).sum() - p) // 2
print("graphical lasso edges", edges, "of", p * (p - 1) // 2,
      "; true edges", ((np.abs(Theta) > 0).sum() - p) // 2)
