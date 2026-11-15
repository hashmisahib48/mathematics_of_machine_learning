# Case study 1: how many days are needed for 20 stocks?
import json
import numpy as np
from returns import true_model, draw
from sklearn.covariance import (EmpiricalCovariance, LedoitWolf,
    OAS, GraphicalLassoCV, log_likelihood)
from sklearn.covariance import empirical_covariance as emp_cov
from sklearn.preprocessing import StandardScaler

Sigma, Theta, sd = true_model()
p, R = 20, 10
Xte = draw(2000, seed=777)
ns = (30, 45, 60, 120, 240, 480)

def glasso(X):
    sc = StandardScaler().fit(X)
    g = GraphicalLassoCV(enet_tol=1e-6).fit(sc.transform(X))
    s2 = np.outer(sc.scale_, sc.scale_)
    return g.covariance_ * s2, g.precision_ / s2, sc.mean_

def ll(cov, prec, loc):
    Ste = emp_cov(Xte - loc, assume_centered=True)
    return log_likelihood(Ste, prec)

out = {}
print("   n   method        error   test ll")
for n in ns:
    names = ("Empirical", "LedoitWolf", "OAS", "GraphLassoCV")
    rows = {k: [] for k in names}
    for r in range(R):
        X = draw(n, seed=500 + r)
        for k, m in (("Empirical", EmpiricalCovariance()),
                     ("LedoitWolf", LedoitWolf()), ("OAS", OAS())):
            m.fit(X)
            rows[k].append((m.error_norm(Sigma), m.score(Xte)))
        C, P, loc = glasso(X)
        rows["GraphLassoCV"].append((((C - Sigma) ** 2).sum() / p,
                                     ll(C, P, loc)))
    for k, v in rows.items():
        e, s = np.mean(v, axis=0)
        out.setdefault(k, []).append((n, float(e), float(s)))
        print(f"{n:4d}   {k:12s}{e:8.2f}{s:10.2f}")
json.dump(out, open("../verify/case_returns.json", "w"))
