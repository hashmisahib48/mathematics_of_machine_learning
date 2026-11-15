# Problem 46B.1: rank and test score of the sample covariance
import sys
import numpy as np
sys.path.insert(0, "../../../ch46/code")     # returns.py of the book
from returns import true_model, draw
from sklearn.covariance import (EmpiricalCovariance, LedoitWolf,
                                empirical_covariance, log_likelihood)

Sigma, Theta, sd = true_model()
Xte = draw(500, seed=99)                     # 500 new days
Ste = empirical_covariance(Xte, assume_centered=True)
print("score of the true model: %.2f" % log_likelihood(Ste, Theta))
print("  n  rank  eig_min(S)  empirical  LedoitWolf  shrinkage")
for n in (10, 15, 19, 20, 21, 25, 30, 40):
    X = draw(n, seed=0)
    emp = EmpiricalCovariance().fit(X)
    lw = LedoitWolf().fit(X)
    rank = np.linalg.matrix_rank(emp.covariance_)
    lam = np.linalg.eigvalsh(emp.covariance_)[0]
    print(f"{n:3d} {rank:5d} {lam:11.1e} {emp.score(Xte):10.1f}"
          f" {lw.score(Xte):11.2f} {lw.shrinkage_:10.2f}")
