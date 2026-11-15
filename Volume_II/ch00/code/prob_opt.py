# Bayes' rule, Gaussian conditioning and a KKT/dual check (Examples
# R.4-1 and R.5-1).
import numpy as np
from scipy.optimize import minimize

pA, pB_A, pB_nA = 0.2, 0.9, 0.1
pB = pB_A * pA + pB_nA * (1 - pA)            # Eq. 4.2-2 (Volume I)
print("P(B) =", round(pB, 4), " P(A|B) =", round(pB_A * pA / pB, 4))

mu = np.array([1., 2.])
Sg = np.array([[4., 2.], [2., 3.]])
xa = 3.0                                     # first one observed
k = Sg[1, 0] / Sg[0, 0]                      # Eq. R.4-2
m = mu[1] + k * (xa - mu[0])
v = Sg[1, 1] - k * Sg[0, 1]
print("conditional mean and variance:", m, v)
rng = np.random.default_rng(0)
Z = rng.multivariate_normal(mu, Sg, 400000)
near = Z[np.abs(Z[:, 0] - xa) < 0.02, 1]
print("by sampling:", near.mean().round(2), near.var().round(2))

c = np.array([3., 4.])                       # minimise |w - c|^2
f = lambda w: np.sum((w - c) ** 2)           # subject to |w|^2 <= 1
con = {"type": "ineq", "fun": lambda w: 1 - w @ w}
res = minimize(f, [0.1, 0.1], method="SLSQP", constraints=[con],
               options={"ftol": 1e-12})
print("constrained minimum:", res.x.round(4), round(res.fun, 4))

def q(lam):                                  # Eq. R.5-3
    L = lambda w: f(w) + lam * (w @ w - 1)   # Lagrangian, Eq. R.5-1
    return minimize(L, [0., 0.]).fun
for lam in (0, 2, 4, 6):
    print(f"q({lam}) = {q(lam):.4f}")
print("gap at w = 0, lambda = 2:", round(f(np.zeros(2)) - q(2), 4))
