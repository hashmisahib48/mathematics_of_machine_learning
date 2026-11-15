# Constraint versus penalty: minimise (w1-3)^2 + (w2-4)^2.
import numpy as np
from scipy.optimize import minimize

c = np.array([3.0, 4.0])                       # unconstrained min.
f = lambda w: np.sum((w - c) ** 2)

print("radius t   lambda   constrained w     penalised w")
for t in (5.0, 3.0, 1.0, 0.5):
    con = {"type": "ineq", "fun": lambda w: t**2 - w @ w}
    res = minimize(f, [0.1, 0.1], constraints=[con], method="SLSQP",
                   options={"ftol": 1e-12})
    lam = 5 / t - 1                            # from Eq. 5.7-5
    pen = c / (1 + lam)                        # min of f + lam w.w
    print(f"{t:8.1f} {lam:8.2f}   {res.x.round(4)}"
          f"   {pen.round(4)}")

# The L1 ball |w1| + |w2| <= 1: search its boundary.
s = np.linspace(-1, 1, 200001)
edge = np.r_[np.c_[s, 1 - abs(s)], np.c_[s, -(1 - abs(s))]]
best = edge[np.argmin(((edge - c) ** 2).sum(axis=1))]
print("closest point of the diamond:", best.round(4))
soft = np.sign(c) * np.maximum(np.abs(c) - 3.0, 0)   # Eq. 5.7-9
print("penalty 6(|w1| + |w2|), threshold 3:", soft)
