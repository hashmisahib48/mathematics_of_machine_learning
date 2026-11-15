# Problem 4B.2: normal maximum likelihood by an optimiser
import numpy as np
from scipy.optimize import minimize

d = np.array([5.0, 7.0, 8.0, 10.0, 10.0])
n = d.size

def nll(p):                           # -log-likelihood, Eq. 4.8-4
    mu, ls = p                        # ls = ln sigma^2 > -inf
    s2 = np.exp(ls)
    return 0.5 * n * np.log(2 * np.pi * s2) \
        + np.sum((d - mu) ** 2) / (2 * s2)

res = minimize(nll, [0.0, 0.0], method="Nelder-Mead",
               options={"xatol": 1e-8, "fatol": 1e-10})
mu_hat, s2_hat = res.x[0], np.exp(res.x[1])
print(f"optimiser: mu = {mu_hat:.4f}  sigma^2 = {s2_hat:.4f}")
print(f"           {res.nit} steps, nll = {res.fun:.4f}")

mu_f = d.mean()                       # Eq. 4.8-6
s2_f = np.mean((d - mu_f) ** 2)
print(f"formulas : mu = {mu_f:.4f}  sigma^2 = {s2_f:.4f}")
print(f"           nll = {nll([mu_f, np.log(s2_f)]):.4f}")
print(f"sum of squares = {np.sum((d - mu_f) ** 2):.1f}")
s2_u = d.var(ddof=1)                  # divisor n - 1
print(f"divisor n - 1: sigma^2 = {s2_u:.4f}")
print(f"nll at that value: {nll([mu_f, np.log(s2_u)]):.4f}")
print(f"closed form n/2 (ln 2 pi s2 + 1) = "
      f"{0.5 * n * (np.log(2 * np.pi * s2_f) + 1):.4f}")
