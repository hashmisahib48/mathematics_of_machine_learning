# Six heights, two groups: one EM iteration by hand, in NumPy.
import numpy as np

x = np.array([154., 160., 164., 170., 176., 184.])
pi = np.array([0.5, 0.5])                 # initial guess
mu = np.array([158., 178.])
var = np.array([64., 64.])


def density(x, pi, mu, var):              # pi_k N(x | mu_k, var_k)
    z = (x[:, None] - mu) ** 2 / var
    return pi * np.exp(-0.5 * z) / np.sqrt(2 * np.pi * var)


def loglik(x, pi, mu, var):               # Eq. 40.3-3
    return np.log(density(x, pi, mu, var).sum(axis=1)).sum()


d = density(x, pi, mu, var)               # numerators of Eq. 40.3-4
gam = d / d.sum(axis=1, keepdims=True)    # E-step: responsibilities
print("pi_k N(x_i | k), their sum, and the responsibilities")
print(np.c_[d, d.sum(axis=1)].round(5))
print(gam.round(4))

Nk = gam.sum(axis=0)                      # M-step, Eqs. 40.4-5 to 7
pi_new = Nk / len(x)
mu_new = gam.T @ x / Nk
var_new = (gam * (x[:, None] - mu_new) ** 2).sum(axis=0) / Nk
print("sum g*x     ", (gam * x[:, None]).sum(axis=0).round(2))
print("N_k         ", Nk.round(4))
print("pi          ", pi_new.round(4))
print("mu          ", mu_new.round(3))
dev2 = (gam * (x[:, None] - mu_new) ** 2).sum(axis=0)
print("sum g*dev^2 ", dev2.round(2))
print("sigma^2     ", var_new.round(3))
print("log L before", loglik(x, pi, mu, var).round(4))
print("log L after ", loglik(x, pi_new, mu_new, var_new).round(4))
