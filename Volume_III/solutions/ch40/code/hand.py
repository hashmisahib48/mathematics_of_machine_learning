# Hand checks for the Chapter 40 solutions (problems 40A.1 to 40A.3). Run: python3 hand.py
import numpy as np
from math import exp, sqrt, pi, log

phi = lambda z: exp(-z * z / 2) / sqrt(2 * pi)
print("40A.1  weights 0.4 and 0.6, means 0 and 3, standard deviation 1")
for x in (2.0, 1.0):
    n1, n2 = phi(x - 0), phi(x - 3)
    a, b = 0.4 * n1, 0.6 * n2
    print(f"  x = {x:.0f}: z = ({x - 0:.0f}, {x - 3:+.0f})  exp(-z^2/2) = ({exp(-(x - 0) ** 2 / 2):.6f}, {exp(-(x - 3) ** 2 / 2):.6f})")
    print(f"        N1 = {n1:.5f}, N2 = {n2:.5f}; pi N = {a:.6f}, {b:.6f}; sum {a + b:.6f}; gamma = ({a / (a + b):.4f}, {b / (a + b):.4f})")

print("\n40A.2  x = 1, 2, 6, 7; gamma_1 = 0.9, 0.8, 0.2, 0.1")
x = np.array([1., 2., 6., 7.]); g1 = np.array([.9, .8, .2, .1]); G = np.c_[g1, 1 - g1]
N = G.sum(0); print("  N_k =", N, " pi =", N / 4)
sx = G.T @ x; print("  sum gamma x =", sx.round(4), " mu =", (sx / N).round(4))
mu = sx / N
for k in range(2):
    terms = G[:, k] * (x - mu[k]) ** 2
    print(f"  component {k + 1}: gamma (x - mu)^2 =", terms.round(4).tolist(), " sum", terms.sum().round(4), " variance", (terms.sum() / N[k]).round(4))
print("  check: N1 mu1 + N2 mu2 =", (N * mu).sum(), " = sum of x =", x.sum())

print("\n40A.3  n = 500, p = 4, K = 3")
n, p, K = 500, 4, 3
cs = {'full': K * p * (p + 1) // 2, 'diag': K * p, 'spherical': K}
ll = {'full': -3010., 'diag': -3040., 'spherical': -3105.}
print("  ln n =", round(log(n), 4))
res = {}
for t in cs:
    q = (K - 1) + K * p + cs[t]
    res[t] = (q, -2 * ll[t] + q * log(n), -2 * ll[t] + 2 * q)
    print(f"  {t:9s} covariance numbers {cs[t]:2d}  q = {K - 1} + {K * p} + {cs[t]} = {q}   -2 ll = {-2 * ll[t]:.0f}   BIC = {-2 * ll[t]:.0f} + {q} x {log(n):.4f} = {-2 * ll[t] + q * log(n):.1f}   AIC = {-2 * ll[t]:.0f} + {2 * q} = {-2 * ll[t] + 2 * q:.0f}")

print("\nextras for the notes and the discussion of 40A.1 to 40A.3")
n1, n2 = phi(2.0), phi(-1.0)
print("  40A.1 with equal weights, x = 2: gamma_1 = %.4f; x = 1: gamma_1 = %.4f" % (n1 / (n1 + n2), phi(1.0) / (phi(1.0) + phi(-2.0))))
xb = 1.5 + log(0.4 / 0.6) / 3
print("  40A.1 gamma_1 = 0.5 at x = 3/2 + ln(0.4/0.6)/3 = %.4f; check gamma_1 there: %.4f" % (xb, 0.4 * phi(xb) / (0.4 * phi(xb) + 0.6 * phi(xb - 3))))
print("  40A.2 variance of component 1 about the overall mean 4 instead of mu_1: %.3f" % ((G[:, 0] * (x - 4) ** 2).sum() / N[0]))
print("  40A.2 check, gamma_i1 + gamma_i2 for every sample =", (G.sum(1)).tolist())
qf, bf, af = res['full']; qd, bd, ad = res['diag']; qs, bs, as_ = res['spherical']
print("  40A.3 BIC full - diag = %.1f, AIC full - diag = %.1f; -2 ll gain of full over diag = %.0f for %d more parameters" % (bf - bd, af - ad, -2 * ll['diag'] + 2 * ll['full'], qf - qd))
print("        charge of %d parameters: BIC %.1f, AIC %d" % (qf - qd, (qf - qd) * log(n), 2 * (qf - qd)))
print("        diag - spherical: -2 ll gain %.0f for %d more parameters; charge BIC %.1f, AIC %d" % (-2 * ll['spherical'] + 2 * ll['diag'], qd - qs, (qd - qs) * log(n), 2 * (qd - qs)))
