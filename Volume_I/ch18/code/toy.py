# Seed sorting in one dimension: LDA and QDA by hand (Section 18.6).
import numpy as np

A = np.array([1.0, 3, 3, 3, 3, 5])      # variety A: length, mm
B = np.array([6.0, 7, 7, 8])            # variety B
n0, n1 = len(A), len(B)
pi0, pi1 = n0 / (n0 + n1), n1 / (n0 + n1)
m0, m1 = A.mean(), B.mean()
v0, v1 = A.var(), B.var()               # divide by n
v = (n0 * v0 + n1 * v1) / (n0 + n1)     # shared, Eq. 18.3-10
print(f"variances {v0:.4f} {v1:.4f}  shared {v:.4f}")

w = (m1 - m0) / v                       # Eq. 18.3-7
b = -(m1**2 - m0**2) / (2 * v) + np.log(pi1 / pi0)
x_eq = (m0 + m1) / 2                    # boundary, equal priors
x_pr = x_eq + v / (m1 - m0) * np.log(pi0 / pi1)    # Eq. 18.3-9
print(f"w = {w:.4f}  b = {b:.4f}")
print(f"boundary: equal priors {x_eq:.4f}, data priors {x_pr:.4f}")
for x in (4.0, 5.0, 5.05, 6.0):
    p1 = 1 / (1 + np.exp(-(w * x + b)))
    print(f"x = {x:4.2f}  P(B | x) = {p1:.4f}")

# QDA: equal log posteriors at the roots (Eq. 18.6-1).
a2 = 1/(2*v0) - 1/(2*v1)
a1 = m1/v1 - m0/v0
a0 = (m0**2/(2*v0) - m1**2/(2*v1)
      - 0.5*np.log(v1/v0) + np.log(pi1/pi0))
print(f"a2 = {a2:.4f}  a1 = {a1:.4f}  a0 = {a0:.4f}")
roots = np.sort(np.roots([a2, a1, a0]))
print("QDA boundaries:", np.round(roots, 4))
