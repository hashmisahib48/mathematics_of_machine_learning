# Numbers quoted in the prose of Chapter 16 (not a listing).
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
x = np.arange(1.0, 9.0)
y = np.array([0, 0, 0, 1, 0, 1, 1, 1.0])
X = np.c_[np.ones(8), x]
m = np.polyfit(x, y, 1)
print("OLS line: slope %.4f intercept %.4f" % tuple(m))
for v in (0, 1, 8, 10):
    print("  OLS at", v, "->", round(float(np.polyval(m, v)), 4))
print("odds table")
for p in (0.1, 0.25, 0.5, 0.75, 0.9):
    print(f"  p={p:4.2f} odds={p/(1-p):6.4f} logodds={np.log(p/(1-p)):7.4f}")


def newton(w):
    p = sig(X @ w)
    g = X.T @ (p - y)
    S = p * (1 - p)
    H = X.T @ (X * S[:, None])
    return p, g, S, H, w - np.linalg.solve(H, g)


w = np.zeros(2)
for t in range(2):
    p, g, S, H, wn = newton(w)
    print(f"iteration {t}: w={w.round(4)} g={g.round(4)}")
    print("  p   =", p.round(4))
    print("  p-y =", (p - y).round(4))
    print("  S   =", S.round(4))
    print("  H   =", H.round(4).tolist(), " step", (wn - w).round(4))
    w = wn
w = np.zeros(2)
for t in range(30):
    w = newton(w)[4]
p = sig(X @ w)
print("optimum", w.round(4), "p =", p.round(4))
print("  sum p =", p.sum().round(6), " sum x(p-y) =",
      (x * (p - y)).sum().round(6), " J =",
      (-(y * np.log(p) + (1 - y) * np.log(1 - p))).sum().round(4))
print("  loss each", (-(y * np.log(p) + (1 - y) * np.log(1 - p))).round(4))
print("  P(6h) =", round(float(sig(w[0] + 6 * w[1])), 4),
      " P(3h) =", round(float(sig(w[0] + 3 * w[1])), 4),
      " P(5h) =", round(float(sig(w[0] + 5 * w[1])), 4))
print("  boundary", round(float(-w[0] / w[1]), 4),
      " odds ratio", round(float(np.exp(w[1])), 4),
      " e^(w*2)", round(float(np.exp(2 * w[1])), 3))
print("accuracy at 0.5:", float(((p > 0.5) == (y == 1)).mean()))
for C, n in ((1.0, 426), (0.1, 426), (np.inf, 426)):
    print(f"C={C}: lambda = 1/(C n) = {1/(C*n):.5f}  (n={n})")
