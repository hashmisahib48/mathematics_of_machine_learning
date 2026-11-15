# Hand checks for the Chapter 31 solutions: Problems 31A.1 to 31A.4.
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sig = lambda z: 1 / (1 + np.exp(-z))
W1 = np.array([[0.5, 1.0], [-1.0, 0.5]]); b1 = np.array([0.5, -0.5])
W2 = np.array([1.0, -1.0]); b2 = 0.0

print("== 31A.1  part (2, 1), class 0")
x = np.array([2.0, 1.0])
print("x W1 =", x @ W1, "  z1 = x W1 + b1 =", x @ W1 + b1)
z1 = x @ W1 + b1; h = np.maximum(z1, 0); z2 = h @ W2 + b2; p = sig(z2)
print("h =", h, "  z2 =", z2, "  exp(-z2) =", round(np.exp(-z2), 4), " p =", round(p, 4), " 1-p =", round(1 - p, 4))
print("loss -ln(1-p) =", round(-np.log(1 - p), 4), "  also ln(1+e^z2) =", round(np.log(1 + np.exp(z2)), 4))

print("\n== 31A.2  part (1, 2), class 0")
x = np.array([1.0, 2.0]); y = 0
z1 = x @ W1 + b1; h = np.maximum(z1, 0); z2 = h @ W2 + b2; p = sig(z2)
print("x W1 =", x @ W1, " z1 =", z1, " h =", h, " z2 =", z2, " p =", round(p, 4))
d2 = p - y; act = (z1 > 0) * 1.0; d1 = d2 * W2 * act
print("delta2 =", round(d2, 4), " W2*delta2 =", W2 * d2, " phi'(z1) =", act, " delta1 =", d1)
print("grad W2 = h*delta2 =", h * d2, "  grad b2 =", round(d2, 4))
print("grad W1 = outer(x, delta1) =\n", np.outer(x, d1), "\ngrad b1 =", d1)
print("zero entries: W2[0], column 0 of W1 (", np.outer(x, d1)[:, 0], "), b1[0]")
print("unit 1 for every part of the data (z1[0] = x1*0.5 - x2*1 + 0.5):")
X8 = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1], [-2, 1], [-1, 2], [1, -2], [2, -1]], float)
print("  z1 of unit 1 for the eight parts:", (X8 @ W1 + b1)[:, 0], "  active on", int(((X8 @ W1 + b1)[:, 0] > 0).sum()), "of 8")

print("\n== 31A.3  64 - 100 - 50 - 10")
sizes = [64, 100, 50, 10]; tot = 0
for a, b in zip(sizes[:-1], sizes[1:]):
    print(f"coefs_ ({a}, {b}): {a * b} weights + {b} intercepts = {a * b + b}"); tot += a * b + b
print("total", tot)
z = np.array([2.0, 0.0, -1.0]); e = np.exp(z)
print("exp(z) =", e, " sum =", round(e.sum(), 4), " softmax =", e / e.sum(), " sums to", round((e / e.sum()).sum(), 6))
print("loss = -ln p_2 =", round(-np.log(e[1] / e.sum()), 4), "   = ln(sum) - z_2 =", round(np.log(e.sum()) - z[1], 4))

print("\n== 31A.4  gradients 1, 1, -2; eta 0.1, mu 0.5")
g = [1.0, 1.0, -2.0]; eta, mu = 0.1, 0.5
w = 0.0; print("plain:")
for t, gt in enumerate(g, 1): w += -eta * gt; print(f"  step {t}: g = {gt:4.1f}  w = {w:.4f}")
w = 0.0; v = 0.0; print("momentum  v = mu v - eta g,  w = w + v:")
for t, gt in enumerate(g, 1): vo = v; v = mu * v - eta * gt; w += v; print(f"  step {t}: mu v_old = {mu * vo:6.3f}  -eta g = {-eta * gt:6.3f}  v = {v:7.4f}  w = {w:.4f}")
w = 0.0; v = 0.0; print("Nesterov  w = w + mu v - eta g  (v the new velocity):")
for t, gt in enumerate(g, 1): v = mu * v - eta * gt; w += mu * v - eta * gt; print(f"  step {t}: v = {v:7.4f}  mu v - eta g = {mu * v - eta * gt:7.4f}  w = {w:.4f}")
