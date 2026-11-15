# Numbers for Problems 3C.1 (kink) and 3C.2 (saturation)
import numpy as np
sg = lambda z: 1 / (1 + np.exp(-z))

print("== 3C.1  minimise f(z) = 0.5 (z - 0.3)^2 + 0.5 |z|, minimum at z = 0")
lam, m = 0.5, 0.3
print("np.sign(0.0) =", np.sign(0.0), "  np.sign(-2.0), np.sign(2.0) =", np.sign(-2.0), np.sign(2.0))
print("optimality: 0 in (z - m) + lam*[-1, 1] at z = 0  <=>  |m| <= lam :", abs(m) <= lam,
      "  subgradient used:", m / lam)
soft = np.sign(m) * max(abs(m) - lam, 0.0); print("soft-threshold of 0.3 at 0.5 =", soft)
f = lambda z: 0.5 * (z - m) ** 2 + lam * abs(z)
g = lambda z: (z - m) + lam * np.sign(z)          # sign(0) = 0 picks the subgradient 0
for name, step in (("fixed step 0.1", lambda t: 0.1), ("step 0.5/t", lambda t: 0.5 / t)):
    z = 2.0; tail = []
    for t in range(1, 401):
        z -= step(t) * g(z); tail.append(z)
    print(f"{name:15s} z after 50, 100, 400 steps: {tail[49]:+.5f} {tail[99]:+.5f} {tail[399]:+.5f}"
          f"   f = {f(tail[399]):.5f}   range of last 10 z: {min(tail[-10:]):+.4f} .. {max(tail[-10:]):+.4f}")
print("f(0) =", f(0.0))
rng = np.random.default_rng(0); z = rng.normal(size=1_000_000)
print("share of exact zeros among 10^6 normal draws:", np.mean(z == 0.0))

print("\n== 3C.2  saturation")
for zz in (0, 2, 6, 10):
    print(f"z = {zz:2d}: sigma = {sg(zz):.6f}  slope = {sg(zz) * (1 - sg(zz)):.3e}")
z = 10.0; p = sg(z); s = p * (1 - p)
print("wrong prediction z = 10, y = 0: p - y =", p, " slope dp/dz =", s)
print("  squared error (p-y)^2: dL/dz = 2 (p-y) slope =", 2 * p * s)
print("  log loss:               dL/dz = p - y        =", p, "  ratio =", p / (2 * p * s))
print("correct prediction z = 10, y = 1: squared:", 2 * (p - 1) * s, "  log loss:", p - 1)

# one weight, two classes: x = +1 has y = 0, x = -1 has y = 1; start with w = 10 (every prediction wrong)
x = np.array([1.0, -1.0]); y = np.array([0.0, 1.0])
def run(loss, eta, w0=10.0, cap=200000):
    w = w0
    for t in range(1, cap + 1):
        p = sg(w * x)
        d = (2 * (p - y) * p * (1 - p)) if loss == "squared" else (p - y)   # dL/dz per sample
        w -= eta * np.mean(d * x)
        if w <= 0: return t, w
    return cap, w
for loss in ("log", "squared"):
    t, w = run(loss, 0.5); print(f"{loss:8s} eta = 0.5: w reaches 0 after {t} steps (w = {w:+.3f})")
w = 10.0
for t in range(1, 1001):
    p = sg(w * x); w -= 0.5 * np.mean(2 * (p - y) * p * (1 - p) * x)
    if t in (1, 10, 100, 1000): print(f"squared, step {t:4d}: w = {w:.6f}")
print("prediction (e^10 - 1)/(2 eta) =", (np.exp(10) - 1) / (2 * 0.5))
