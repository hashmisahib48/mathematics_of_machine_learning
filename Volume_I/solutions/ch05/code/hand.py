# Hand checks for the Chapter 5 class A solutions (5A.1 - 5A.4). Output: verify/hand.out
import numpy as np

print("== 5A.1  f = 3(w-2)^2 + 1, eta = 0.1")
w, e0 = 0.0, 2.0
print(" t   w_t     error w_t-2   ratio")
prev = w - 2
for t in range(1, 4):
    w = w - 0.1 * 6 * (w - 2)                # f'(w) = 6(w - 2)
    print(f"{t:2d} {w:8.4f} {w - 2:12.4f} {(w - 2) / prev:8.3f}")
    prev = w - 2
print("curvature h = 6, factor 1 - eta*h =", round(1 - 0.1 * 6, 4), " limit 2/h =", round(2 / 6, 4))
for eta in (0.30, 0.3333, 0.34):             # just below / at / above the limit
    w = 0.0
    for _ in range(60): w = w - eta * 6 * (w - 2)
    print(f" eta {eta}: |w_60 - 2| = {abs(w - 2):.3g}")
print("f at w0..w3:", [round(3 * (v - 2) ** 2 + 1, 4) for v in (0, 1.2, 1.68, 1.872)])

print("\n== 5A.2  delivery loss, eta = 0.005")
x = np.arange(1., 6.); y = np.array([3., 4, 6, 7, 10]); X = np.c_[np.ones(5), x]
J = lambda v: float(((y - X @ v) ** 2).sum()); g = lambda v: -2 * X.T @ (y - X @ v)
v = np.zeros(2); print("start", v, "J", J(v), "grad", g(v), " sum y =", y.sum(), " sum xy =", (x * y).sum())
for k in (1, 2):
    gv = g(v); v = v - 0.005 * gv
    r = y - X @ v
    print(f"step {k}: w = {v.round(4)}  J = {J(v):.4f}  residuals {r.round(4)}  r^2 {(r**2).round(4)}  grad now {g(v).round(4)}")
    print("   sum r =", r.sum().round(4), " sum x r =", (x * r).sum().round(4), " sum r^2 =", (r**2).sum().round(4))
print("minimum 1.10 at", np.linalg.solve(X.T @ X, X.T @ y).round(4))

print("\n== 5A.3  Newton on e^w - 2w")
w, ws = 0.0, np.log(2); f1 = lambda w: np.exp(w) - 2; f2 = lambda w: np.exp(w)
print(" t  f'(w_{t-1})   f''(w_{t-1})   w_t        error     err_t/err_{t-1}^2")
prev = w - ws
for t in range(1, 4):
    a, b = f1(w), f2(w); w = w - a / b
    print(f"{t:2d} {a:11.6f} {b:11.6f} {w:10.6f} {w - ws:11.3e} {(w - ws) / prev**2:9.4f}")
    prev = w - ws
w4 = w - (np.exp(w) - 2) / np.exp(w); print("fourth step: w4 =", round(w4, 9), " error", f"{w4 - ws:.2e}")
print("exact ln 2 =", round(ws, 6), "; f'''/(2 f'') at ln2 =", 0.5, "; f(ln2) =", round(2 - 2 * ws, 6))

print("\n== 5A.4  extremes of x + 2y on x^2 + y^2 = 5")
for lam in (0.5, -0.5):
    xx, yy = -1 / (2 * lam), -1 / lam
    print(f" lambda {lam:+.1f}: (x, y) = ({xx:.0f}, {yy:.0f})  x^2+y^2 = {xx**2+yy**2:.0f}  f = {xx+2*yy:.0f}")
th = np.linspace(0, 2 * np.pi, 2000001); val = np.sqrt(5) * (np.cos(th) + 2 * np.sin(th))
i, j = val.argmin(), val.argmax()
print(" scan of the circle: min", round(val[i], 4), "at", (np.sqrt(5) * np.cos(th[i])).round(3), (np.sqrt(5) * np.sin(th[i])).round(3),
      " max", round(val[j], 4), "at", (np.sqrt(5) * np.cos(th[j])).round(3), (np.sqrt(5) * np.sin(th[j])).round(3))
print(" |grad f| =", np.sqrt(5).round(4), " |grad g| at (-1,-2) =", round(np.hypot(2, 4), 4), " ratio =", round(np.sqrt(5) / np.hypot(2, 4), 4))

print("\n== 5B.1  range of eta for which momentum (mu = 0.8) has |z| = sqrt(mu) in both directions")
mu = 0.8
x = np.arange(1., 6.); Xd = np.c_[np.ones(5), x]; hf, hs = np.linalg.eigvalsh(2 * Xd.T @ Xd)
lo = (1 + mu - 2 * np.sqrt(mu)) / hf; hi = (1 + mu + 2 * np.sqrt(mu)) / hs
print(f" need (1+mu-eta h)^2 < 4 mu for h = {hf:.2f} and {hs:.2f}: eta between {lo:.5f} and {hi:.5f}; sqrt(mu) = {np.sqrt(mu):.4f}")
mu = 0.95                                            # variation for the instructor note of 5B.1
for h in (hf, hs):
    b = 1 + mu - 0.008 * h; z = np.roots([1, -b, mu])
    print(f" mu = 0.95, eta = 0.008, h = {h:.2f}: b^2 = {b*b:.3f} vs 4mu = {4*mu:.2f}, factor {abs(z).max():.4f}")
