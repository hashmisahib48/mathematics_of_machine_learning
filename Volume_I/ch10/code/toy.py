# Ridge by hand: delivery data and the two-reading toy.
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])    # odometer distance, km
x2 = np.array([1.0, 2.0, 3.0, 5.0, 4.0])   # route-planner distance
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])   # time, min
xc, yc = x - x.mean(), y - y.mean()
S_xy, S_xx = xc @ yc, xc @ xc
print(f"one feature: S_xy = {S_xy:.0f}   S_xx = {S_xx:.0f}")
for a in (0, 10):
    w = S_xy / (S_xx + a)                  # Eq. 10.5-1
    b = y.mean() - w * x.mean()
    r = yc - w * xc
    sse, pen = r @ r, a * w**2
    print(f"alpha={a:2d} w={w:.3f} b={b:.3f} SSE={sse:.4f}"
          f" penalty={pen:.4f} J={sse + pen:.4f}")

X = np.c_[x, x2]
Xc = X - X.mean(axis=0)
G, g = Xc.T @ Xc, Xc.T @ yc                # X'X and X'y, centred
print("\nX'X =", G.tolist(), "  X'y =", g.tolist())
for a in (0, 1):
    w = np.linalg.solve(G + a * np.eye(2), g)    # Eq. 10.3-5
    b = y.mean() - X.mean(axis=0) @ w
    r = yc - Xc @ w
    print(f"alpha={a}  w={np.round(w, 4)}  b={b:.4f}"
          f"  SSE={r @ r:.4f}")

lam, V = np.linalg.eigh(G)                 # Eq. 10.4-1
vg = np.abs(V.T @ g)                       # |v_j' X'y|
print("\neigenvalues:", lam, "  |v'g|:", np.round(vg, 4))
for a in (0, 1):
    f = lam / (lam + a)                    # Eq. 10.4-4
    c = vg / (lam + a)                     # coefficient along v_j
    print(f"alpha={a}  f={np.round(f, 2)}  df={f.sum():.2f}"
          f"  |c|={np.round(c, 4)}")

print("\ny_4 = 6, 7, 8:   OLS w1, w2   |   ridge(1) w1, w2")
for t in (6.0, 7.0, 8.0):
    yy = np.where(np.arange(5) == 3, t, y)
    w0 = LinearRegression().fit(X, yy).coef_
    w1 = Ridge(alpha=1).fit(X, yy).coef_
    print(f"  {t:.0f}        {w0[0]:7.3f} {w0[1]:7.3f}   |"
          f"  {w1[0]:6.3f} {w1[1]:6.3f}")
