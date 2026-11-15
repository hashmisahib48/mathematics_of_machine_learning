# Problem 3C.3: gradient descent on the delivery loss from (3, 1). Writes fig/s3c3.png, verify/c3.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *

x = np.array([1., 2, 3, 4, 5]); y = np.array([3., 4, 6, 7, 10]); X = np.c_[np.ones(5), x]
J = lambda v: float(np.sum((y - X @ v) ** 2)); gJ = lambda v: -2 * X.T @ (y - X @ v)
H = 2 * X.T @ X; vs = np.linalg.solve(X.T @ X, X.T @ y)
lam, U = np.linalg.eigh(H)
print("eigenvalues of H:", lam.round(2), " eigenvectors (columns):", U.round(3).tolist())
print("2/lambda_max =", round(2 / lam[1], 4))
v0 = np.array([3., 1.])
r0 = y - X @ v0; print('at (3, 1): residuals', r0, ' sum r =', r0.sum(), ' sum x r =', x @ r0, ' gradient', gJ(v0), ' J =', J(v0), ' minimum', vs.round(4))
for eta in (0.005, 0.01, 0.016):
    v = v0.copy(); path = [v.copy()]; n = 0
    while np.linalg.norm(v - vs) > 0.01 and n < 5000:
        v = v - eta * gJ(v); path.append(v.copy()); n += 1
    fac = 1 - eta * lam
    print(f"eta = {eta}: factors per step along the eigen-directions {fac.round(4)}; steps to reach distance 0.01: {n}")
    if eta == 0.01: P10 = np.array(path)
print("first 6 points for eta = 0.01:\n", P10[:7].round(4))
print("first step:", P10[1] - P10[0], " = -eta*grad =", -0.01 * gJ(v0))
print("J along the path:", [round(J(p), 3) for p in P10[:8]])
# component of the error along the flat and the steep axis at the start
e0 = v0 - vs; c = U.T @ e0; print("error at start:", e0, " components (flat, steep):", c.round(4))
print("flat component after 20 steps:", (c[0] * (1 - 0.01 * lam[0]) ** 20).round(4), " steep:", (c[1] * (1 - 0.01 * lam[1]) ** 20).round(6))
ang = np.degrees(np.arccos((-gJ(v0)) @ (vs - v0) / np.linalg.norm(gJ(v0)) / np.linalg.norm(vs - v0))); print("angle between -grad and the direction to the minimum:", round(ang, 1))

fig, ax = plt.subplots(figsize=(4.4, 2.3), dpi=240)
bb = np.linspace(-1.5, 4.2, 300); ww = np.linspace(0.6, 2.6, 200); Bg, Wg = np.meshgrid(bb, ww)
Jg = ((y[None, None, :] - Bg[..., None] - Wg[..., None] * x[None, None, :]) ** 2).sum(-1)
ax.contour(Bg, Wg, Jg, levels=[1.5, 3, 6, 12, 25, 50, 100], colors=BLUE, linewidths=0.8)
ax.plot(P10[:, 0], P10[:, 1], '-', color=GREEN, lw=0.9, label='descent path, $\\eta$ = 0.01'); ax.plot(P10[:60:10, 0], P10[:60:10, 1], 'o', color=GREEN, ms=2.6)
ax.plot([3, vs[0]], [1, vs[1]], '--', color=OR, lw=1.0, label='straight line to the minimum')
arr(ax, (3, 1), (3, 1.55), c=RED, lw=1.3, ms=8); ax.text(3.12, 1.3, '$-\\nabla J$', fontsize=8, color=RED, va='center')
ax.plot(*vs, '*', color=OR, ms=9); ax.text(vs[0] - 0.1, vs[1] + 0.13, 'minimum', fontsize=8, color=OR, ha='right')
ax.set_xlabel('$b$', fontsize=8); ax.set_ylabel('$w$', fontsize=8); ax.tick_params(labelsize=8); ax.legend(fontsize=7.5, loc='upper left', frameon=True, facecolor='white', edgecolor='none', framealpha=0.95)
ax.set_xlim(-1.5, 4.2); ax.set_ylim(0.6, 2.6)
save(fig, 'fig/s3c3.png')
