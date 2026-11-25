import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib')); sys.path.insert(0, os.path.join(HERE, '..', 'code'))
from figstyle import *
from gd import sgd
P = lambda n: os.path.join(HERE, 'png', n)
def fin(fig, n): fig.tight_layout(pad=0.3); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7.5); ax.set_ylabel(yl, fontsize=7.5); ax.tick_params(labelsize=7, length=2.2)
# 5.2-1 error of w for four rates (students)
fig, ax = plt.subplots(figsize=(1.75, 1.9), dpi=300)
for eta, c in [(0.1, BLUE), (0.25, GREEN), (0.45, OR), (0.55, RED)]:
    f = 1 - 4 * eta; k = np.arange(0, 16 if eta < 0.5 else 6); e = 5.1 * f ** k
    ax.plot(k, e, 'o-', ms=2, lw=0.9, color=c, label=rf'$\eta={eta}$')
ax.axhline(0, color='k', lw=0.4); ax.set_ylim(-16, 28); ax.legend(fontsize=6.2, frameon=False, ncol=2, loc='upper right', columnspacing=0.8, handlelength=1.6)
sty(ax, 'iteration $k$', r'$w_k-w^\ast$'); fin(fig, 'f51.png')
# 5.2-2 scaling: paths on raw and standardized deliveries
X = np.array([[2, 1], [4, 2], [5, 1], [7, 3], [9, 2.]]); y = np.array([14, 23, 24, 35, 38.])
fig, axs = plt.subplots(2, 1, figsize=(1.75, 2.45), dpi=300)
sets = []
for D in (X, (X - X.mean(0)) / X.std(0)):
    Dc = D - D.mean(0); yc = y - y.mean(); A = Dc.T @ Dc; n = 5; eta = 0.9 * n / np.trace(A)
    ws = np.linalg.solve(A, Dc.T @ yc); w = np.zeros(2); path = [w.copy()]
    for k in range(25): w = w + 2 * eta / n * Dc.T @ (yc - Dc @ w); path.append(w.copy())
    path = np.array(path); lam = np.linalg.eigvalsh(A); sets.append((Dc, yc, path, ws, lam.max() / lam.min(), eta))
    print('5.2-2 eta', round(eta, 4), 'eigenvalue ratio of A', round(lam.max() / lam.min(), 2))
R = max(((p.max(0) - p.min(0)).max() / 2 + 0.8) for _, _, p, _, _, _ in sets)      # one half-width for both panels
for ax, (Dc, yc, path, ws, ratio, eta), ttl in zip(axs, sets, ['raw features', 'standardized']):
    cc = (path.max(0) + path.min(0)) / 2
    a, b = np.meshgrid(np.linspace(cc[0] - R, cc[0] + R, 300), np.linspace(cc[1] - R, cc[1] + R, 300))
    J = ((yc[None, None, :] - a[..., None] * Dc[:, 0] - b[..., None] * Dc[:, 1]) ** 2).mean(-1)
    Jm = J.min(); ax.contour(a, b, J, levels=Jm + (J.max() - Jm) * np.array([0.003, 0.01, 0.03, 0.08, 0.16, 0.3, 0.5]), colors=BLUE, linewidths=0.5)
    ax.set_aspect('equal'); ax.set_xlim(cc[0] - R, cc[0] + R); ax.set_ylim(cc[1] - R, cc[1] + R)
    ax.plot(path[:, 0], path[:, 1], 'o-', color=OR, ms=1.8, lw=0.8); ax.plot(*ws, '*', color=RED, ms=6)
    ax.set_title(ttl, fontsize=7.5, color=NAVY); sty(ax, '$w_1$', '$w_2$')
fin(fig, 'f52.png')
# 5.2-3 SGD vs batch for students
xt = np.array([-2, -1, 0, 1, 2.]); yt = np.array([-10, -5, -1, 6, 10.])
fig, ax = plt.subplots(figsize=(1.75, 1.4), dpi=300)
w = 0.0; hist = [w]; rng = np.random.default_rng(1)
for t in range(12):
    eta = 0.05 / (1 + t)
    for i in rng.permutation(5): e = yt[i] - w * xt[i]; w += eta * 2 * xt[i] * e; hist.append(w)
ax.plot(np.arange(len(hist)) / 5, hist, color=OR, lw=0.9, label='stochastic')
w = 0.0; hb = [w]
for k in range(12): w = w + 0.05 * 2 / 5 * (xt @ (yt - w * xt)); hb.append(w)
ax.plot(np.arange(13), hb, 'o-', ms=2, color=BLUE, lw=0.9, label='batch')
ax.axhline(5.1, color=GREY, lw=0.6, ls='--', label=r'$w^\ast=5.1$'); ax.legend(fontsize=6.2, frameon=False, loc='lower right')
ax.set_xticks([0, 3, 6, 9, 12]); sty(ax, 'epoch', '$w$'); fin(fig, 'f53.png'); print('ok')
