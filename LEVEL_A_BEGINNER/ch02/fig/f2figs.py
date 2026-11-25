"""Figures of Chapter 2 (sidebar width)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
P = lambda n: os.path.join(HERE, 'png', n)
def fin(fig, n): fig.tight_layout(pad=0.3); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7); ax.tick_params(labelsize=6.3, length=2.2)
# 2.2-1 deviations from the means
x = np.array([1, 2, 3, 4, 5.]); y = np.array([52, 57, 61, 68, 72.])
fig, ax = plt.subplots(figsize=(2.05, 1.8), dpi=300)
ax.axvline(3, color=GREY, lw=0.7, ls='--'); ax.axhline(62, color=GREY, lw=0.7, ls='--')
for a, b in zip(x, y):
    c = GREEN if (a - 3) * (b - 62) > 0 else RED
    ax.add_patch(plt.Rectangle((min(a, 3), min(b, 62)), abs(a - 3), abs(b - 62), fc=c, alpha=0.18, ec=c, lw=0.5))
ax.plot(x, y, 'o', ms=3.4, color=BLUE, zorder=4)
ax.text(3.05, 50.4, r'$\bar x=3$', fontsize=6.5, color=GREY); ax.text(0.75, 62.6, r'$\bar y=62$', fontsize=6.5, color=GREY)
ax.set_xlim(0.6, 5.4); ax.set_ylim(49, 75); sty(ax, 'hours studied, $x$', 'test score, $y$'); fin(fig, 'f21.png')
# 2.3-1 two vectors
fig, ax = plt.subplots(figsize=(2.05, 1.8), dpi=300)
for v, c, lab in [((3, 4), BLUE, r'$\mathbf{u}=(3,4)$'), ((4, 3), RED, r'$\mathbf{v}=(4,3)$')]:
    arr(ax, (0, 0), v, c=c, lw=1.2, ms=8); ax.text(v[0] + 0.1, v[1] + 0.1, lab, fontsize=6.6, color=c)
t = np.linspace(np.arctan2(3, 4), np.arctan2(4, 3), 30); ax.plot(1.4 * np.cos(t), 1.4 * np.sin(t), color=GREY, lw=0.7)
ax.text(1.5, 1.25, r'$\theta$', fontsize=7, color=GREY)
ax.set_xlim(-0.3, 5.6); ax.set_ylim(-0.3, 5.0); ax.set_aspect('equal'); sty(ax, '$x_1$', '$x_2$'); fin(fig, 'f22.png')
# 2.4-1 parabola with tangent
xs = np.array([1, 2, 3.]); ys = np.array([2, 4, 7.]); J = lambda w: ((ys[:, None] - w * xs[:, None]) ** 2).sum(0)
w = np.linspace(0.6, 3.6, 200); fig, ax = plt.subplots(figsize=(2.05, 1.8), dpi=300)
ax.plot(w, J(w), color=BLUE, lw=1.2)
wt = np.array([0.6, 1.6]); ax.plot(wt, J(np.array([1.0]))[0] - 34 * (wt - 1), color=OR, lw=1.0)
ax.plot([1], J(np.array([1.0])), 'o', color=OR, ms=3.5); ax.text(1.1, 32, 'slope −34', fontsize=6.3, color=OR)
wb = 31 / 14; ax.plot([wb], J(np.array([wb])), 'o', color=RED, ms=3.8); ax.text(wb, 4.5, 'minimum\nslope 0', fontsize=6.3, color=RED, ha='center')
ax.set_ylim(-2, 48); sty(ax, '$w$', '$J(w)$'); fin(fig, 'f23.png')
# 2.4-2 contours and gradient
fig, ax = plt.subplots(figsize=(2.05, 1.9), dpi=300)
a, b = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200)); F = a ** 2 + 3 * b ** 2
ax.contour(a, b, F, levels=[0.3, 1, 2, 3.5, 5.5, 8], colors=BLUE, linewidths=0.6)
for p in [(1.5, 0.8), (-1.2, 1.0), (0.6, -1.1), (-1.6, -0.5)]:
    g = np.array([2 * p[0], 6 * p[1]]); g = g / np.linalg.norm(g) * 0.5
    arr(ax, p, (p[0] + g[0], p[1] + g[1]), c=RED, lw=0.9, ms=6); arr(ax, p, (p[0] - g[0], p[1] - g[1]), c=GREEN, lw=0.9, ms=6)
ax.plot([0], [0], 'o', color=NAVY, ms=3); ax.set_aspect('equal'); sty(ax, '$w_1$', '$w_2$')
ax.text(1.95, 1.75, r'$\nabla J$', color=RED, fontsize=7, ha='right'); ax.text(1.95, -1.9, r'$-\nabla J$', color=GREEN, fontsize=7, ha='right'); fin(fig, 'f24.png')
# 2.5-1 normal densities
fig, ax = plt.subplots(figsize=(2.05, 1.7), dpi=300); t = np.linspace(-5, 7, 400)
for mu, s, c in [(0, 1, BLUE), (2, 2, RED)]:
    ax.plot(t, np.exp(-(t - mu) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi)), color=c, lw=1.1, label=rf'$\mu={mu},\ \sigma={s}$')
ax.legend(fontsize=5.8, frameon=False); sty(ax, '$x$', '$p(x)$'); fin(fig, 'f25.png')
# 2.6-1 Euclidean and Manhattan distance
fig, ax = plt.subplots(figsize=(2.05, 1.8), dpi=300)
ax.set_xticks(range(0, 6)); ax.set_yticks(range(0, 5)); ax.grid(color=GRID, lw=0.5)
ax.plot([1, 4], [1, 4], color=BLUE, lw=1.4, label='Euclidean 4.24'); ax.plot([1, 4, 4], [1, 1, 4], color=OR, lw=1.4, label='Manhattan 6')
ax.plot([1, 4], [1, 4], 'o', color=NAVY, ms=3.5, zorder=4); ax.set_xlim(0, 5); ax.set_ylim(0, 4.6); ax.set_aspect('equal')
ax.legend(fontsize=5.8, frameon=False, loc='upper left'); sty(ax, '$x_1$', '$x_2$'); fin(fig, 'f26.png')
print('ok')
