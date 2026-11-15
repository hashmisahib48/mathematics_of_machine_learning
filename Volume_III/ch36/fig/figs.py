# Toy figures of Chapter 36 (the eight customers). Run: python3 fig/figs.py
from common import *

XL, YL = (-0.6, 6.6), (-0.8, 9.2)
good = lloyd_path(X8, [[0, 5], [5, 5]])
poor = lloyd_path(X8, [[3, 8], [1, 8]])
NUM['good_J_assign'] = [round(p[2], 4) for p in good]
NUM['poor_J_assign'] = [round(p[2], 4) for p in poor]
# J after each half step of the good run
J_half = []
for k, (mu, lab, J) in enumerate(good):
    J_half.append(('assign', k + 1, J))
    if k + 1 < len(good):
        newmu = np.array([X8[lab == c].mean(0) for c in range(2)])
        J_half.append(('update', k + 1, inertia_of(X8, lab, newmu)))
NUM['J_half'] = [(a, k, round(j, 4)) for a, k, j in J_half]


def toy_panel(ax, mu, lab, nums=True, seg=True, arrow_to=None):
    voronoi_shade(ax, mu, XL, YL)
    if seg:
        for i in range(len(X8)):
            ax.plot([X8[i, 0], mu[lab[i], 0]], [X8[i, 1], mu[lab[i], 1]], color=RED, lw=0.8, alpha=0.85, zorder=3)
    pts(ax, X8, lab, s=30)
    for i, (x, y) in enumerate(X8):
        if nums:
            ax.text(x + 0.2, y + 0.15, str(i + 1), fontsize=8, color=NAVY, zorder=8)
    centres(ax, mu, s=70)
    if arrow_to is not None:
        for a, b in zip(mu, arrow_to):
            arr(ax, a, b, c=GREEN, lw=1.3, ms=8, shrinkA=5, shrinkB=5, zorder=6)
    ax.set_xlim(*XL); ax.set_ylim(*YL); ax.set_xticks(range(0, 7, 2)); ax.set_yticks(range(0, 9, 2))
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)


# ---- f360 opener: final clustering and the descent of J
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.55), dpi=240, gridspec_kw={'width_ratios': [1, 1.12]})
mu, lab, J = good[-1]
toy_panel(axs[0], mu, lab, nums=False)
axs[0].set_xlabel('east (km)'); axs[0].set_ylabel('north (km)')
axs[0].text(0.0, 0.2, 'red: distance to depot', fontsize=8.5, color=RED)
ax = axs[1]
xs = np.arange(len(J_half)); ys = [j for _, _, j in J_half]
ax.plot(xs, ys, '-', color=RED, lw=1.2, zorder=2)
for x_, (kind, k, j) in zip(xs, J_half):
    ax.scatter([x_], [j], s=34, marker='o' if kind == 'assign' else 's', color=RED if kind == 'assign' else OR,
               edgecolor='white', lw=0.7, zorder=5)
    ax.text(x_ + 0.12, j + 5, f'{j:g}', fontsize=8.5, color=NAVY, ha='left')
ax.set_xticks(xs); ax.set_xticklabels([f'{"A" if kd == "assign" else "U"}{k}' for kd, k, _ in J_half])
ax.set_ylim(0, 112); ax.set_xlim(-0.3, len(xs) - 0.2); ax.set_ylabel('inertia $J$'); ax.set_xlabel('step: A = assign, U = update'); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig, 'f360.png', 470)

# ---- f362 three Lloyd iterations
fig, axs = plt.subplots(1, 3, figsize=(5.4, 2.1), dpi=240, sharey=True)
for k, ax in enumerate(axs):
    mu, lab, J = good[k]
    nxt = np.array([X8[lab == c].mean(0) for c in range(2)]) if k < 2 else None
    toy_panel(ax, mu, lab, arrow_to=nxt)
    ax.set_title(f'assignment {k + 1}:  $J$ = {J:g}', fontsize=9, color=NAVY)
    ax.set_xlabel('east (km)')
axs[0].set_ylabel('north (km)')
fig.tight_layout(w_pad=0.5); sv(fig, 'f362.png', 470)

# ---- f363 Voronoi cells and the bisector
rng = np.random.default_rng(4)
M = np.array([[1.5, 1.5], [4.5, 1.0], [7.0, 3.0], [2.5, 5.0], [5.8, 5.6]])
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.3), dpi=240, gridspec_kw={'width_ratios': [1.15, 1]})
ax = axs[0]
voronoi_shade(ax, M, (0, 8.5), (-0.2, 7), cols=SHADE)
centres(ax, M, 80)
P = rng.uniform([0.2, 0.2], [8.2, 6.8], size=(14, 2))
lab = ((P[:, None] - M[None]) ** 2).sum(2).argmin(1)
pts(ax, P, lab, s=22)
ax.set_xlim(0, 8.5); ax.set_ylim(-0.2, 7); ax.set_xticks([]); ax.set_yticks([])
ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
ax = axs[1]
a, b = np.array([1.0, 1.0]), np.array([4.0, 3.0]); m = (a + b) / 2; dvec = b - a; nrm = np.array([-dvec[1], dvec[0]]) / np.linalg.norm(dvec)
t = np.linspace(-2.6, 2.6, 2); L = m[None] + t[:, None] * nrm[None]
xx = np.array([[-0.2, -0.5], [5.2, 4.8]])
voronoi_shade(ax, np.array([a, b]), (-0.2, 5.6), (-0.5, 4.6), cols=[LB, LO])
ax.plot(L[:, 0], L[:, 1], color=GREEN, lw=1.6, zorder=3)
q = m + 1.5 * nrm
for p_ in (a, b):
    ax.plot([q[0], p_[0]], [q[1], p_[1]], color=RED, lw=1.0, zorder=4)
ax.plot([a[0], b[0]], [a[1], b[1]], color=GREY, lw=0.9, ls='--', zorder=3)
ax.scatter(*zip(a, b), s=85, marker='X', color=GREEN, edgecolor='white', lw=0.9, zorder=7)
ax.scatter(*q, s=26, color=NAVY, zorder=8)
ax.text(a[0] - 0.1, a[1] - 0.45, r'$\mu_1$', fontsize=9.5, color=NAVY, ha='center'); ax.text(b[0] + 0.1, b[1] + 0.2, r'$\mu_2$', fontsize=9.5, color=NAVY)
ax.text(q[0] + 0.12, q[1] + 0.05, r'$x$', fontsize=9.5, color=NAVY)
ax.text(2.7, 0.2, 'bisector', fontsize=8.5, color=GREEN, ha='center')
ax.set_xlim(-0.2, 5.6); ax.set_ylim(-0.5, 4.6); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
fig.tight_layout(w_pad=0.8); sv(fig, 'f363.png', 470)

# ---- f364 two starts, two local minima
fig, axs = plt.subplots(1, 2, figsize=(4.6, 2.45), dpi=240, sharey=True)
for ax, path, nm in ((axs[0], good, 'start (0,5), (5,5)'), (axs[1], poor, 'start (3,8), (1,8)')):
    mu, lab, J = path[-1]
    toy_panel(ax, mu, lab)
    for c in range(2):
        tr = np.array([p[0][c] for p in path] + [mu[c]])
        ax.plot(tr[:, 0], tr[:, 1], '--', color=GREEN, lw=1.0, zorder=6)
        ax.scatter([tr[0, 0]], [tr[0, 1]], s=48, marker='o', facecolor='white', edgecolor=GREEN, lw=1.3, zorder=9)
    ax.set_title(f'{nm}\n$J$ = {path[-1][2]:g}', fontsize=9, color=NAVY)
    ax.set_xlabel('east (km)')
axs[0].set_ylabel('north (km)')
fig.tight_layout(w_pad=0.6); sv(fig, 'f364.png', 400)

json.dump(NUM, open('verify/fig_numbers_toy.json', 'w'), indent=1)
json.dump(CHK, open('verify/fig_textsize_toy.json', 'w'), indent=1)
print(NUM); print({k: v['min_font_printed'] for k, v in CHK.items()})
