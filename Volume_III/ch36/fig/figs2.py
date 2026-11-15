# Seeding, variants and flow chart of Chapter 36. Run: python3 fig/figs2.py
from common import *
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans, kmeans_plusplus
from sklearn.datasets import make_blobs

# ---- f365: D^2 weights on the toy, and the effect of seeding on 8 blobs
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.45), dpi=240, gridspec_kw={'width_ratios': [1, 1.15]})
ax = axs[0]
d2 = ((X8 - X8[4]) ** 2).sum(1)
NUM['d2_first5'] = d2.tolist()
for i, (x, y) in enumerate(X8):
    if i == 4:
        continue
    ax.scatter([x], [y], s=d2[i] * 3.2 + 10, color=BLUE, alpha=0.35, edgecolor=BLUE, lw=0.8, zorder=3)
    ax.scatter([x], [y], s=10, color=BLUE, zorder=5)
    off = {0: (0.35, -0.2), 1: (0.3, -0.2), 2: (0.3, 0.15), 3: (0.35, 0.3), 5: (0.35, -0.2), 6: (0.35, 0.3), 7: (0.0, 0.7)}[i]
    ax.text(x + off[0] - (0.2 if i in (7,) else 0), y + off[1], f'{int(d2[i])}', fontsize=8.5, color=NAVY, zorder=8)
ax.scatter([4], [0], s=85, marker='X', color=GREEN, edgecolor='white', lw=0.9, zorder=7)
ax.text(4.35, -0.55, 'first depot', fontsize=8.5, color=GREEN, ha='left', va='center')
ax.set_xlim(-0.4, 6.8); ax.set_ylim(-1.2, 9.9); ax.set_xlabel('east (km)'); ax.set_ylabel('north (km)')
ax.set_title('area $\\propto D^2$', fontsize=9, color=NAVY)
ax = axs[1]
Xb, _ = make_blobs(800, centers=8, cluster_std=0.9, center_box=(-10, 10), random_state=4)
R = 200
Jr = np.array([KMeans(8, init='random', n_init=1, random_state=s).fit(Xb).inertia_ for s in range(R)])
Jp = np.array([KMeans(8, init='k-means++', n_init=1, random_state=s).fit(Xb).inertia_ for s in range(R)])
best = float(min(Jr.min(), Jp.min()))
NUM['seed_best'] = round(best, 2)
for nm, J in (('random', Jr), ('k-means++', Jp)):
    NUM[f'seed_{nm}'] = {'mean': round(float(J.mean()), 1), 'median': round(float(np.median(J)), 1),
                         'share_within_2pct': round(float((J < best * 1.02).mean()), 3), 'max': round(float(J.max()), 1)}
bins = np.linspace(best * 0.98, 4000, 51)
ax.hist(Jr, bins=bins, color=OR, alpha=0.75, label='random', edgecolor='white', lw=0.4)
ax.hist(Jp, bins=bins, color=BLUE, alpha=0.75, label='k-means++', edgecolor='white', lw=0.4)
ax.axvline(best, color=GREEN, lw=1.4, ls='--', label='best found')
ax.set_xlabel('final inertia of a single run'); ax.set_ylabel(f'runs (of {R})'); ax.legend(loc='upper right', fontsize=8.5)
fig.tight_layout(w_pad=1.0); sv(fig, 'f365.png', 470)
# best of ten
NUM['seed_blobs_kpp_n_iter'] = int(KMeans(8, init='k-means++', n_init=1, random_state=0).fit(Xb).n_iter_)
b10 = [KMeans(8, init='random', n_init=10, random_state=s).fit(Xb).inertia_ for s in range(30)]
NUM['random_n_init10_share_within_2pct'] = round(float((np.array(b10) < best * 1.02).mean()), 3)
b10p = [KMeans(8, init='k-means++', n_init=10, random_state=s).fit(Xb).inertia_ for s in range(30)]
NUM['kpp_n_init10_share_within_2pct'] = round(float((np.array(b10p) < best * 1.02).mean()), 3)

# ---- f366: mini-batch paths, mini-batch convergence, bisecting births
Xc, yc = make_blobs(6000, centers=5, cluster_std=1.0, center_box=(-9, 9), random_state=11)
fig, axs = plt.subplots(1, 3, figsize=(5.4, 2.15), dpi=240)
ax = axs[0]
ax.scatter(Xc[:, 0], Xc[:, 1], s=1.5, color=LGREY, zorder=1, rasterized=True)
C0 = Xc[np.random.default_rng(20).choice(len(Xc), 5, replace=False)]   # five random data points
left = np.argsort(Xc[:, 0])[:300]
C1 = Xc[np.random.default_rng(5).choice(left, 5, replace=False)]   # a poor start: all on the left
mb = MiniBatchKMeans(5, init=C1, n_init=1, batch_size=64, random_state=2, reassignment_ratio=0)
rng = np.random.default_rng(0)
paths = [C1.copy()]
for t in range(60):
    mb.partial_fit(Xc[rng.choice(len(Xc), 64)]); paths.append(mb.cluster_centers_.copy())
paths = np.array(paths)
for c in range(5):
    ax.plot(paths[:, c, 0], paths[:, c, 1], '-', color=PUR, lw=1.0, zorder=4)
    ax.scatter(paths[0, c, 0], paths[0, c, 1], s=22, facecolor='white', edgecolor=PUR, lw=1.1, zorder=6)
centres(ax, paths[-1], 55)
ax.set_title('60 steps of 64 points', fontsize=9, color=NAVY); ax.set_xticks([]); ax.set_yticks([])
ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
ax = axs[1]
full = KMeans(5, n_init=10, random_state=0).fit(Xc); Jopt = full.inertia_
NUM['full_J_blobs5'] = round(float(Jopt), 2)
for bs, col, ls in ((16, ORANGE if False else OR, '-'), (128, BLUE, '--'), (1024, GREEN, '-.')):
    seen, rel = [], []
    m = MiniBatchKMeans(5, init=C0, n_init=1, batch_size=bs, random_state=2, reassignment_ratio=0)
    rng = np.random.default_rng(1); n_seen = 0
    for t in range(int(24000 / bs)):
        m.partial_fit(Xc[rng.choice(len(Xc), bs)]); n_seen += bs
        if t % max(1, int(24000 / bs / 40)) == 0:
            seen.append(n_seen); rel.append(100 * (-m.score(Xc) / Jopt - 1))
    ax.plot(seen, rel, ls, color=col, lw=1.2, label=f'batch {bs}')
    NUM[f'mb_final_excess_pct_bs{bs}'] = round(float(rel[-1]), 4)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.01, 300)
ax.set_xlabel('points seen'); ax.set_ylabel('excess inertia (%)'); ax.legend(loc='upper right', fontsize=8)
ax.set_title('mini-batch inertia', fontsize=9, color=NAVY); grid(ax)
ax = axs[2]
labs = {k: BisectingKMeans(k, random_state=0).fit(Xc).labels_ for k in range(1, 6)}
born = {}
for k in range(2, 6):
    for c in set(labs[k]):
        S = frozenset(np.where(labs[k] == c)[0])
        if S not in born:
            born[S] = k
final = labs[5]
for c in range(5):
    m_ = final == c; S = frozenset(np.where(m_)[0])
    ax.scatter(Xc[m_, 0], Xc[m_, 1], s=2, color=PTCOL[c], zorder=2, rasterized=True)
    cx, cy = Xc[m_].mean(0)
    ax.text(cx, cy, f'{born[S]}', fontsize=10, color=NAVY, ha='center', va='center', zorder=8, weight='bold',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=GREEN, lw=0.8))
ax.set_title('first appears at $K$ =', fontsize=9, color=NAVY); ax.set_xticks([]); ax.set_yticks([])
ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
NUM['bisect_births'] = sorted(born.values())
NUM['bisect_J'] = round(float(BisectingKMeans(5, random_state=0).fit(Xc).inertia_), 2)
fig.tight_layout(w_pad=0.7); sv(fig, 'f366.png', 470)

# ---- f368 flow chart
fig, ax = plt.subplots(figsize=(5.4, 2.3), dpi=240)
ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis('off')
def bx_(x, y, w, h, t, fc=LB, ec=BLUE, fs=8.6):
    box(ax, x, y, w, h, t, fc=fc, ec=ec, fs=fs)
bx_(0.05, 3.1, 2.15, 1.3, 'seed $K$ centres\n(k-means++)', fc=LGREY, ec=GREY)
bx_(2.75, 3.1, 2.85, 1.3, 'assign each point\nto its nearest centre')
bx_(6.15, 3.1, 2.85, 1.3, 'move each centre to\nthe mean of its points', fc='#E3F4EA', ec=GREEN)
bx_(9.45, 3.1, 1.5, 1.3, 'labels\nsame?', fc=LO, ec=OR)
for a_, b_ in (((2.2, 3.75), (2.75, 3.75)), ((5.6, 3.75), (6.15, 3.75)), ((9.0, 3.75), (9.45, 3.75))):
    arr(ax, a_, b_, c=NAVY, lw=1.1, ms=8)
ax.plot([9.7, 9.7, 4.1], [3.1, 2.25, 2.25], color=RED, lw=1.1); arr(ax, (4.1, 2.25), (4.1, 3.1), c=RED, lw=1.1, ms=8)
ax.text(6.9, 2.0, 'no: repeat', fontsize=8.6, color=RED, ha='center', va='top')
bx_(8.95, 0.1, 2.0, 1.0, 'yes: keep\nthis run', fc=LO, ec=OR)
arr(ax, (10.7, 3.1), (10.7, 1.1), c=NAVY, lw=1.1, ms=8)
bx_(2.2, 0.1, 5.5, 1.0, '$n_{\\mathrm{init}}$ runs from different seeds;\nkeep the lowest inertia', fc=LGREY, ec=GREY)
arr(ax, (8.95, 0.6), (7.7, 0.6), c=NAVY, lw=1.1, ms=8)
sv(fig, 'f368.png', 470)

json.dump(NUM, open('verify/fig_numbers_seed.json', 'w'), indent=1)
json.dump(CHK, open('verify/fig_textsize_seed.json', 'w'), indent=1)
print(json.dumps(NUM)); print({k: v['min_font_printed'] for k, v in CHK.items()})
