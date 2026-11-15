# All figures of Chapter 38. Run: python3 fig/figs.py
# (PNGs into ch38/, numbers into verify/fig_numbers.json)
from common import *
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from birds import birds, varying, gps
from hdbscan_tree import hdbscan_scratch
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN, OPTICS, MeanShift, estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score as ari
plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                     'axes.labelsize': 9, 'legend.fontsize': 8.5})
sv = Saver()
NUM = {}
ONLY = set(sys.argv[1:])
want = lambda k: (not ONLY) or (k in ONLY)

XB, YB = birds()
TOY = np.array([[1, 1], [2, 1], [2, 2], [3, 2], [1, 2],
                [6, 1], [7, 1], [7, 2], [8, 2], [4, 5]], float)

# ------------------------------------------------ 38.0-1 the idea
if want('f380'):
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.45), dpi=240, sharey=True)
    km = KMeans(2, n_init=10, random_state=0).fit(XB).labels_
    db = DBSCAN(eps=0.2, min_samples=5).fit(XB).labels_
    NUM['f380'] = {'ari_kmeans': round(ari(YB, km), 3), 'ari_dbscan': round(ari(YB, db), 3),
                   'noise_dbscan': int((db == -1).sum())}
    for ax, lab, t in ((axs[0], km, 'k-means, $K=2$'), (axs[1], db, 'DBSCAN, $\\varepsilon=0.2$, $m=5$')):
        scatter_labels(ax, XB, lab, s=15)
        ax.set_title(t, fontsize=9.5, color=NAVY)
        ax.set_xlabel('east')
        ax.set_xlim(-1.6, 2.6); ax.set_ylim(-1.05, 1.55)
        ax.set_xticks([-1, 0, 1, 2]); grid(ax)
    axs[0].set_ylabel('north')
    axs[0].scatter(*KMeans(2, n_init=10, random_state=0).fit(XB).cluster_centers_.T, s=70, marker='*', color=GREEN, edgecolor='white', lw=0.6, zorder=6)
    axs[1].scatter([], [], s=14, marker='x', color=NOISE, lw=0.8, label='noise (label $-1$)')
    axs[1].legend(loc='lower right', handletextpad=0.2, borderaxespad=0.2)
    fig.tight_layout(w_pad=0.6)
    sv(fig, 'f380.png', 470)

# ------------------------------------------------ 38.1-1 the ten birds
if want('f381'):
    fig, ax = plt.subplots(figsize=(3.3, 2.3), dpi=240)
    ax.scatter(TOY[:, 0], TOY[:, 1], s=46, color=BLUE, edgecolor='white', lw=0.8, zorder=5)
    off = {1: (-0.28, -0.05), 2: (0.12, -0.38), 3: (0.12, 0.12), 4: (0.12, 0.12), 5: (-0.28, 0.1),
           6: (-0.1, 0.22), 7: (0.0, -0.45), 8: (-0.2, 0.2), 9: (0.15, 0.1), 10: (0.14, 0.1)}
    for k, (x, y) in enumerate(TOY, 1):
        ax.text(x + off[k][0], y + off[k][1], str(k), fontsize=9, color=NAVY, ha='left', va='bottom')
    ax.set_xlim(0, 9.4); ax.set_ylim(0, 6); ax.set_xlabel('east (km)'); ax.set_ylabel('north (km)')
    ax.set_xticks(range(0, 10, 2)); ax.set_yticks(range(0, 7, 2)); grid(ax); ax.set_aspect('equal')
    sv(fig, 'f381.png', 330)

# ------------------------------------------------ 38.2-1 neighbourhoods and reachability
if want('f382'):
    eps, m = 1.5, 4
    D = np.sqrt(((TOY[:, None] - TOY[None]) ** 2).sum(-1))
    cnt = (D <= eps).sum(1)
    fig, axs = plt.subplots(1, 2, figsize=(5.3, 2.1), dpi=240)
    ax = axs[0]
    for k, col in ((2, GREEN), (3, OR), (9, RED)):
        ax.add_patch(Circle(TOY[k], eps, fc='none', ec=col, lw=1.3, ls='-' if k != 9 else '--', zorder=2))
    ax.scatter(TOY[:, 0], TOY[:, 1], s=26, color=BLUE, edgecolor='white', lw=0.6, zorder=5)
    hs = [Line2D([0], [0], color=GREEN, lw=1.4), Line2D([0], [0], color=OR, lw=1.4), Line2D([0], [0], color=RED, lw=1.4, ls='--')]
    ax.legend(hs, ['point 3: core', 'point 4: border', 'point 10: noise'],
              loc='upper right', fontsize=8.2, handlelength=1.4, handletextpad=0.4, borderaxespad=0.0, labelspacing=0.3)
    ax.set_xlim(-0.3, 9.4); ax.set_ylim(-0.7, 6.8); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('$\\varepsilon$-neighbourhoods, $m=4$', fontsize=9.5, color=NAVY)
    ax = axs[1]
    lab = DBSCAN(eps=eps, min_samples=m).fit(TOY)
    core = np.zeros(10, bool); core[lab.core_sample_indices_] = True
    for i in np.flatnonzero(core):
        ax.add_patch(Circle(TOY[i], eps, fc=CL[lab.labels_[i]], ec='none', alpha=0.13, zorder=1))
    for i in range(10):
        k = lab.labels_[i]
        if k == -1:
            ax.scatter(*TOY[i], s=40, marker='x', color=NOISE, lw=1.2, zorder=5)
        elif core[i]:
            ax.scatter(*TOY[i], s=40, marker='o', color=CL[k], edgecolor='white', lw=0.6, zorder=5)
        else:
            ax.scatter(*TOY[i], s=40, marker='o', facecolor='white', edgecolor=CL[k], lw=1.4, zorder=5)
    for a, b in ((4, 2), (2, 3)):       # chain 5 -> 3 -> 4
        ax.add_patch(FancyArrowPatch(TOY[a] + (TOY[b] - TOY[a]) * 0.1, TOY[b] - (TOY[b] - TOY[a]) * 0.12,
                                     arrowstyle='-|>', mutation_scale=8, color=NAVY, lw=1.0, zorder=6))
    ax.text(2.0, 4.0, 'chain 5 → 3 → 4', fontsize=8.5, color=NAVY, ha='center')
    ax.scatter([], [], s=30, color=BLUE, label='core'); ax.scatter([], [], s=30, facecolor='white', edgecolor=BLUE, lw=1.3, label='border')
    ax.scatter([], [], s=30, marker='x', color=NOISE, label='noise')
    ax.legend(loc='upper right', handletextpad=0.1, borderaxespad=0.0, labelspacing=0.2, fontsize=8.2, bbox_to_anchor=(1.0, 1.0))
    ax.set_xlim(-0.3, 9.4); ax.set_ylim(-0.7, 6.8); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('two clusters and one stray', fontsize=9.5, color=NAVY)
    fig.tight_layout(w_pad=0.4)
    sv(fig, 'f382.png', 470)

# ------------------------------------------------ 38.3-1 the k-distance plot
if want('f383'):
    fig, ax = plt.subplots(figsize=(3.9, 2.5), dpi=240)
    NUM['kdist'] = {}
    for m_, ls, col in ((3, ':', '#B79ACB'), (5, '-', PUR), (12, '--', '#4B2565')):
        d = np.sort(NearestNeighbors(n_neighbors=m_).fit(XB).kneighbors(XB)[0][:, -1])[::-1]
        ax.plot(np.arange(1, len(d) + 1), d, ls=ls, color=col, lw=1.5, label=f'$m={m_}$')
        NUM['kdist'][str(m_)] = {'max': round(float(d[0]), 3), 'rank12': round(float(d[12]), 3), 'rank20': round(float(d[19]), 3),
                                 'median': round(float(np.median(d)), 3), 'above_0.2': int((d > 0.2).sum()), 'above_0.25': int((d > 0.25).sum())}
    ax.axhline(0.2, color=GREY, lw=0.9, ls='-.')
    ax.text(150, 0.215, '$\\varepsilon=0.2$', fontsize=9, color=GREY)
    ax.set_xlabel('points, sorted by $c_m$ (largest first)'); ax.set_ylabel('core distance $c_m$')
    ax.set_xlim(0, 252); ax.set_ylim(0, 0.65); grid(ax)
    ax.legend(loc='upper right', handlelength=1.8, borderaxespad=0.1)
    ax.annotate('the strays', xy=(10, 0.5), xytext=(40, 0.52), fontsize=8.5, color=NAVY, va='center', arrowprops=dict(arrowstyle='-|>', color=NAVY, lw=0.8))
    sv(fig, 'f383.png', 340)


def sl_tree(h):
    t = h._single_linkage_tree_      # private attribute, used for the figure only
    return np.c_[t['left_node'], t['right_node'], t['value'], t['cluster_size']]


def condensed_layout(res, n):
    rows, kids = res['rows'], res['kids']
    size = {c: s for (p, c, l, s) in rows if c >= n}
    size[n] = n
    death = {}
    for p, c, l, s in rows:
        if np.isfinite(l):
            death[p] = max(death.get(p, 0), l)
    birth = res['birth']
    xs, cnt = {}, [0]

    def lay(c):
        if not kids[c]:
            xs[c] = cnt[0]; cnt[0] += 1
        else:
            for k in kids[c]:
                lay(k)
            xs[c] = np.mean([xs[k] for k in kids[c]])
    lay(n)
    return xs, size, death, birth


# ------------------------------------------------ 38.4-1 HDBSCAN: clusters and condensed tree
if want('f384'):
    X, y = varying()
    ms, mcs = 5, 8
    h = HDBSCAN(min_samples=ms, min_cluster_size=mcs, copy=False).fit(X)
    res = hdbscan_scratch(X, ms, mcs, 'eom', Z=sl_tree(h))
    assert np.array_equal(res['labels'], h.labels_)
    hl = HDBSCAN(min_samples=ms, min_cluster_size=mcs, cluster_selection_method='leaf', copy=False).fit(X)
    NUM['f384'] = {'n': int(len(X)), 'clusters_eom': int(h.labels_.max() + 1), 'noise_eom': int((h.labels_ == -1).sum()),
                   'ari_eom': round(ari(y, h.labels_), 3), 'clusters_leaf': int(hl.labels_.max() + 1),
                   'noise_leaf': int((hl.labels_ == -1).sum()), 'ari_leaf': round(ari(y, hl.labels_), 3),
                   'n_condensed_clusters': len(res['S']), 'S': {str(k): round(float(v), 1) for k, v in res['S'].items()},
                   'kids': {str(k): v for k, v in res['kids'].items()}, 'selected': res['selected'],
                   'birth_lambda': {str(k): round(float(v), 3) for k, v in res['birth'].items()}}
    fig, axs = plt.subplots(1, 2, figsize=(5.3, 2.6), dpi=240, gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axs[0]
    lab = h.labels_
    for k in sorted(set(lab)):
        mk = lab == k
        if k == -1:
            ax.scatter(X[mk, 0], X[mk, 1], s=12, marker='x', color=NOISE, lw=0.8, zorder=3)
        else:
            ax.scatter(X[mk, 0], X[mk, 1], s=15, marker=MK[k], color=CL[k], edgecolor='white', lw=0.3, zorder=4)
    ax.set_xlim(-1.6, 7.2); ax.set_ylim(-1.6, 4.7); ax.set_aspect('equal'); grid(ax)
    ax.set_xticks([0, 2, 4, 6]); ax.set_yticks([0, 2, 4]); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
    ax.set_title('EOM selection', fontsize=9.5, color=NAVY)
    ax = axs[1]
    xs, size, death, birth = condensed_layout(res, len(X))
    sel = set(res['selected'])
    top = 0.62
    for c in res['kids']:
        lam0 = max(birth[c], top); lam1 = death[c]
        col = GREEN if c in sel else GREY
        ax.plot([xs[c], xs[c]], [lam0, lam1], color=col, lw=0.8 + 5.5 * size[c] / len(X), solid_capstyle='butt', zorder=3, alpha=0.95 if c in sel else 0.6)
        for k in res['kids'][c]:
            ax.plot([xs[c], xs[k]], [birth[k], birth[k]], color=GREY, lw=0.8, zorder=2)
        if c != len(X):
            yl = np.sqrt(lam0 * lam1) if lam1 / lam0 > 1.2 else lam1 * 1.13
            ax.text(xs[c] + 0.14, yl, f"{res['S'][c]:.0f}", fontsize=8.2, color=GREEN if c in sel else GREY,
                    ha='left', va='center', weight='bold' if c in sel else 'normal')
    ax.set_yscale('log'); ax.invert_yaxis()
    ax.set_ylim(26, top); ax.set_yticks([1, 2, 5, 10, 20]); ax.set_yticklabels(['1', '2', '5', '10', '20'])
    ax.set_xlim(-0.6, 6.7); ax.set_xticks([]); ax.set_ylabel('$\\lambda = 1/\\varepsilon$'); ax.spines['bottom'].set_visible(False)
    ax.set_title('condensed tree; numbers are stabilities $S$', fontsize=9.5, color=NAVY)
    ax.minorticks_off()
    fig.tight_layout(w_pad=0.8)
    sv(fig, 'f384.png', 470)

# ------------------------------------------------ 38.5-1 OPTICS: reachability plot
if want('f385'):
    X, y = varying()
    o = OPTICS(min_samples=8, xi=0.1, min_cluster_size=20).fit(X)
    reach = o.reachability_[o.ordering_].copy()
    lab_o = o.labels_[o.ordering_]
    from sklearn.cluster import cluster_optics_dbscan
    cuts = {}
    for eps in (0.35, 0.8):
        l = cluster_optics_dbscan(reachability=o.reachability_, core_distances=o.core_distances_, ordering=o.ordering_, eps=eps)
        cuts[str(eps)] = {'clusters': int(l.max() + 1), 'noise': int((l == -1).sum()), 'ari': round(ari(y, l), 3)}
    NUM['f385'] = {'xi_clusters': int(o.labels_.max() + 1), 'xi_noise': int((o.labels_ == -1).sum()), 'xi_ari': round(ari(y, o.labels_), 3),
                   'cuts': cuts, 'n_inf': int(np.isinf(o.reachability_).sum()), 'first_point': int(o.ordering_[0]),
                   'max_finite_reach': round(float(reach[np.isfinite(reach)].max()), 3), 'hierarchy_len': int(len(o.cluster_hierarchy_))}
    fig = plt.figure(figsize=(5.3, 2.75), dpi=240)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.0], height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); axr = fig.add_subplot(gs[1, :])
    scatter_labels(ax, X, o.labels_, s=11)
    ax.set_title('OPTICS, $\\xi=0.1$', fontsize=9.5, color=NAVY); clean(ax)
    l35 = cluster_optics_dbscan(reachability=o.reachability_, core_distances=o.core_distances_, ordering=o.ordering_, eps=0.35)
    scatter_labels(ax2, X, l35, s=11); clean(ax2)
    ax2.set_title('cut at $\\varepsilon=0.35$', fontsize=9.5, color=NAVY)
    xs_ = np.arange(len(X))
    cols = [NOISE if l == -1 else CL[l % 8] for l in lab_o]
    rr = np.minimum(np.where(np.isfinite(reach), reach, 9.0), 1.3)
    axr.bar(xs_, rr, width=1.0, color=cols, lw=0)
    axr.axhline(0.35, color=GREY, lw=0.9, ls='--'); axr.axhline(0.8, color=GREY, lw=0.9, ls='-.')
    axr.text(len(X) + 3, 0.35, '$\\varepsilon=0.35$', fontsize=8.5, color=GREY, va='center')
    axr.text(len(X) + 3, 0.8, '$\\varepsilon=0.8$', fontsize=8.5, color=GREY, va='center')
    axr.set_xlim(0, len(X)); axr.set_ylim(0, 1.3)
    axr.set_xlabel('position in the OPTICS ordering'); axr.set_ylabel('reachability')
    axr.set_yticks([0, 0.5, 1.0])
    fig.subplots_adjust(left=0.1, right=0.86, top=0.92, bottom=0.17)
    sv(fig, 'f385.png', 470)

# ------------------------------------------------ 38.6-1 mean shift in one and two dimensions
def ms_path(X, x0, h, maxit=100):
    x = np.array(x0, float); path = [x.copy()]
    for _ in range(maxit):
        w = np.linalg.norm(X - x, axis=-1) <= h if X.ndim > 1 else np.abs(X - x) <= h
        nx = X[w].mean(axis=0)
        path.append(nx.copy())
        if np.linalg.norm(nx - x) <= 1e-3 * h:
            break
        x = nx
    return np.array(path)


if want('f386'):
    from sklearn.datasets import make_blobs
    rng = np.random.default_rng(4)
    x1 = np.r_[rng.normal(0.0, 0.5, 22), rng.normal(3.2, 0.6, 18)]
    h1 = 0.45
    xg = np.linspace(-2, 5.5, 1500)
    def gauss(x, X, h):
        return np.exp(-0.5 * ((np.atleast_1d(x)[:, None] - X[None]) / h) ** 2).sum(1) / (len(X) * h * np.sqrt(2 * np.pi))
    def gauss_path(x0, X, h, maxit=200):
        x = float(x0); p = [x]
        for _ in range(maxit):
            w = np.exp(-0.5 * ((x - X) / h) ** 2)
            nx = float((w * X).sum() / w.sum()); p.append(nx)
            if abs(nx - x) < 1e-5:
                break
            x = nx
        return np.array(p)
    f = gauss(xg, x1, h1)
    nmodes = lambda ff: int(sum(1 for i in range(1, len(ff) - 1) if ff[i] > ff[i - 1] and ff[i] > ff[i + 1]))
    modes_h = {str(hh): nmodes(gauss(xg, x1, hh)) for hh in (0.15, 0.25, 0.35, 0.45, 0.6, 0.8, 1.0, 1.5)}
    fig, axs = plt.subplots(1, 2, figsize=(5.3, 2.35), dpi=240, gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axs[0]
    ax.plot(xg, f, color=PUR, lw=1.6)
    ax.fill_between(xg, f, color=PUR, alpha=0.08)
    ax.plot(x1, np.full_like(x1, -0.015), '|', color=BLUE, ms=6, mew=0.9, clip_on=False)
    NUM['f386'] = {'modes_by_h': modes_h, 'h': h1, 'paths': {}}
    for s0 in (-1.3, 1.4, 2.0, 4.6):
        p = gauss_path(s0, x1, h1)
        sel_ = np.r_[0:min(len(p), 6)]
        pp = p[sel_]
        ax.plot(pp, gauss(pp, x1, h1), 'o-', color=GREEN, ms=3.2, lw=0.9, zorder=5, mfc='white', mec=GREEN, mew=0.9)
        NUM['f386']['paths'][str(s0)] = {'iterations': int(len(p) - 1), 'limit': round(float(p[-1]), 3)}
    ax.set_xlim(-2, 5.5); ax.set_ylim(-0.03, 0.45); ax.set_xlabel('$x$'); ax.set_ylabel('density estimate')
    ax.set_title('Gaussian kernel, $h=0.45$: climbing', fontsize=9.2, color=NAVY)
    ax.set_yticks([0, 0.2, 0.4]); grid(ax)
    ax = axs[1]
    Xb, yb3 = make_blobs(n_samples=120, centers=[[0, 0], [4, 0.5], [2, 4]], cluster_std=0.65, random_state=2)
    h2 = 2.0
    for p0 in Xb[::3]:
        p = ms_path(Xb, p0, h2)
        ax.plot(p[:, 0], p[:, 1], '-', color=GREEN, lw=0.7, zorder=3, alpha=0.8)
    ax.scatter(Xb[:, 0], Xb[:, 1], s=9, color=BLUE, edgecolor='white', lw=0.3, zorder=2)
    ms = MeanShift(bandwidth=h2).fit(Xb)
    ax.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1], s=60, marker='*', color=RED, edgecolor='white', lw=0.5, zorder=6)
    ax.add_patch(Circle(Xb[0], h2, fc='none', ec=PUR, lw=1.0, ls='--', zorder=4))
    ax.scatter(*Xb[0], s=22, color=PUR, zorder=5)
    NUM['f386']['blobs'] = {'centres': ms.cluster_centers_.round(3).tolist(), 'h': h2, 'n_iter': int(ms.n_iter_)}
    ax.set_xlim(-2.3, 6.3); ax.set_ylim(-2.0, 5.9); ax.set_aspect('equal'); clean(ax)
    ax.set_title('paths of 40 seeds, $h=2$', fontsize=9.2, color=NAVY)
    fig.tight_layout(w_pad=0.6)
    sv(fig, 'f386.png', 470)

# ------------------------------------------------ 38.8-1 the four algorithms as flow charts
if want('f388'):
    rows = [('DBSCAN', ['radius\nquery', 'core test\n$|N_\\varepsilon|\\geq m$', 'seed a new\ncluster', 'expand\nby a queue', 'leftovers:\nnoise'], BLUE),
            ('HDBSCAN', ['core\ndistance', 'mutual\nreachability', 'spanning\ntree', 'merge\nhierarchy', 'condense,\nstability', 'select\nclusters'], GREEN),
            ('OPTICS', ['core\ndistance', 'least\nreachability', 'update\nneighbours', 'reachability\nplot', 'extract $\\xi$\nor cut'], PUR),
            ('Mean shift', ['seed at\neach point', 'move to\nwindow mean', 'repeat until\nstable', 'merge\nnear modes', 'label by\nnearest mode'], OR)]
    fig, ax = plt.subplots(figsize=(5.3, 3.2), dpi=240)
    ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis('off')
    bw, gap, bh = 14.6, 2.4, 9.6
    ys = [54.5, 37.5, 21.5, 5.5]
    for r_, (name, items, col) in enumerate(rows):
        y0 = ys[r_]
        ax.text(0, y0 + bh + 2.2, name, fontsize=9.5, color=col, weight='bold', va='center')
        for k, t in enumerate(items):
            x0 = k * (bw + gap)
            ax.add_patch(FancyBboxPatch((x0, y0), bw, bh, boxstyle='round,pad=0.0,rounding_size=1.2', fc='white', ec=col, lw=1.1))
            ax.text(x0 + bw / 2, y0 + bh / 2, t, fontsize=8.0, color=NAVY, ha='center', va='center', linespacing=1.1)
            if k:
                ax.add_patch(FancyArrowPatch((x0 - gap + 0.1, y0 + bh / 2), (x0 - 0.1, y0 + bh / 2), arrowstyle='-|>', mutation_scale=7, color=GREY, lw=0.9))
        if name == 'DBSCAN':
            x2 = 3 * (bw + gap) + bw / 2; x1 = 2 * (bw + gap) + bw / 2
            ax.add_patch(FancyArrowPatch((x2, y0 + bh + 0.1), (x1, y0 + bh + 0.1), arrowstyle='-|>', mutation_scale=7, color=RED, lw=0.9, connectionstyle='arc3,rad=0.6'))
            ax.text(79, y0 + bh + 3.0, 'next unlabelled core', fontsize=8.2, color=RED, ha='center', va='center')
    sv(fig, 'f388.png', 470)

# ------------------------------------------------ 38.11-1 DBSCAN gallery over eps x min_samples
if want('f3811a'):
    epss, mss = (0.1, 0.2, 0.4), (5, 15)
    fig, axs = plt.subplots(2, 3, figsize=(5.0, 3.15), dpi=240, sharex=True, sharey=True)
    NUM['f3811a'] = {}
    for i, e in enumerate(epss):
        for j, m_ in enumerate(mss):
            ax = axs[j, i]
            lab = DBSCAN(eps=e, min_samples=m_).fit(XB).labels_
            scatter_labels(ax, XB, lab, s=7, noise_s=8, lw=0.2)
            nc, nn = int(lab.max() + 1), int((lab == -1).sum())
            NUM['f3811a'][f'{e}_{m_}'] = {'clusters': nc, 'noise': nn, 'ari': round(ari(YB, lab), 3)}
            ax.text(0.04, 0.05, f'{nc} cl., {nn} noise', transform=ax.transAxes, fontsize=8.2, color=NAVY, va='bottom',
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.8))
            clean(ax)
            if j == 0:
                ax.set_title(f'$\\varepsilon={e}$', fontsize=9.5, color=PUR)
            if i == 0:
                ax.set_ylabel(f'$m={m_}$', fontsize=9.5, color=NAVY)
    for ax in axs.ravel():
        ax.set_xlim(-1.6, 2.6); ax.set_ylim(-1.05, 1.55)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.01, wspace=0.04, hspace=0.05)
    sv(fig, 'f3811a.png', 470)

# ------------------------------------------------ 38.11-2 curves over eps and over min_cluster_size
if want('f3811b'):
    X2, y2 = XB, YB
    es = np.round(np.arange(0.04, 0.62, 0.02), 3)
    nc, nz, ar = [], [], []
    for e in es:
        l = DBSCAN(eps=e, min_samples=5).fit(X2).labels_
        nc.append(int(l.max() + 1)); nz.append(float((l == -1).mean())); ar.append(ari(y2, l))
    mcs_list = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60]
    ae, al, ce, cl = [], [], [], []
    for mc in mcs_list:
        for meth, A_, C_ in (('eom', ae, ce), ('leaf', al, cl)):
            l = HDBSCAN(min_cluster_size=mc, cluster_selection_method=meth, copy=False).fit(X2).labels_
            A_.append(ari(y2, l)); C_.append(int(l.max() + 1))
    NUM['f3811b'] = {'eps': es.tolist(), 'n_clusters': nc, 'noise_frac': [round(v, 3) for v in nz], 'ari': [round(v, 3) for v in ar],
                    'best_eps_ari': round(float(max(ar)), 3), 'best_eps': float(es[int(np.argmax(ar))]),
                    'eps_good_range': [float(es[i]) for i in range(len(es)) if ar[i] > 0.9],
                    'mcs': mcs_list, 'ari_eom': [round(v, 3) for v in ae], 'ari_leaf': [round(v, 3) for v in al],
                    'clusters_eom': ce, 'clusters_leaf': cl}
    fig, axs = plt.subplots(1, 2, figsize=(5.3, 2.35), dpi=240, gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axs[0]
    ax.plot(es, ar, color=GREEN, lw=1.7, label='adjusted Rand index')
    ax.plot(es, nz, color=RED, lw=1.4, ls='--', label='share of noise')
    ax.set_xlabel('$\\varepsilon$'); ax.set_ylim(-0.05, 1.05); ax.set_xlim(0.04, 0.6); grid(ax)
    good = [e for e, a_ in zip(es, ar) if a_ > 0.9]
    ax.axvspan(min(good) - 0.01, max(good) + 0.01, color=PUR, alpha=0.12, lw=0)
    ax2 = ax.twinx(); ax2.spines['right'].set_visible(True); ax2.step(es, nc, where='mid', color=PUR, lw=1.3, ls=':', label='clusters')
    ax2.set_ylabel('clusters', color=PUR); ax2.set_ylim(0, 30); ax2.tick_params(axis='y', colors=PUR)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='center right', fontsize=8.2, handlelength=1.8, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.62))
    ax.set_title('DBSCAN, $m=5$', fontsize=9.5, color=NAVY)
    ax = axs[1]
    ax.plot(mcs_list, ae, '-o', color=GREEN, ms=3.2, lw=1.4, label='eom')
    ax.plot(mcs_list, al, '--s', color=RED, ms=3.2, lw=1.2, label='leaf')
    ax.set_xscale('log'); ax.set_xticks([3, 5, 10, 20, 40, 60]); ax.set_xticklabels(['3', '5', '10', '20', '40', '60']); ax.minorticks_off()
    ax.set_xlabel('`min_cluster_size`'.replace('`', '')); ax.set_ylabel('adjusted Rand index'); ax.set_ylim(-0.05, 1.05); grid(ax)
    ax.legend(loc='upper right', fontsize=8.2, borderaxespad=0.2, handlelength=1.8)
    ax.set_title('HDBSCAN', fontsize=9.5, color=NAVY)
    fig.tight_layout(w_pad=0.7)
    sv(fig, 'f3811b.png', 470)

# ------------------------------------------------ 38.11-3 four methods on three data sets
if want('f3811c'):
    from sklearn.datasets import make_circles
    rng = np.random.default_rng(5)
    Xc, yc = make_circles(n_samples=240, factor=0.4, noise=0.05, random_state=0)
    Xc = np.r_[Xc, rng.uniform(-1.4, 1.4, (12, 2))]; yc = np.r_[yc, np.full(12, -1)]
    sets = [('flocks', XB, YB, (0.2, 5)), ('densities', *varying(), (0.35, 8)), ('rings', Xc, yc, (0.2, 5))]
    names = ['DBSCAN', 'HDBSCAN', 'OPTICS', 'MeanShift']
    fig, axs = plt.subplots(3, 4, figsize=(5.3, 4.3), dpi=240)
    NUM['f3811c'] = {}
    for i, (nm, X, y, (e, m_)) in enumerate(sets):
        labs = [DBSCAN(eps=e, min_samples=m_).fit(X).labels_,
                HDBSCAN(min_cluster_size=15, copy=False).fit(X).labels_,
                OPTICS(min_samples=8, xi=0.1, min_cluster_size=20).fit(X).labels_,
                MeanShift(bandwidth=estimate_bandwidth(X)).fit(X).labels_]
        for j, lab in enumerate(labs):
            ax = axs[i, j]
            scatter_labels(ax, X, lab, s=5, noise_s=6, lw=0.15)
            nc, nn = int(lab.max() + 1), int((lab == -1).sum())
            a = ari(y, lab)
            NUM['f3811c'][f'{nm}_{names[j]}'] = {'clusters': nc, 'noise': nn, 'ari': round(a, 3)}
            clean(ax)
            ax.set_xlabel(f'{nc} cl., {nn} noise, ARI {a:.2f}', fontsize=8.2, color=NAVY, labelpad=2)
            ax.set_xlim(X[:, 0].min() - 0.15, X[:, 0].max() + 0.15); ax.set_ylim(X[:, 1].min() - 0.15, X[:, 1].max() + 0.15)
            if i == 0:
                ax.set_title(names[j], fontsize=9.5, color=NAVY)
            if j == 0:
                ax.set_ylabel(nm, fontsize=9.5, color=NAVY)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.955, bottom=0.025, wspace=0.05, hspace=0.2)
    sv(fig, 'f3811c.png', 470)

# ------------------------------------------------ 38.12-1 case study: GPS stops
if want('f3813'):
    X, y = gps()
    runs = [('truth', y), ('k-means, $K=4$', KMeans(4, n_init=10, random_state=0).fit(X).labels_),
            ('DBSCAN, $\\varepsilon=30$', DBSCAN(eps=30, min_samples=5).fit(X).labels_),
            ('DBSCAN, $\\varepsilon=60$', DBSCAN(eps=60, min_samples=5).fit(X).labels_),
            ('HDBSCAN, $m_{\\mathrm{c}}=20$', HDBSCAN(min_cluster_size=20, copy=False).fit(X).labels_),
            ('MeanShift', MeanShift(bandwidth=estimate_bandwidth(X)).fit(X).labels_)]
    fig, axs = plt.subplots(2, 3, figsize=(5.3, 3.9), dpi=240)
    NUM['f3813'] = {}
    for ax, (t, lab) in zip(axs.ravel(), runs):
        nc, nn = int(lab.max() + 1), int((lab == -1).sum()); a = ari(y, lab)
        NUM['f3813'][t] = {'clusters': nc, 'noise': nn, 'ari': round(a, 3)}
        scatter_labels(ax, X, lab, s=5, noise_s=6, lw=0.15)
        ax.set_xlim(-330, 930); ax.set_ylim(-330, 1130); clean(ax)
        ax.add_patch(Rectangle((-70, -60), 250, 120, fc='none', ec=NAVY, lw=0.7, zorder=6))
        ins = ax.inset_axes([0.52, 0.02, 0.45, 0.25])
        zz = (X[:, 0] > -70) & (X[:, 0] < 180) & (X[:, 1] > -60) & (X[:, 1] < 60)
        scatter_labels(ins, X[zz], lab[zz], s=3.5, noise_s=5, lw=0.1)
        ins.set_xlim(-70, 180); ins.set_ylim(-60, 60); clean(ins)
        ins.patch.set_facecolor('white')
        ax.set_title(t, fontsize=9.2, color=NAVY)
        ax.set_xlabel(('' if t == 'truth' else f'{nc} cl., {nn} noise, ARI {a:.2f}') if t != 'truth' else '4 places + 80 stray fixes', fontsize=8.2, color=NAVY, labelpad=2)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.93, bottom=0.07, wspace=0.05, hspace=0.3)
    sv(fig, 'f3813.png', 470)

if ONLY == set() or want('save'):
    json.dump(NUM, open('verify/fig_numbers.json', 'w'), indent=1)
    json.dump(sv.chk, open('verify/fig_textsize.json', 'w'), indent=1)
