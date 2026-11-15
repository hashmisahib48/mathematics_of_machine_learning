# All figures of Chapter 39. Run: python3 fig/figs.py [name ...]
# (PNGs into ch39/, numbers into verify/fig_numbers.json)
from common import *
import warnings
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from data import rings, TOY, toy_adjacency, corpus
from sklearn.cluster import (AffinityPropagation, KMeans, SpectralClustering,
                             SpectralCoclustering, SpectralBiclustering)
from sklearn.datasets import (make_blobs, make_moons, make_biclusters,
                              make_checkerboard)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score as ari, consensus_score
from sklearn.metrics.pairwise import rbf_kernel
plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                     'axes.labelsize': 9, 'legend.fontsize': 8.5})
warnings.simplefilter('ignore', UserWarning)       # "graph not fully connected"
warnings.simplefilter('ignore', RuntimeWarning)    # ARPACK fallback
sv = Saver()
NUM = {}
ONLY = set(sys.argv[1:])
want = lambda k: (not ONLY) or (k in ONLY)
X2, y2 = rings()


def lsym_eigs(W, k=8):
    W = W.copy(); np.fill_diagonal(W, 0); d = W.sum(1)
    Ls = np.eye(len(W)) - W / np.sqrt(np.outer(d, d))
    lam, V = eigh(Ls, subset_by_index=[0, k - 1])
    return lam, V / np.sqrt(d)[:, None]


def spectral(X, K, **kw):
    return SpectralClustering(K, random_state=0, **kw).fit(X)


# ------------------------------------------------ 39.0-1 the idea
if want('f390'):
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.05), dpi=240, sharey=True)
    kmm = KMeans(2, n_init=10, random_state=0).fit(X2)
    sp = spectral(X2, 2, gamma=30.0)
    NUM['f390'] = {'ari_kmeans': round(ari(y2, kmm.labels_), 3), 'ari_spectral': round(ari(y2, sp.labels_), 3)}
    for ax, lab, t in ((axs[0], kmm.labels_, 'k-means, $K=2$'), (axs[1], sp.labels_, 'spectral, rbf, $\\gamma=30$')):
        scatter_labels(ax, X2, lab, s=15)
        ax.set_title(t, fontsize=9.5, color=NAVY)
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.3, 1.3); ax.set_aspect('equal')
        ax.set_xticks([-1, 0, 1]); grid(ax); ax.set_xlabel('$x_1$')
    axs[0].set_ylabel('$x_2$')
    axs[0].scatter(*kmm.cluster_centers_.T, s=80, marker='*', color=GREEN, edgecolor='white', lw=0.6, zorder=6)
    fig.tight_layout(w_pad=0.6)
    sv(fig, 'f390.png', 470)

# ------------------------------------------------ 39.1-1 the six points
if want('f391'):
    W = toy_adjacency()
    fig, ax = plt.subplots(figsize=(3.2, 1.75), dpi=240)
    for i in range(6):
        for j in range(i + 1, 6):
            if W[i, j]:
                link = (i, j) == (2, 3)
                ax.plot(*TOY[[i, j]].T, color=RED if link else GREY, lw=1.6 if link else 1.0, ls='--' if link else '-', zorder=1)
    ax.scatter(TOY[:, 0], TOY[:, 1], s=46, color=BLUE, edgecolor='white', lw=0.8, zorder=5)
    off = {1: (-0.55, -0.1), 2: (-0.15, 0.2), 3: (-0.45, -0.62), 4: (0.1, -0.62), 5: (-0.15, 0.2), 6: (0.3, -0.1)}
    for k, (x, yv) in enumerate(TOY, 1):
        ax.text(x + off[k][0], yv + off[k][1], str(k), fontsize=9, color=NAVY, ha='left', va='bottom')
    ax.text(3.5, 0.28, 'link', fontsize=8.5, color=RED, ha='center')
    ax.set_xlim(-1, 8); ax.set_ylim(-1, 3); ax.set_aspect('equal'); ax.set_xticks(range(0, 8, 2)); ax.set_yticks([0, 2])
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); grid(ax)
    sv(fig, 'f391.png', 330)

# ------------------------------------------------ 39.2-1 three cuts and the relaxed vector
if want('f392'):
    W = toy_adjacency(); d = W.sum(1); L = np.diag(d) - W
    lam, F = eigh(L, np.diag(d)); f = F[:, 1] / F[0, 1]
    cuts = [([0, 1, 2], 'cut (1,2,3 | 4,5,6)'), ([0, 1], 'cut (1,2 | 3,4,5,6)'), ([0], 'cut (1 | 2,...,6)')]
    fig, axs = plt.subplots(2, 2, figsize=(5.2, 3.05), dpi=240)
    info = []
    for ax, (A, t) in zip(axs.ravel()[:3], cuts):
        B = [i for i in range(6) if i not in A]
        cut = W[np.ix_(A, B)].sum(); vA, vB = d[A].sum(), d[B].sum(); nc = cut * (1 / vA + 1 / vB)
        info.append({'A': [a + 1 for a in A], 'cut': float(cut), 'volA': float(vA), 'volB': float(vB), 'ncut': round(float(nc), 4)})
        for i in range(6):
            for j in range(i + 1, 6):
                if W[i, j]:
                    cutedge = (i in A) != (j in A)
                    ax.plot(*TOY[[i, j]].T, color=RED if cutedge else GREY, lw=1.8 if cutedge else 0.9, ls='--' if cutedge else '-', zorder=1)
        for i in range(6):
            ax.scatter(*TOY[i], s=44, color=BLUE if i in A else OR, marker='o' if i in A else 's', edgecolor='white', lw=0.7, zorder=5)
            ax.text(TOY[i, 0], TOY[i, 1] + 0.42, str(i + 1), fontsize=8.5, color=NAVY, ha='center')
        ax.set_title(t, fontsize=9.2, color=NAVY)
        ax.text(3.5, -1.05, f'cut = {cut:.0f},  Ncut = {nc:.3f}', fontsize=9, color=RED, ha='center')
        ax.set_xlim(-1, 8); ax.set_ylim(-1.5, 3.2); ax.set_aspect('equal'); ax.axis('off')
    ax = axs[1, 1]
    cols = [BLUE if v > 0 else OR for v in f]
    ax.bar(range(1, 7), f, color=cols, width=0.65)
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_xticks(range(1, 7)); ax.set_xlabel('node'); ax.set_ylabel('$f_i$')
    ax.set_title('relaxed vector, $\\lambda_2=%.4f$' % lam[1], fontsize=9.2, color=NAVY)
    ax.set_ylim(-1.3, 1.3); grid(ax)
    NUM['f392'] = {'cuts': info, 'lambda2': round(float(lam[1]), 4), 'f': f.round(4).tolist()}
    fig.tight_layout(h_pad=0.5, w_pad=0.6)
    sv(fig, 'f392.png', 470)

# ------------------------------------------------ 39.3-1 three rings, embedding, spectrum
if want('f393'):
    X3, y3 = make_blobs([70, 90, 110], centers=[(0, 0), (4.5, 0.5), (2, 4.5)], cluster_std=[0.6, 0.8, 0.9], random_state=3)
    g = 0.3
    Wr = rbf_kernel(X3, gamma=g)
    lam, U = lsym_eigs(Wr, 8)
    lab3 = spectral(X3, 3, gamma=g).labels_
    NUM['f393'] = {'gamma': g, 'eigenvalues': lam.round(5).tolist(), 'ari_spectral': round(ari(y3, lab3), 3),
                   'ari_kmeans': round(ari(y3, KMeans(3, n_init=10, random_state=0).fit_predict(X3)), 3),
                   'u_range': [round(float(np.ptp(U[:, 1])), 4), round(float(np.ptp(U[:, 2])), 4)],
                   'u1_range': round(float(np.ptp(U[:, 0])), 12), 'n': len(X3)}
    fig, axs = plt.subplots(1, 3, figsize=(5.3, 1.75), dpi=240, gridspec_kw={'width_ratios': [1, 1, 1.05]})
    ax = axs[0]
    for k in range(3):
        ax.scatter(*X3[y3 == k].T, s=7, marker=MK[k], color=CL[k], edgecolor='none')
    ax.set_aspect('equal'); ax.set_xticks([0, 3, 6]); ax.set_yticks([0, 3, 6]); grid(ax)
    ax.set_title('three groups', fontsize=9.2, color=NAVY)
    ax = axs[1]
    for k in range(3):
        ax.scatter(U[y3 == k, 1], U[y3 == k, 2], s=8, marker=MK[k], color=CL[k], edgecolor='none')
    ax.set_xlabel('$u_2$'); ax.set_ylabel('$u_3$'); grid(ax)
    ax.set_title('rows of the embedding', fontsize=9.2, color=NAVY)
    ax = axs[2]
    ax.bar(range(1, 9), lam, color=[PUR] * 3 + [GREY] * 5, width=0.65)
    ax.set_xticks(range(1, 9)); ax.set_xlabel('index $m$'); ax.set_ylabel('$\\lambda_m$'); grid(ax)
    ax.set_title('eigenvalues of $\\mathbf{L}_{\\mathrm{sym}}$', fontsize=9.2, color=NAVY)
    fig.tight_layout(w_pad=0.7)
    sv(fig, 'f393.png', 470)

# ------------------------------------------------ 39.4-1 affinity propagation
if want('f394'):
    Xb, yb = make_blobs(90, centers=3, cluster_std=0.9, random_state=4)
    S = -((Xb[:, None] - Xb[None]) ** 2).sum(-1)
    med = float(np.median(S))
    prefs = -np.logspace(-1, 2.3, 22)
    cnt = []
    for p in prefs:
        cnt.append(len(AffinityPropagation(preference=p, random_state=0, max_iter=400).fit(Xb).cluster_centers_indices_))
    ap1 = AffinityPropagation(random_state=0).fit(Xb)
    pk = -120.0
    ap2 = AffinityPropagation(preference=pk, random_state=0).fit(Xb)
    NUM['f394'] = {'median': round(med, 3), 'k_default': int(len(ap1.cluster_centers_indices_)), 'ari_default': round(ari(yb, ap1.labels_), 3),
                   'p_low': pk, 'k_low': int(len(ap2.cluster_centers_indices_)), 'ari_low': round(ari(yb, ap2.labels_), 3),
                   'prefs': prefs.round(3).tolist(), 'counts': cnt}
    fig, axs = plt.subplots(2, 2, figsize=(5.2, 3.45), dpi=240)
    # (a) the messages of Example 39.6-2
    ax = axs[0, 0]
    xs = np.array([0.0, 1.0, 3.0]); ys = np.zeros(3)
    ax.scatter(xs, ys, s=60, color=BLUE, edgecolor='white', zorder=5)
    for k, (x, lab) in enumerate(zip(xs, ('1', '2', '3'))):
        ax.text(x, -0.55, lab, ha='center', fontsize=9.5, color=NAVY)
    ax.add_patch(FancyArrowPatch((0.0, 0.15), (0.92, 0.15), arrowstyle='-|>', mutation_scale=9, color=BLUE, lw=1.3, connectionstyle='arc3,rad=-0.55'))
    ax.add_patch(FancyArrowPatch((1.08, -0.15), (0.08, -0.15), arrowstyle='-|>', mutation_scale=9, color=GREEN, lw=1.3, connectionstyle='arc3,rad=-0.55'))
    ax.add_patch(FancyArrowPatch((3.0, 0.15), (1.1, 0.15), arrowstyle='-|>', mutation_scale=9, color=BLUE, lw=1.3, connectionstyle='arc3,rad=0.35'))
    ax.add_patch(FancyArrowPatch((1.1, -0.15), (2.92, -0.15), arrowstyle='-|>', mutation_scale=9, color=GREEN, lw=1.3, connectionstyle='arc3,rad=0.35'))
    ax.text(0.5, 0.7, '$r(1,2)=5$', fontsize=8.8, color=BLUE, ha='center')
    ax.text(0.5, -1.45, '$a(1,2)=-3$', fontsize=8.8, color=GREEN, ha='center')
    ax.text(2.1, 1.05, '$r(3,2)=2$', fontsize=8.8, color=BLUE, ha='center')
    ax.text(2.1, -1.45, '$a(3,2)=0$', fontsize=8.8, color=GREEN, ha='center')
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(-1.9, 1.5); ax.axis('off')
    ax.set_title('round 1 of Example 39.6-2', fontsize=9.2, color=NAVY)
    for ax, ap, t in ((axs[0, 1], ap1, 'median preference'), (axs[1, 0], ap2, '$p=%d$' % pk)):
        lab = ap.labels_; ex = ap.cluster_centers_indices_
        for k in range(len(ex)):
            m = lab == k
            for i in np.flatnonzero(m):
                ax.plot([Xb[i, 0], Xb[ex[k], 0]], [Xb[i, 1], Xb[ex[k], 1]], color='#BBBBBB', lw=0.5, zorder=1)
            ax.scatter(*Xb[m].T, s=11, marker=MK[k % 8], color=CL[k % 8], edgecolor='none', zorder=3)
        ax.scatter(*Xb[ex].T, s=70, marker='*', color=GREEN, edgecolor='white', lw=0.5, zorder=6)
        ax.set_title('%s: %d exemplars' % (t, len(ex)), fontsize=9.2, color=NAVY)
        ax.set_xlim(Xb[:, 0].min() - 0.5, Xb[:, 0].max() + 0.5); ax.set_ylim(Xb[:, 1].min() - 0.5, Xb[:, 1].max() + 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_visible(True); s_.set_linewidth(0.5); s_.set_color('#999999')
    ax = axs[1, 1]
    ax.semilogx(-prefs, cnt, '-o', ms=3, color=PUR, lw=1.2)
    ax.axvline(-med, color=GREY, ls='--', lw=0.9); ax.text(-med * 1.15, 55, 'median', fontsize=8.5, color=GREY, rotation=90, va='top')
    ax.set_xlabel('$-p$ (preference)'); ax.set_ylabel('exemplars'); grid(ax)
    ax.set_title('number of clusters', fontsize=9.2, color=NAVY)
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    sv(fig, 'f394.png', 470)

# ------------------------------------------------ 39.5-1 biclustering
if want('f395'):
    rng = np.random.default_rng(0)
    A1, rows1, cols1 = make_biclusters((80, 80), 3, noise=5, shuffle=False, random_state=0)
    perm_r = np.random.default_rng(1).permutation(80); perm_c = np.random.default_rng(2).permutation(80)
    A1s = A1[perm_r][:, perm_c]
    co = SpectralCoclustering(3, random_state=0).fit(A1s)
    cs1 = consensus_score(co.biclusters_, (rows1[:, perm_r], cols1[:, perm_c]))
    A1o = A1s[np.argsort(co.row_labels_)][:, np.argsort(co.column_labels_)]
    A2, rows2, cols2 = make_checkerboard((80, 80), (4, 3), noise=10, shuffle=False, random_state=0)
    A2s = A2[perm_r][:, perm_c]
    bi = SpectralBiclustering((4, 3), method='log', random_state=0).fit(A2s)
    cs2 = consensus_score(bi.biclusters_, (rows2[:, perm_r], cols2[:, perm_c]))
    A2o = A2s[np.argsort(bi.row_labels_)][:, np.argsort(bi.column_labels_)]
    NUM['f395'] = {'consensus_coclustering': round(cs1, 3), 'consensus_biclustering': round(cs2, 3)}
    fig, axs = plt.subplots(1, 4, figsize=(5.3, 1.55), dpi=240)
    for ax, M, t in ((axs[0], A1s, 'blocks,\nshuffled'), (axs[1], A1o, 'co-clustering,\nscore %.2f' % cs1),
                     (axs[2], A2s, 'checkerboard,\nshuffled'), (axs[3], A2o, 'biclustering,\nscore %.2f' % cs2)):
        ax.imshow(M, cmap='Blues', aspect='equal', interpolation='nearest')
        ax.set_title(t, fontsize=8.8, color=NAVY); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_visible(True); s_.set_linewidth(0.5); s_.set_color('#999999')
    axs[0].set_ylabel('rows'); 
    for ax in axs: ax.set_xlabel('columns')
    fig.tight_layout(w_pad=0.4)
    sv(fig, 'f395.png', 470)

# ------------------------------------------------ 39.7-1 the algorithms as flow charts
if want('f397'):
    rows = [('Spectral clustering', ['affinity\n$\\mathbf{W}$', 'degrees,\n$\\mathbf{L}_{\\mathrm{sym}}$', '$K$ smallest\neigenvectors', 'rows\n$\\mathbf{u}_i$', 'k-means\non rows'], BLUE),
            ('Affinity propagation', ['similarities\n$s(i,k)$', 'responsibility\n$r(i,k)$', 'availability\n$a(i,k)$', 'damp,\nrepeat', 'exemplars:\n$a_{kk}+r_{kk}>0$'], GREEN),
            ('Spectral co-clustering', ['normalise\n$\\mathbf{A}_n$', 'SVD of\n$\\mathbf{A}_n$', 'stack\n$\\mathbf{Z}$ (Eq. 39.5-4)', 'k-means on\nrows of $\\mathbf{Z}$', 'split rows\nand columns'], PUR)]
    fig, ax = plt.subplots(figsize=(5.3, 2.5), dpi=240)
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis('off')
    bw, gap, bh = 17.0, 3.7, 10.6
    ys = [38.5, 21.5, 4.5]
    for r_, (name, items, col) in enumerate(rows):
        y0 = ys[r_]
        ax.text(0, y0 + bh + 2.4, name, fontsize=9.5, color=col, weight='bold', va='center')
        for k, t in enumerate(items):
            x0 = k * (bw + gap)
            ax.add_patch(FancyBboxPatch((x0, y0), bw, bh, boxstyle='round,pad=0.0,rounding_size=1.2', fc='white', ec=col, lw=1.1))
            ax.text(x0 + bw / 2, y0 + bh / 2, t, fontsize=8.0, color=NAVY, ha='center', va='center', linespacing=1.1)
            if k:
                ax.add_patch(FancyArrowPatch((x0 - gap + 0.1, y0 + bh / 2), (x0 - 0.1, y0 + bh / 2), arrowstyle='-|>', mutation_scale=7, color=GREY, lw=0.9))
    sv(fig, 'f397.png', 470)

# ------------------------------------------------ 39.10-1 gamma and neighbours gallery
if want('f3910a'):
    gs = [1, 10, 30, 1000]; ks = [3, 5, 15, 50]
    fig, axs = plt.subplots(2, 4, figsize=(5.3, 2.55), dpi=240)
    NUM['f3910a'] = {'rbf': {}, 'knn': {}}
    for j, g in enumerate(gs):
        W = rbf_kernel(X2, gamma=g); ncomp = connected_components(W > 1e-8)[0]
        sp = spectral(X2, 2, gamma=g)
        a = ari(y2, sp.labels_); NUM['f3910a']['rbf'][str(g)] = {'ari': round(a, 3), 'components_above_1e-8': int(ncomp)}
        ax = axs[0, j]; scatter_labels(ax, X2, sp.labels_, s=5); clean(ax)
        ax.set_title('$\\gamma=%g$' % g, fontsize=9, color=NAVY); ax.text(0, -1.55, 'ARI %.2f' % (round(a, 2) + 0.0), fontsize=8.8, color=RED, ha='center', va='top')
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal')
    for j, k in enumerate(ks):
        sp = spectral(X2, 2, affinity='nearest_neighbors', n_neighbors=k)
        nc = connected_components(sp.affinity_matrix_)[0]
        a = ari(y2, sp.labels_); NUM['f3910a']['knn'][str(k)] = {'ari': round(a, 3), 'components': int(nc)}
        ax = axs[1, j]; scatter_labels(ax, X2, sp.labels_, s=5); clean(ax)
        ax.set_title('$k=%d$' % k, fontsize=9, color=NAVY); ax.text(0, -1.55, 'ARI %.2f' % (round(a, 2) + 0.0), fontsize=8.8, color=RED, ha='center', va='top')
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal')
    fig.tight_layout(h_pad=1.2, w_pad=0.3)
    sv(fig, 'f3910a.png', 470)

# ------------------------------------------------ 39.10-2 the validation curves
if want('f3910b'):
    gs = np.logspace(-1, 3, 17); ks = [3, 4, 5, 7, 10, 15, 20, 30, 40, 60, 100]
    ar, gap, nc = [], [], []
    for g in gs:
        sp = spectral(X2, 2, gamma=g); ar.append(ari(y2, sp.labels_))
        lam, _ = lsym_eigs(rbf_kernel(X2, gamma=g), 4); gap.append(lam[2] - lam[1])
    ar_k, nc_k = [], []
    for k in ks:
        sp = spectral(X2, 2, affinity='nearest_neighbors', n_neighbors=k); ar_k.append(ari(y2, sp.labels_)); nc_k.append(connected_components(sp.affinity_matrix_)[0])
    kmari = ari(y2, KMeans(2, n_init=10, random_state=0).fit_predict(X2))
    NUM['f3910b'] = {'gamma': gs.round(3).tolist(), 'ari_gamma': np.round(ar, 3).tolist(), 'gap': np.round(gap, 4).tolist(),
                     'k': ks, 'ari_k': np.round(ar_k, 3).tolist(), 'components_k': [int(v) for v in nc_k], 'ari_kmeans': round(kmari, 3)}
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.2), dpi=240)
    ax = axs[0]
    ax.semilogx(gs, ar, '-o', ms=3, color=PUR, lw=1.3)
    ax.axhline(kmari, color=GREY, ls='--', lw=0.9); ax.text(0.12, kmari + 0.05, 'k-means', fontsize=8.5, color=GREY)
    ax.set_xlabel('$\\gamma$ (rbf)'); ax.set_ylabel('ARI'); ax.set_ylim(-0.1, 1.1); grid(ax)
    ax = axs[1]
    ax.plot(ks, ar_k, '-o', ms=3, color=PUR, lw=1.3, label='ARI')
    ax2 = ax.twinx(); ax2.step(ks, nc_k, where='mid', color=RED, lw=1.0, label='components'); ax2.set_ylabel('graph components', color=RED); ax2.set_yscale('log'); ax2.set_ylim(0.8, 60); ax2.set_yticks([1, 2, 5, 10, 20]); ax2.set_yticklabels(['1', '2', '5', '10', '20']); ax2.minorticks_off(); ax2.spines['right'].set_visible(True); ax2.tick_params(axis='y', colors=RED)
    ax.set_xscale('log'); ax.set_xlabel('$k$ (n_neighbors)'); ax.set_ylabel('ARI'); ax.set_ylim(-0.1, 1.1); grid(ax)
    fig.tight_layout(w_pad=0.8)
    sv(fig, 'f3910b.png', 470)

# ------------------------------------------------ 39.10-3 four methods on three data sets
if want('f3910c'):
    sets = [('two rings', rings(), 30.0), ('two crescents', make_moons(300, noise=0.06, random_state=0), 30.0),
            ('three blobs', make_blobs(300, centers=3, cluster_std=1.0, random_state=2), 1.0)]
    cols = ['k-means', 'spectral, rbf', 'spectral, 15 nn', 'affinity prop.']
    fig, axs = plt.subplots(3, 4, figsize=(5.3, 3.7), dpi=240)
    NUM['f3910c'] = {}
    for i, (name, (X, y), g) in enumerate(sets):
        K = len(set(y))
        labs = [KMeans(K, n_init=10, random_state=0).fit(X).labels_, spectral(X, K, gamma=g).labels_,
                spectral(X, K, affinity='nearest_neighbors', n_neighbors=15).labels_, AffinityPropagation(random_state=0).fit(X).labels_]
        NUM['f3910c'][name] = {'gamma': g, 'ari': [round(ari(y, l), 3) for l in labs], 'k_ap': int(labs[3].max() + 1)}
        for j, l in enumerate(labs):
            ax = axs[i, j]; scatter_labels(ax, X, l, s=4, lw=0.2); clean(ax); ax.set_aspect('equal' if i < 2 else 'auto')
            ax.set_title(('%s\n' % cols[j] if i == 0 else '') + 'ARI %.2f' % (round(ari(y, l), 2) + 0.0) + (', $K$=%d' % (l.max() + 1) if j == 3 else ''), fontsize=8.6, color=NAVY if i else NAVY)
        axs[i, 0].set_ylabel(name, fontsize=9, color=NAVY)
    fig.tight_layout(h_pad=0.6, w_pad=0.2)
    sv(fig, 'f3910c.png', 470)

# ------------------------------------------------ 39.11-1 the corpus
if want('f3911'):
    docs, topic, truth = corpus()
    cv = CountVectorizer(stop_words='english', min_df=2); X = cv.fit_transform(docs).toarray()
    words = np.array(cv.get_feature_names_out())
    co = SpectralCoclustering(4, random_state=0).fit(X)
    ro = np.argsort(co.row_labels_, kind='stable'); cc = np.argsort(co.column_labels_, kind='stable')
    Ks = [2, 3, 4, 5, 6]; aris = []; ins = []
    for K in Ks:
        m = SpectralCoclustering(K, random_state=0).fit(X)
        aris.append(ari(topic, m.row_labels_))
        ins.append(sum(X[m.row_labels_ == k][:, m.column_labels_ == k].sum() for k in range(K)) / X.sum())
    NUM['f3911'] = {'shape': list(X.shape), 'Ks': Ks, 'ari': np.round(aris, 3).tolist(), 'inside': np.round(ins, 3).tolist()}
    fig, axs = plt.subplots(1, 3, figsize=(5.3, 1.75), dpi=240, gridspec_kw={'width_ratios': [1, 1, 1.05]})
    for ax, M, t in ((axs[0], X, 'as generated'), (axs[1], X[ro][:, cc], 'rows, columns sorted')):
        ax.imshow(M > 0, cmap='Blues', aspect='auto', interpolation='nearest')
        ax.set_title(t, fontsize=9.2, color=NAVY); ax.set_xlabel('words (107)'); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values(): s_.set_visible(True); s_.set_linewidth(0.5); s_.set_color('#999999')
    axs[0].set_ylabel('documents (80)')
    ax = axs[2]
    ax.plot(Ks, aris, '-o', ms=3, color=PUR, lw=1.3, label='ARI of rows')
    ax.plot(Ks, ins, '-s', ms=3, color=GREEN, lw=1.1, label='mass in blocks')
    ax.set_xlabel('$K$'); ax.set_ylim(0.3, 1.05); ax.set_xticks(Ks); grid(ax); ax.legend(loc='lower right', handlelength=1.3, labelspacing=0.3, borderaxespad=0.2)
    ax.set_title('choosing $K$', fontsize=9.2, color=NAVY)
    fig.tight_layout(w_pad=0.6)
    sv(fig, 'f3911.png', 470)

import json
path = 'verify/fig_numbers.json'
old = json.load(open(path)) if os.path.exists(path) else {}
old.update(NUM); json.dump(old, open(path, 'w'), indent=1)
json.dump(sv.chk if not os.path.exists('verify/fig_textsize.json') else {**json.load(open('verify/fig_textsize.json')), **sv.chk}, open('verify/fig_textsize.json', 'w'), indent=1)
print('figs ok')
