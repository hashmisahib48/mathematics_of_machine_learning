# Figures 41.6-1, 41.10-1, 41.11-1.  Run: python3 fig/figs3.py   (needs verify/stability.out)
from common import *
NUM = {}

# ---- 41.6-1: the two routes to a score
fig, ax = plt.subplots(figsize=(4.7, 2.3), dpi=240); ax.axis('off'); ax.set_xlim(-0.1, 11.3); ax.set_ylim(0.55, 4.9)
def bx(x, y, w, h, t, fc, ec, tc=NAVY, fs=8.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=tc, linespacing=1.15)
def ar(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, color=GREY, lw=0.9))
rows = [(3.35, 'no true classes needed', ['data $\\mathbf{X}$,\nclusters $u$', 'distances\n(all pairs or\nto centroids)', '$a(i)$, $b(i)$;\nSSB, SSW;\n$s_k$, $d_{kl}$', 'silhouette\nCH, DB']),
        (0.75, 'true classes needed', ['classes $c$,\nclusters $u$', 'contingency\ntable $n_{ij}$\n(one pass)', 'pair counts;\nentropies;\nexpected values', 'RI, ARI, FM\nMI, NMI, AMI\nH, C, V'])]
fills = ['#F1EAF7', LB, LB, '#E3F4EA']; edges = [PUR, BLUE, BLUE, GREEN]; W_ = 2.2; STEP = 2.5
for y, ttl, txt in rows:
    ax.text(0.0, y + 1.28, ttl, fontsize=8.5, color=NAVY, style='italic', va='bottom')
    for k, t in enumerate(txt):
        x = k * STEP; bx(x, y, W_, 1.2, t, fills[k], edges[k])
        if k: ar(x - STEP + W_ + 0.03, y + 0.6, x - 0.03, y + 0.6)
bx(9.95, 1.35, 1.3, 2.2, 'compare\n$K$ and\nmethods;\ncheck\nstability\n(§41.10)', '#FDF0E0', OR)
ar(3 * STEP + W_ + 0.03, 3.95, 9.92, 3.2); ar(3 * STEP + W_ + 0.03, 1.35, 9.92, 1.75)
sv(fig, 'f461.png', 400)

# ---- 41.11-1: K scan on wine and blobs
Xw, yw = load_wine(return_X_y=True); from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(Xw)
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.3), dpi=240, sharey=True)
NUM['scan'] = {}
for ax, (nm_, X, y, Kt) in zip(axs, (('wine, standardised', Xs, yw, 3), ('four blobs', XB, YB, 4))):
    Ks = list(range(2, 9)); S = []; C = []; Db = []; A = []
    for K in Ks:
        lab = KMeans(K, n_init=10, random_state=0).fit_predict(X)
        S.append(silhouette_score(X, lab)); C.append(chs(X, lab)); Db.append(dbs(X, lab)); A.append(ari(y, lab))
    NUM['scan'][nm_] = {'K': Ks, 'sil': [round(v, 3) for v in S], 'ch': [round(v, 1) for v in C], 'db': [round(v, 3) for v in Db], 'ari': [round(v, 3) for v in A],
                        'best_sil': Ks[int(np.argmax(S))], 'best_ch': Ks[int(np.argmax(C))], 'best_db': Ks[int(np.argmin(Db))], 'best_ari': Ks[int(np.argmax(A))]}
    ax.axvline(Kt, color=PUR, lw=1.0, ls=':')
    ax.plot(Ks, S, '-o', color=BLUE, ms=3.5, lw=1.4, label='silhouette')
    ax.plot(Ks, np.array(C) / max(C), '-^', color=TEAL, ms=3.8, lw=1.4, label='CH / max CH')
    ax.plot(Ks, A, '--s', color=GREY, ms=3.5, lw=1.4, label='ARI')
    ax.set_title(nm_, fontsize=9, color=NAVY); ax.set_xlabel('$K$'); ax.set_ylim(0, 1.05); ax.set_xticks(Ks); grid(ax)
axs[0].legend(loc='upper right', fontsize=8.5, handlelength=1.8, borderaxespad=0.1, labelspacing=0.2)
fig.tight_layout(w_pad=0.7); sv(fig, 'f4111.png', 470)

# ---- 41.10-1: the moons, and stability
km = KMeans(2, n_init=10, random_state=0).fit_predict(XM)
fig, axs = plt.subplots(1, 3, figsize=(5.4, 2.1), dpi=240, gridspec_kw={'width_ratios': [1, 1, 1.25]})
NUM['moons'] = {}
for ax, lab, ttl in ((axs[0], YM, 'true halves'), (axs[1], km, 'k-means')):
    d = {'sil': silhouette_score(XM, lab), 'ch': chs(XM, lab), 'db': dbs(XM, lab), 'ari': ari(YM, lab)}
    NUM['moons'][ttl] = {k: round(float(v), 3) for k, v in d.items()}
    pts(ax, XM, lab, s=6); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=9, color=NAVY)
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    ax.set_xlabel(f"silhouette {d['sil']:.2f}\nARI {d['ari']:.2f}", fontsize=8.5, color=NAVY)
rows = [l.split() for l in open('verify/stability.out').read().strip().split('\n')[1:]]
Ks = [int(r[0]) for r in rows]; sw = [float(r[1]) for r in rows]; sb = [float(r[2]) for r in rows]
NUM['stability'] = {'K': Ks, 'wine': sw, 'blobs': sb}
ax = axs[2]
ax.plot(Ks, sw, '-o', color=BLUE, ms=3.5, lw=1.4, label='wine'); ax.plot(Ks, sb, '--s', color=OR, ms=3.5, lw=1.4, label='blobs')
ax.set_xlabel('$K$'); ax.set_ylabel('stability (mean ARI)'); ax.set_ylim(0.4, 1.03); ax.set_xticks(Ks); grid(ax)
ax.legend(loc='lower left', fontsize=8.5, handlelength=1.8, borderaxespad=0.1, labelspacing=0.2)
fig.tight_layout(w_pad=0.6); sv(fig, 'f4101.png', 470)
dump('fig_numbers3.json', NUM); dump('fig_textsize3.json', CHK)
