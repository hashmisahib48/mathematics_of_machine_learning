# Figures 41.0-1 .. 41.2-1.  Run: python3 fig/figs1.py
from common import *

# ---- 41.0-1: three clusterings of the same four blobs
fig, axs = plt.subplots(1, 3, figsize=(5.4, 2.35), dpi=240, sharex=True, sharey=True)
NUM['open'] = {}
for ax, K in zip(axs, (3, 4, 5)):
    km = KMeans(K, n_init=10, random_state=0).fit(XB); lab = km.labels_
    d = {'sil': silhouette_score(XB, lab), 'ch': chs(XB, lab), 'db': dbs(XB, lab), 'ari': ari(YB, lab), 'ami': ami(YB, lab)}
    NUM['open'][K] = {k: round(float(v), 3) for k, v in d.items()}
    pts(ax, XB, lab, s=5)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='*', s=70, color=GREEN, ec='white', lw=0.6, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'$K$ = {K}', fontsize=9.5, color=NAVY)
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
    ax.set_xlabel(f"silhouette {d['sil']:.2f}   CH {d['ch']:.0f}\nDB {d['db']:.2f}   ARI {d['ari']:.2f}", fontsize=8.5, color=NAVY)
fig.tight_layout(w_pad=0.4); sv(fig, 'f410.png', 470)

# ---- 41.1-1: the eight customers, clusters, segments and the contingency table
fig, axs = plt.subplots(1, 3, figsize=(5.4, 2.0), dpi=240, gridspec_kw={'width_ratios': [1, 1, 0.95]})
for ax, lab, ttl in ((axs[0], CLU, 'clusters (k-means)'), (axs[1], SEG, 'segments (the shop)')):
    pts(ax, X8, lab, s=34)
    for (x, y), nm in zip(X8, NAMES): ax.text(x + 0.25, y + 0.28, nm, fontsize=8.5, color=NAVY)
    ax.set_xlim(0, 9.4); ax.set_ylim(0, 9.4); ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 4, 8]); ax.set_title(ttl, fontsize=9, color=NAVY)
    ax.set_xlabel('visits per month'); ax.set_aspect('equal')
axs[0].set_ylabel('basket (units)')
axs[1].set_yticklabels([])
ax = axs[2]
N = np.zeros((3, 3), int); np.add.at(N, (SEG, CLU), 1)
ax.imshow(N, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('c', ['#FFFFFF', '#9CC0EE']), vmin=0, vmax=4.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(N[i, j]), ha='center', va='center', fontsize=10, color=NAVY, weight='bold' if N[i, j] else 'normal')
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc='none', ec=GRID, lw=0.8))
ax.set_xticks(range(3)); ax.set_xticklabels(['1', '2', '3']); ax.set_yticks(range(3)); ax.set_yticklabels(['1', '2', '3'])
ax.set_xlabel('cluster $j$'); ax.set_ylabel('segment $i$'); ax.set_title('counts $n_{ij}$', fontsize=9, color=NAVY)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(w_pad=0.6); sv(fig, 'f411.png', 470)

dump('fig_numbers1.json', NUM); dump('fig_textsize1.json', CHK)
