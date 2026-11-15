# Problem 38C.2: one Gaussian blob and HDBSCAN. Writes fig/s38c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch38/code'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.cluster import HDBSCAN
from hdbscan_tree import hdbscan_scratch

X = np.random.default_rng(0).normal(size=(150, 2)); n = 150          # the data of the book
h = HDBSCAN(min_cluster_size=10, copy=False).fit(X)
h1 = HDBSCAN(min_cluster_size=10, allow_single_cluster=True, copy=False).fit(X)
print("default:               clusters %d, sizes %s, noise %d" % (h.labels_.max() + 1, np.bincount(h.labels_[h.labels_ >= 0]).tolist(), (h.labels_ == -1).sum()))
print("allow_single_cluster:  clusters %d, sizes %s, noise %d" % (h1.labels_.max() + 1, np.bincount(h1.labels_[h1.labels_ >= 0]).tolist(), (h1.labels_ == -1).sum()))
t = h._single_linkage_tree_                                             # private attribute, as in the chapter's figure script
res = hdbscan_scratch(X, 10, 10, 'eom', Z=np.c_[t['left_node'], t['right_node'], t['value'], t['cluster_size']])
assert np.array_equal(res['labels'], h.labels_)
S, kids, rows = res['S'], res['kids'], res['rows']
print("the single cluster consists of the same 36 samples as the two clusters:", bool(((h1.labels_ >= 0) == (h.labels_ >= 0)).all()))
print("condensed tree: clusters", sorted(S), " children of the root", kids[n])
print("stability of the root %.1f, of its children %s; sum of the children %.2f" % (S[n], [round(S[k], 2) for k in kids[n]], sum(S[k] for k in kids[n])))
kid_rows = [(c, 1 / l, s) for p, c, l, s in rows if p == n and c >= n]
print("the root splits at eps = %.3f into clusters of %s samples" % (kid_rows[0][1], [r[2] for r in kid_rows]))
drop = [l for p, c, l, s in rows if p == n and c < n]
print("samples that fall out of the root one by one before the split: %d (eps from %.3f down to %.3f)" % (len(drop), 1 / min(drop), 1 / max(drop)))
print("noise of the default run = those samples:", (h.labels_ == -1).sum() == len(drop))
print("with the root allowed, EOM keeps the root if %.1f >= %.2f: %s" % (S[n], sum(S[k] for k in kids[n]), S[n] >= sum(S[k] for k in kids[n])))
r = np.linalg.norm(X, axis=1)
print("radius of the samples: clustered median %.2f, noise median %.2f; 36 clustered samples lie within radius %.2f of the centre" % (np.median(r[h.labels_ >= 0]), np.median(r[h.labels_ == -1]), r[h.labels_ >= 0].max()))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.0), dpi=240, sharey=True)
for ax, lab, title in ((axs[0], h.labels_, 'default: 2 clusters, 114 noise'), (axs[1], h1.labels_, 'allow_single_cluster=True:\n1 cluster, 114 noise')):
    m = lab == -1
    ax.scatter(X[m, 0], X[m, 1], marker='x', s=9, color='#7A7A7A', lw=0.7, zorder=2)
    for k, (col, mk) in enumerate(zip([BLUE, OR], ['o', 's'])):
        q = lab == k; ax.scatter(X[q, 0], X[q, 1], s=12, color=col, marker=mk, edgecolor='white', lw=0.3, zorder=4)
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-3, 3); ax.set_aspect('equal'); ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2]); ax.set_title(title, fontsize=8.4, pad=3)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylabel('$x_2$'); [a.set_xlabel('$x_1$') for a in axs]
fig.subplots_adjust(wspace=0.06, left=0.08, right=0.99, top=0.84, bottom=0.17)
fs = min(t_.get_fontsize() for t_ in fig.findobj(mtext.Text) if t_.get_text().strip() and t_.get_visible())
fig.savefig('fig/s38c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s38c2.png') as im: wd = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (wd, fs, fs * 4.9 / wd))
