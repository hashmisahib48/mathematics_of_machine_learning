# Problem 37C.2: a bridge between two groups, single linkage against Ward. Writes fig/s37c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering as AC

rng = np.random.default_rng(4)                      # the data of the book (fig/probs.py)
Ba = rng.normal([0, 0], 0.5, (40, 2)); Bb = rng.normal([5, 0], 0.5, (40, 2)); bridge = np.c_[np.linspace(1, 4, 7), rng.normal(0, 0.08, 7)]
X = np.r_[Ba, Bb, bridge]; n = len(X)
ls = AC(n_clusters=2, linkage='single').fit_predict(X); lw = AC(n_clusters=2, linkage='ward').fit_predict(X)
print("sizes: single", sorted(np.bincount(ls).tolist()), " Ward", sorted(np.bincount(lw).tolist()))
Dm = squareform(pdist(X)); T = minimum_spanning_tree(Dm).toarray(); ei, ej = np.nonzero(T); w = T[ei, ej]; o = np.argsort(-w)
print("five longest edges of the minimum spanning tree:", np.round(w[o[:5]], 3).tolist())
print("  the longest joins samples", int(ei[o[0]]), "and", int(ej[o[0]]), "- the singleton of single linkage is sample", int(np.argmin(np.bincount(ls)[ls])))
single = int(np.argmin(np.bincount(ls)[ls])); print("  singleton at", X[single].round(3).tolist(), " (group A mean %s)" % Ba.mean(0).round(2).tolist(), " distance to nearest other sample %.3f" % np.sort(Dm[single])[1])
bidx = np.arange(80, 87); gaps = np.diff(bridge[:, 0]); print("bridge spacing in x: %.3f; gaps in the tree between consecutive bridge samples: %s" % (gaps[0], np.round([Dm[i, i+1] for i in bidx[:-1]], 3).tolist()))
print("distance of the bridge ends to the groups: %.3f (to A), %.3f (to B)" % (Dm[bidx[0], :40].min(), Dm[bidx[-1], 40:80].min()))
print("bridge edges in the 10 longest tree edges:", int(sum(((ei[k] >= 80) or (ej[k] >= 80)) for k in o[:10])))
ssw = lambda l: sum(((X[l == c] - X[l == c].mean(0))**2).sum() for c in np.unique(l))
sst = ((X - X.mean(0))**2).sum()
print("SS_total %.2f;  SSW single partition %.2f;  SSW Ward partition %.2f" % (sst, ssw(ls), ssw(lw)))
print("Ward cost of the last merge = SST - SSW: single partition %.2f;  Ward partition %.2f" % (sst - ssw(ls), sst - ssw(lw)))
# Delta for removing the singleton from the rest, and for halving the data
d2 = ((X[single] - np.delete(X, single, 0).mean(0))**2).sum(); print("cost of the singleton: 1*86/87 * %.3f = %.2f" % (d2, 86/87*d2))
nA = (lw == lw[0]).sum(); print("Ward partition: sizes", nA, n - nA, " bridge samples with group A: %d, with group B: %d" % ((lw[80:] == lw[0]).sum(), (lw[80:] == lw[40]).sum()))
print("centre distance squared of the Ward clusters %.3f, factor %d*%d/%d = %.2f" % (((X[lw == lw[0]].mean(0) - X[lw != lw[0]].mean(0))**2).sum(), nA, n-nA, n, nA*(n-nA)/n))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.75), dpi=240, sharey=True)
def draw(ax, lab, title):
    ax.scatter(X[:, 0], X[:, 1], c=np.where(lab == lab[0], BLUE, OR), s=6, lw=0, zorder=3)
    ax.scatter(bridge[:, 0], bridge[:, 1], marker='^', s=14, facecolor='none', edgecolor=NAVY, lw=0.5, zorder=4)
    for a, b_ in zip(ei, ej): ax.plot(X[[a, b_], 0], X[[a, b_], 1], color='#b0b6c0', lw=0.35, zorder=1)
    ax.set_xlim(-1.8, 6.8); ax.set_ylim(-1.7, 1.7); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8.4, pad=3)
    for s in ax.spines.values(): s.set_visible(False)
draw(axs[0], ls, 'single linkage: 1 + 86\n(grey: the spanning tree)'); draw(axs[1], lw, 'Ward: %d + %d' % tuple(sorted(np.bincount(lw))) + '\n(triangles: the bridge)')
axs[0].scatter(*X[single], s=40, facecolor='none', edgecolor=RED, lw=0.9, zorder=6)
fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.80, bottom=0.01)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s37c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s37c2.png') as im: wd = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (wd, fs, fs * 4.9 / wd))
